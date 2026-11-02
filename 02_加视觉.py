# -*- coding: utf-8 -*-
"""02 加视觉 + 力控：相机拍照 → 找红色方块 → 反投影 → 运动到位 → 力控夹持

一条完整链路：视觉找位置 → 逆运动学过去 → 力控夹住 → 抬起。

三种夹持方式（同一个脚本内切换，不维护多份副本）：
    --closed-loop   P 反馈闭环（默认）：按实测接触力动态补偿，抗滑移
    --open-loop     开环基准组：夹持力固定，不做反馈
    --position      原版：位置控制合拢（保留下来做回归，验证视觉链路没被改坏）

运行：
    .venv\\Scripts\\python.exe 02_加视觉.py                # 闭环（默认）
    .venv\\Scripts\\python.exe 02_加视觉.py --open-loop    # 基准组
    .venv\\Scripts\\python.exe 02_加视觉.py --position     # 原版位置控制

环境变量：
    GRIPPER_HEADLESS=1      不开仿真窗口（相机切 TINY_RENDERER）
    GRIPPER_FAST=1          不按实时节奏等待
    GRIPPER_NO_SHOW=1       不弹窗（也认 GRIPPER_NO_SHM），只存文件
    GRIPPER_CLOSED_LOOP=1   等价于 --closed-loop；=0 等价于 --open-loop
    GRIPPER_SEED=1234       固定方块随机位置，方便开环/闭环一对一对比

产物（文件名带模式标记，不会互相覆盖）：
    视觉抓取_<模式>.log     控制台全文（UTF-8）
    视觉抓取_<模式>.png     力曲线（仅力控两种模式）
"""

import math
import os
import sys
import time
import random

import numpy as np
import cv2
import pybullet as p
import pybullet_data

# ---------------- 视觉（沿用原版，未改动） ----------------
W, H = 640, 480                 # 相机分辨率
FOV = 55                        # 垂直视场角（度）
CAM_POS = [0.30, -0.55, 0.80]   # 相机位置
CAM_TARGET = [0.55, 0.0, 0.05]  # 相机看向哪里
PLANE_Z = 0.02                  # 方块中心在大约这个高度（用于射线与平面求交）

# ---------------- 夹持 ----------------
FINGER_OPEN = 0.04
FINGER_CLOSE = 0.0
FORCE = 200                     # 位置控制模式下手指电机的力上限
APPROACH_Z = 0.16
GRASP_Z = 0.015
LIFT_Z = 0.22
ARM_FORCE = 500

TARGET_FORCE = 0.40             # 目标夹持力 (N)：低于理论阈值 0.49 N，开环会滑
KP_FORCE = 0.5                  # 力环比例增益（对象增益≈1，必须 < 1）
K_SLIP = 0.05                   # 滑移补偿增益 (N/mm)
MAX_CMD = 5.0                   # 输出上限 (N)
MAX_CMD_RATE = 30.0             # 输出变化率上限 (N/s)
MEAS_ALPHA = 0.3                # 实测力低通系数
T_RAMP = 0.20                   # 夹持力爬升时间 (s)

T_GRASP = 0.45                  # 合拢阶段 (s)
T_LIFT = 0.70                   # 抬起阶段 (s)
T_HOLD = 0.45                   # 保持阶段 (s)

SLIP_RISE_MM = 40.0             # 抬升不足这个高度算滑移
SLIP_DETECT_MM = 5.0            # 夹爪-方块位移差超过这个值算"开始打滑"
MU = 1.0
FINGER_DAMPING = 10.0
PRE_CLOSE = 0.032

TIME_STEP = 1.0 / 500.0
DISPLAY_SEC = 3.0               # 相机画面显示多久（到点自动关，按键可提前）

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

SHOW = not (os.environ.get("GRIPPER_NO_SHOW") == "1" or os.environ.get("GRIPPER_NO_SHM") == "1")
USE_GUI = os.environ.get("GRIPPER_HEADLESS") != "1"
PAUSE = os.environ.get("GRIPPER_FAST") != "1"
SEED = os.environ.get("GRIPPER_SEED")

import matplotlib                                  # noqa: E402
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------
# 输出：同时写控制台和日志文件
# ---------------------------------------------------------------
class Log:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.file.write(str(msg) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# ---------------------------------------------------------------
# 相机：像素 → 世界坐标（沿用原版，未改动）
# ---------------------------------------------------------------
def camera_basis():
    cam = np.array(CAM_POS, dtype=float)
    tgt = np.array(CAM_TARGET, dtype=float)
    f = (tgt - cam)
    f /= np.linalg.norm(f)                 # 前
    up = np.array([0.0, 0.0, 1.0])         # 世界上方
    r = np.cross(f, up)
    r /= np.linalg.norm(r)                 # 右
    u = np.cross(r, f)                     # 上（已被正交化）
    return cam, f, r, u


def pixel_to_world(u_px, v_px):
    """把图像像素反投影到 z=PLANE_Z 的平面上"""
    cam, f, r, u = camera_basis()
    tan_half = np.tan(np.radians(FOV) / 2)
    aspect = W / H

    ndc_x = (u_px / W) * 2 - 1
    ndc_y = 1 - (v_px / H) * 2

    direction = f + r * (ndc_x * tan_half * aspect) + u * (ndc_y * tan_half)
    direction /= np.linalg.norm(direction)

    if abs(direction[2]) < 1e-6:
        return None
    t = (PLANE_Z - cam[2]) / direction[2]
    if t < 0:
        return None
    return cam + direction * t


def find_red_block(rgb):
    """在图像里找红色方块，返回 (u, v, 面积, 轮廓)"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 120, 70]), np.array([180, 255, 255]))
    mask = mask1 | mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 60:
        return None
    M = cv2.moments(c)
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return u, v, cv2.contourArea(c), c


# ---------------------------------------------------------------
# 仿真辅助
# ---------------------------------------------------------------
def grasp_orn():
    return p.getQuaternionFromEuler([math.pi, 0, 0])


def step_once():
    p.stepSimulation()
    if PAUSE:
        time.sleep(TIME_STEP)


def move_to(panda, fingers, pos, steps, finger_target=FINGER_OPEN):
    orn = grasp_orn()
    for _ in range(steps):
        joints = p.calculateInverseKinematics(panda, 11, pos, orn)
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=ARM_FORCE)
        for j in fingers:
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=finger_target, force=FORCE)
        step_once()


def read_grip_force(panda, cube, fingers):
    """单侧夹持力（同一指头上的多接触点先求和）、总法向力、接触点数"""
    per_link = {}
    count = 0
    for c in p.getContactPoints(bodyA=panda, bodyB=cube):
        if c[3] in fingers:
            per_link[c[3]] = per_link.get(c[3], 0.0) + c[9]
            count += 1
    total = sum(per_link.values())
    return ((total / len(per_link)) if per_link else 0.0), total, count


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------
# 夹持：力控（开环 / 闭环）
# ---------------------------------------------------------------
def grasp_force_phase(panda, cube, fingers, gx, gy, mode):
    """mode = "open-loop" 或 "closed-loop"；返回采样数据"""
    move_to(panda, fingers, [gx, gy, APPROACH_Z], 200)
    move_to(panda, fingers, [gx, gy, GRASP_Z], 200)
    move_to(panda, fingers, [gx, gy, GRASP_Z], 200, finger_target=PRE_CLOSE)

    for j in fingers:
        p.setJointMotorControl2(panda, j, p.VELOCITY_CONTROL, force=0)

    n = int(round((T_GRASP + T_LIFT + T_HOLD) / TIME_STEP))
    t_arr = np.arange(n) * TIME_STEP
    cmd_arr = np.zeros(n)
    force_arr = np.zeros(n)
    rise_arr = np.zeros(n)
    slip_arr = np.zeros(n)

    cube_z0 = p.getBasePositionAndOrientation(cube)[0][2]
    grip_z0 = p.getLinkState(panda, 11)[0][2]

    measured, contacts, filt, cmd = 0.0, 0, 0.0, 0.0
    for k in range(n):
        t = t_arr[k]
        if t < T_GRASP:
            target = [gx, gy, GRASP_Z]
        else:
            frac = min((t - T_GRASP) / T_LIFT, 1.0)
            target = [gx, gy, GRASP_Z + (LIFT_Z - GRASP_Z) * frac]

        base = TARGET_FORCE * min(t / T_RAMP, 1.0)
        if mode == "open-loop" or contacts == 0:
            cmd_desired = base
        else:
            grip_rise = (p.getLinkState(panda, 11)[0][2] - grip_z0) * 1000.0
            cube_rise = (p.getBasePositionAndOrientation(cube)[0][2] - cube_z0) * 1000.0
            slip_def = max(0.0, grip_rise - cube_rise)
            cmd_desired = base + KP_FORCE * (TARGET_FORCE - filt) + K_SLIP * slip_def
            cmd_desired = clamp(cmd_desired, 0.0, MAX_CMD)

        lim = MAX_CMD_RATE * TIME_STEP
        cmd = clamp(clamp(cmd_desired, cmd - lim, cmd + lim), 0.0, MAX_CMD)
        cmd_arr[k] = cmd

        joints = p.calculateInverseKinematics(panda, 11, target, grasp_orn())
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=ARM_FORCE)
        for j in fingers:
            p.setJointMotorControl2(panda, j, p.TORQUE_CONTROL, force=-cmd)

        step_once()

        measured, _, contacts = read_grip_force(panda, cube, fingers)
        filt = MEAS_ALPHA * measured + (1.0 - MEAS_ALPHA) * filt
        force_arr[k] = measured
        rise_arr[k] = (p.getBasePositionAndOrientation(cube)[0][2] - cube_z0) * 1000.0
        grip_rise = (p.getLinkState(panda, 11)[0][2] - grip_z0) * 1000.0
        slip_arr[k] = max(0.0, grip_rise - rise_arr[k])

    return {"t": t_arr, "cmd": cmd_arr, "force": force_arr,
            "rise": rise_arr, "slip": slip_arr, "mode": mode}


# ---------------------------------------------------------------
# 夹持：原版位置控制（回归用）
# ---------------------------------------------------------------
def grasp_position_phase(panda, cube, fingers, gx, gy):
    move_to(panda, fingers, [gx, gy, APPROACH_Z], 160)
    move_to(panda, fingers, [gx, gy, GRASP_Z], 160)
    move_to(panda, fingers, [gx, gy, GRASP_Z], 120, finger_target=FINGER_CLOSE)
    move_to(panda, fingers, [gx, gy, LIFT_Z], 160, finger_target=FINGER_CLOSE)
    return p.getBasePositionAndOrientation(cube)[0]


# ---------------------------------------------------------------
# 指标
# ---------------------------------------------------------------
def summarize(data):
    """把一次夹持的采样数据压成几个可对比的数字"""
    t = data["t"]
    force = data["force"]
    rise = data["rise"]
    slip = data["slip"]
    cmd = data["cmd"]

    # 打滑苗头：夹爪与方块的位移差超过 SLIP_DETECT_MM，并且持续 0.1 s
    onset = None
    need = max(1, int(0.1 / TIME_STEP))
    run = 0
    for k in range(len(t)):
        run = run + 1 if slip[k] > SLIP_DETECT_MM else 0
        if run >= need:
            onset = float(t[k - need + 1])
            break

    rise_end = float(rise[-1])
    slipped = rise_end < SLIP_RISE_MM

    hold_mask = t >= (T_GRASP + T_LIFT)
    track_err = float(np.mean(np.abs(cmd[hold_mask] - force[hold_mask]))) if hold_mask.any() else 0.0
    dev_target = float(np.mean(force[hold_mask] - TARGET_FORCE)) if hold_mask.any() else 0.0

    hold_time = (onset - T_GRASP) if (slipped and onset is not None) else (float(t[-1]) - T_GRASP)

    return {
        "onset": onset,
        "hold_time": max(0.0, hold_time),
        "track_err": track_err,
        "dev_target": dev_target,
        "rise_end": rise_end,
        "slipped": slipped,
        "peak_force": float(force.max()),
        "mean_cmd": float(cmd[hold_mask].mean()) if hold_mask.any() else 0.0,
    }


# ---------------------------------------------------------------
# 画图
# ---------------------------------------------------------------
def plot_curve(data, stats, n_min, tag, base_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = data["t"]

    ax1.axhline(TARGET_FORCE, color="black", linestyle="--", linewidth=1.2,
                label="设定力 %.2f N" % TARGET_FORCE)
    ax1.plot(t, data["force"], color="tab:red", linewidth=1.6, label="实测接触力")
    ax1.axhline(n_min, color="gray", linestyle=":", linewidth=1.2,
                label="理论最小夹持力 %.2f N" % n_min)
    if stats["onset"] is not None:
        ax1.axvline(stats["onset"], color="tab:orange", linestyle="--", linewidth=1.4,
                    label="打滑苗头 %.2f s" % stats["onset"])
    ax1.set_ylabel("单侧接触力 (N)")
    ax1.set_title("视觉 + %s：力跟踪与滑移" % tag)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(t, data["rise"], color="tab:blue", linewidth=1.6, label="方块抬升高度")
    ax2.axhline(SLIP_RISE_MM, color="gray", linestyle=":", linewidth=1.2,
                label="滑移判定线 %.0f mm" % SLIP_RISE_MM)
    ax2.set_xlabel("仿真时间 (s)")
    ax2.set_ylabel("抬升 (mm)")
    ax2.set_title("抬升高度：掉了就是没夹住")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.axvline(T_GRASP, color="gray", linestyle="-.", linewidth=1.0, alpha=0.8)

    fig.tight_layout()
    out = os.path.join(PROJ_DIR, base_name + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def pick_mode():
    if "--position" in sys.argv:
        return "position"
    if "--open-loop" in sys.argv:
        return "open-loop"
    if "--closed-loop" in sys.argv:
        return "closed-loop"
    env = os.environ.get("GRIPPER_CLOSED_LOOP")
    if env == "0":
        return "open-loop"
    return "closed-loop"


def main():
    mode = pick_mode()
    tag = {"position": "位置控制", "open-loop": "开环", "closed-loop": "闭环"}[mode]
    view_tag = "gui" if USE_GUI else "headless"
    base_name = "视觉抓取_%s_%s" % (mode, view_tag)
    log_path = os.path.join(PROJ_DIR, base_name + ".log")
    log = Log(log_path)

    log("=" * 64)
    log("02 视觉 + 力控串联　模式：%s（%s）" % (mode, tag))
    log("=" * 64)

    if SEED is not None:
        rng = random.Random(int(SEED))
        log("随机种子：%s（开环/闭环可用同一个种子做一对一对比）" % SEED)
    else:
        rng = random.Random()

    p.connect(p.GUI if USE_GUI else p.DIRECT)
    p.setTimeStep(TIME_STEP)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    for j, q in zip(range(7), [0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
        p.resetJointState(panda, j, q)

    fingers = []
    for j in range(p.getNumJoints(panda)):
        if "finger_joint" in p.getJointInfo(panda, j)[1].decode():
            fingers.append(j)
            p.changeDynamics(panda, j, lateralFriction=MU, jointDamping=FINGER_DAMPING)

    cube_x = 0.50 + rng.uniform(-0.03, 0.05)
    cube_y = rng.uniform(-0.05, 0.05)
    cube = p.loadURDF("cube_small.urdf", [cube_x, cube_y, PLANE_Z])
    p.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])
    p.changeDynamics(cube, -1, lateralFriction=MU, spinningFriction=0.0, rollingFriction=0.0)

    for _ in range(90):
        step_once()

    if USE_GUI:
        p.resetDebugVisualizerCamera(1.0, 35, -35, [0.45, 0, 0.15])

    # ---- 视觉 ----
    view = p.computeViewMatrix(CAM_POS, CAM_TARGET, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(FOV, W / H, 0.05, 3.0)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if USE_GUI else p.ER_TINY_RENDERER
    img = p.getCameraImage(W, H, viewMatrix=view, projectionMatrix=proj, renderer=renderer)
    rgb = np.reshape(np.array(img[2], dtype=np.uint8), (H, W, 4))[:, :, :3]

    vis = rgb.copy()
    found = find_red_block(rgb)
    if not found:
        log("[FAIL] 没找到红色方块，检查相机位置或颜色阈值")
        p.disconnect()
        log.close()
        return 1

    u, v, area, contour = found
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    cv2.circle(vis, (u, v), 4, (0, 255, 255), -1)
    cv2.putText(vis, "target (%d,%d)" % (u, v), (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    world = pixel_to_world(u, v)
    log("检测到像素：(%d, %d)  面积 %.0f" % (u, v, area))
    log("反投影世界坐标：x=%.3f y=%.3f z=%.3f" % tuple(world))
    log("方块真实位置：  x=%.3f y=%.3f z=%.3f" % (cube_x, cube_y, PLANE_Z))
    err = float(np.linalg.norm(np.array([world[0] - cube_x, world[1] - cube_y])))
    log("平面误差：%.1f mm" % (err * 1000))

    gx, gy = float(world[0]), float(world[1])

    # ---- 夹持 ----
    # 退出码只反映"流程有没有正常跑完"，不反映这一次抓没抓住（那是实验结果，记在日志里）
    code = 0
    if mode == "position":
        pos = grasp_position_phase(panda, cube, fingers, gx, gy)
        ok = pos[2] > 0.10
        log("抓取结果：%s（方块高度 %.3f m）" % ("成功" if ok else "[FAIL] 失败", pos[2]))
    else:
        data = grasp_force_phase(panda, cube, fingers, gx, gy, mode)
        mass = p.getDynamicsInfo(cube, -1)[0]
        n_min = mass * 9.8 / (2.0 * MU * MU)
        stats = summarize(data)

        log("目标夹持力 %.2f N，理论最小夹持力 %.2f N" % (TARGET_FORCE, n_min))
        if stats["onset"] is None:
            log("打滑苗头：无（位移差始终没超过 %.0f mm）" % SLIP_DETECT_MM)
        else:
            log("打滑苗头：%.2f s（夹爪-方块位移差 > %.0f mm 持续 0.1 s）"
                % (stats["onset"], SLIP_DETECT_MM))
        log("力跟踪误差（指令 vs 实测）：%.3f N" % stats["track_err"])
        if mode == "closed-loop":
            log("相对设定力偏差：%+.3f N（闭环靠补偿主动偏离设定，不是误差）" % stats["dev_target"])
        else:
            log("相对设定力偏差：%+.3f N" % stats["dev_target"])
        log("夹持维持时长：%.2f s" % stats["hold_time"])
        log("保持段平均指令：%.2f N，峰值接触力：%.2f N" % (stats["mean_cmd"], stats["peak_force"]))
        log("方块最终抬升：%.1f mm" % stats["rise_end"])
        log("抓取结果：%s" % ("成功" if not stats["slipped"] else "[FAIL] 滑脱"))

        out_png = plot_curve(data, stats, n_min, tag, base_name)
        log("力曲线已保存：%s" % out_png)

    # ---- 相机画面 ----
    if SHOW and USE_GUI:
        cv2.imshow("camera view", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        t0 = time.time()
        while time.time() - t0 < DISPLAY_SEC:
            if cv2.waitKey(30) & 0xFF != 255:
                break
        cv2.destroyAllWindows()

    p.disconnect()
    log("日志已保存：%s" % log_path)
    log("完成。")
    log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
