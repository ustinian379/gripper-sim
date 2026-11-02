# -*- coding: utf-8 -*-
"""04 闭环力控：在 03 的开环基础上加 P 环（力反馈 + 滑移补偿）

和 03 的区别只有一个：手指该用多大劲，是"算"出来的，不是"定"死的。

    开环：cmd = 目标力                        （定值，方块真要滑也没人管）
    闭环：cmd = 目标力 + Kp*(目标力 - 实测力) + K_slip*滑移量      （每步现算）

两个反馈项各管一件事：
  1. 力环  Kp*(目标-实测)：实测没到目标就多给一点，超了就收一点
  2. 滑移补偿 K_slip*滑移量：滑移量 = 夹爪升了多少 - 方块跟着升了多少（mm）。
     一旦方块开始跟不上夹爪，这一项会把夹持力顶上去了，等它跟上再退回来。

本脚本会把「开环」和「闭环」用同一个目标力各跑一遍，直接对比：
    目标力挑 0.35 N —— 比理论最小夹持力 0.49 N 还低，开环必滑，看闭环能不能救回来。

运行：
    projects/gripper-sim/.venv/Scripts/python.exe 04_闭环力控.py

环境变量：
    GRIPPER_NO_SHOW=1     不弹绘图窗口，只存 PNG（也认 GRIPPER_NO_SHM）
    GRIPPER_HEADLESS=1    不开仿真窗口
    GRIPPER_FAST=1        不按实时节奏等待
"""

import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

# ---------------- 可调参数 ----------------
TIME_STEP = 1.0 / 500.0          # 仿真步长
CUBE_POS = [0.55, 0.0, 0.02]     # 方块初始位置
HOME = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
APPROACH_Z = 0.16
GRASP_Z = 0.015
LIFT_Z = 0.30
FINGER_OPEN = 0.04
ARM_FORCE = 500
T_CLOSE = 0.45                   # 合拢阶段 (s)
T_LIFT = 0.70                    # 抬起阶段 (s)
T_HOLD = 0.45                    # 保持阶段 (s)
SLIP_RISE_MM = 40.0              # 抬升不足这个高度算滑移
MU = 1.0                         # 摩擦系数（Bullet 里两物体相乘）
FINGER_DAMPING = 10.0            # 手指关节阻尼
PRE_CLOSE = 0.032                # 切力模式前的预夹位置

TARGET_FORCE = 0.35              # 目标夹持力 (N)，故意选在理论阈值以下
KP_FORCE = 0.5                   # 力环比例增益：这里的"对象增益"≈1，Kp 必须 < 1，
                                 # 否则离散一拍延迟就会变成正负翻转的极限环（抖成一片）
K_SLIP = 0.05                    # 滑移补偿增益 (N/mm)
MAX_CMD = 5.0                    # 控制器输出上限 (N)
MAX_CMD_RATE = 30.0              # 输出变化率上限 (N/s)，防止一步猛冲
MEAS_ALPHA = 0.3                 # 实测力的低通系数（越小越平滑、越迟钝）
T_RAMP = 0.20                    # 目标力爬升时间 (s)

OUTPUT_PNG = "力控闭环对比.png"
# -----------------------------------------

SHOW_PLOT = not (os.environ.get("GRIPPER_NO_SHOW") == "1" or os.environ.get("GRIPPER_NO_SHM") == "1")
USE_GUI = os.environ.get("GRIPPER_HEADLESS") != "1"
PAUSE = os.environ.get("GRIPPER_FAST") != "1"

import matplotlib                                  # noqa: E402
if not SHOW_PLOT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------
# 场景
# ---------------------------------------------------------------
def load_scene():
    p.resetSimulation()
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    cube = p.loadURDF("cube_small.urdf", CUBE_POS)
    p.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])
    p.changeDynamics(cube, -1, lateralFriction=MU, spinningFriction=0.0, rollingFriction=0.0)

    fingers = []
    for j in range(p.getNumJoints(panda)):
        if "finger_joint" in p.getJointInfo(panda, j)[1].decode():
            fingers.append(j)
            p.changeDynamics(panda, j, lateralFriction=MU, jointDamping=FINGER_DAMPING)

    for j, q in zip(range(7), HOME):
        p.resetJointState(panda, j, q)
    return panda, cube, fingers


def grasp_orn():
    return p.getQuaternionFromEuler([math.pi, 0, 0])


def step_once():
    p.stepSimulation()
    if PAUSE:
        time.sleep(TIME_STEP)


def move_arm(panda, fingers, pos, steps, finger_target=FINGER_OPEN):
    orn = grasp_orn()
    for _ in range(steps):
        joints = p.calculateInverseKinematics(panda, 11, pos, orn)
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=ARM_FORCE)
        for j in fingers:
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=finger_target, force=ARM_FORCE)
        step_once()


def read_grip_force(panda, cube, fingers):
    """返回 (单侧夹持力 N, 总法向力 N, 接触点数)；同一指头上的多接触点先求和。"""
    per_link = {}
    count = 0
    for c in p.getContactPoints(bodyA=panda, bodyB=cube):
        if c[3] in fingers:
            per_link[c[3]] = per_link.get(c[3], 0.0) + c[9]
            count += 1
    total = sum(per_link.values())
    per_side = total / len(per_link) if per_link else 0.0
    return per_side, total, count


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------
# 一次抓取：mode = "open"（定力）或 "closed"（P 反馈）
# ---------------------------------------------------------------
def run_trial(mode, level=TARGET_FORCE):
    panda, cube, fingers = load_scene()
    orn = grasp_orn()
    approach_steps = int(0.5 / TIME_STEP)

    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], APPROACH_Z], approach_steps)
    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], GRASP_Z], approach_steps)
    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], GRASP_Z], approach_steps,
             finger_target=PRE_CLOSE)

    for j in fingers:
        p.setJointMotorControl2(panda, j, p.VELOCITY_CONTROL, force=0)

    n_steps = int(round((T_CLOSE + T_LIFT + T_HOLD) / TIME_STEP))
    t_arr = np.arange(n_steps) * TIME_STEP
    force_arr = np.zeros(n_steps)
    cmd_arr = np.zeros(n_steps)
    count_arr = np.zeros(n_steps, dtype=int)
    rise_arr = np.zeros(n_steps)
    slip_arr = np.zeros(n_steps)

    cube_z0 = p.getBasePositionAndOrientation(cube)[0][2]
    grip_z0 = p.getLinkState(panda, 11)[0][2]

    measured, contacts = 0.0, 0          # 上一步的实测值，闭环就用它算下一步的输出
    meas_filt = 0.0                      # 低通后的实测力，喂给 P 环
    cmd = 0.0
    for k in range(n_steps):
        t = t_arr[k]
        if t < T_CLOSE:
            target = [CUBE_POS[0], CUBE_POS[1], GRASP_Z]
        else:
            frac = min((t - T_CLOSE) / T_LIFT, 1.0)
            target = [CUBE_POS[0], CUBE_POS[1], GRASP_Z + (LIFT_Z - GRASP_Z) * frac]

        base = level * min(t / T_RAMP, 1.0)          # 前馈：目标力本身

        if mode == "open" or contacts == 0:
            # 开环，或者还没接触到方块：只给前馈，别让 P 项在空气里放大
            cmd_desired = base
        else:
            err = level - meas_filt                   # 力环误差
            grip_rise = (p.getLinkState(panda, 11)[0][2] - grip_z0) * 1000.0
            cube_rise = (p.getBasePositionAndOrientation(cube)[0][2] - cube_z0) * 1000.0
            slip_def = max(0.0, grip_rise - cube_rise)   # 方块没跟上夹爪 = 在滑
            cmd_desired = base + KP_FORCE * err + K_SLIP * slip_def
            cmd_desired = clamp(cmd_desired, 0.0, MAX_CMD)

        # 输出变化率限幅：一步最多走 MAX_CMD_RATE * TIME_STEP
        step_limit = MAX_CMD_RATE * TIME_STEP
        cmd = clamp(cmd_desired, cmd - step_limit, cmd + step_limit)
        cmd = clamp(cmd, 0.0, MAX_CMD)
        cmd_arr[k] = cmd

        joints = p.calculateInverseKinematics(panda, 11, target, orn)
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=ARM_FORCE)
        for j in fingers:
            p.setJointMotorControl2(panda, j, p.TORQUE_CONTROL, force=-cmd)

        step_once()

        measured, _, contacts = read_grip_force(panda, cube, fingers)
        meas_filt = MEAS_ALPHA * measured + (1.0 - MEAS_ALPHA) * meas_filt
        force_arr[k] = measured
        count_arr[k] = contacts
        rise_arr[k] = (p.getBasePositionAndOrientation(cube)[0][2] - cube_z0) * 1000.0
        grip_rise = (p.getLinkState(panda, 11)[0][2] - grip_z0) * 1000.0
        slip_arr[k] = max(0.0, grip_rise - rise_arr[k])

    hold_mask = t_arr >= (T_CLOSE + T_LIFT)
    rise_end = float(rise_arr[-1])
    return {
        "mode": mode,
        "t": t_arr,
        "cmd": cmd_arr,
        "force": force_arr,
        "rise": rise_arr,
        "slip": slip_arr,
        "level": level,
        "peak_cmd": float(cmd_arr.max()),
        "mean_cmd_hold": float(cmd_arr[hold_mask].mean()),
        "mean_force_hold": float(force_arr[hold_mask].mean()),
        "rise_end": rise_end,
        "slipped": rise_end < SLIP_RISE_MM,
    }


# ---------------------------------------------------------------
# 画图
# ---------------------------------------------------------------
def plot_compare(open_tr, closed_tr, n_min):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    c_open, c_closed = "tab:blue", "tab:red"
    level = open_tr["level"]

    # 1) 力跟踪
    ax1.axhline(level, color="black", linestyle="--", linewidth=1.2,
                label="设定力 %.2f N" % level)
    ax1.plot(open_tr["t"], open_tr["force"], color=c_open, linewidth=1.6,
             label="开环 实测接触力")
    ax1.plot(closed_tr["t"], closed_tr["force"], color=c_closed, linewidth=1.6,
             label="闭环 实测接触力")
    ax1.axhline(n_min, color="gray", linestyle=":", linewidth=1.2,
                label="理论最小夹持力 %.2f N" % n_min)
    ax1.set_ylabel("单侧接触力 (N)")
    ax1.set_title("同目标力 %.2f N：开环 vs P 闭环" % level)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    # 2) 控制器输出
    ax2.plot(open_tr["t"], open_tr["cmd"], color=c_open, linestyle="--", linewidth=1.4,
             label="开环输出（恒定）")
    ax2.plot(closed_tr["t"], closed_tr["cmd"], color=c_closed, linewidth=1.6,
             label="闭环输出（实时补偿）")
    ax2.axhline(level, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("手指施力指令 (N)")
    ax2.set_title("控制器输出：闭环在方块跟不上时自动加力")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")

    # 3) 抬升高度
    ax3.plot(open_tr["t"], open_tr["rise"], color=c_open, linewidth=1.6,
             label="开环 %s" % ("[FAIL] 滑移" if open_tr["slipped"] else "成功"))
    ax3.plot(closed_tr["t"], closed_tr["rise"], color=c_closed, linewidth=1.6,
             label="闭环 %s" % ("[FAIL] 滑移" if closed_tr["slipped"] else "成功"))
    ax3.axhline(SLIP_RISE_MM, color="gray", linestyle=":", linewidth=1.2)
    ax3.text(0.02, SLIP_RISE_MM + 8, "滑移判定线 %.0f mm" % SLIP_RISE_MM,
             fontsize=9, color="gray", transform=ax3.get_yaxis_transform())
    ax3.set_xlabel("仿真时间 (s)")
    ax3.set_ylabel("方块抬升高度 (mm)")
    ax3.set_title("抬升高度：谁跟着夹爪走，谁掉下去了")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9, loc="upper left")

    for ax in (ax1, ax2, ax3):
        ax.axvline(T_CLOSE, color="gray", linestyle="-.", linewidth=1.0, alpha=0.8)

    fig.tight_layout()
    out = os.path.join(PROJ_DIR, OUTPUT_PNG)
    fig.savefig(out, dpi=130)
    print("对比曲线已保存：%s" % out)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return out


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main():
    print("=" * 64)
    print("04 闭环力控：P 反馈 + 滑移补偿")
    print("=" * 64)

    p.connect(p.GUI if USE_GUI else p.DIRECT)

    _, cube, _ = load_scene()
    mass = p.getDynamicsInfo(cube, -1)[0]
    mu_eff = MU * MU
    n_min = mass * 9.8 / (2.0 * mu_eff)
    print("方块质量 %.3f kg，摩擦系数 %.2f（有效 %.2f）" % (mass, MU, mu_eff))
    print("理论最小夹持力（单侧）≈ %.2f N" % n_min)
    print("目标力 %.2f N（故意低于阈值），Kp=%.1f，K_slip=%.2f N/mm，输出上限 %.1f N"
          % (TARGET_FORCE, KP_FORCE, K_SLIP, MAX_CMD))
    print("-" * 64)

    open_tr = run_trial("open")
    print("[开环] 峰值指令 %.2f N | 保持段指令 %.2f N | 保持段实测 %.2f N | 抬高 %.1f mm | %s"
          % (open_tr["peak_cmd"], open_tr["mean_cmd_hold"], open_tr["mean_force_hold"],
             open_tr["rise_end"], "[FAIL] 滑移" if open_tr["slipped"] else "成功"))

    closed_tr = run_trial("closed")
    print("[闭环] 峰值指令 %.2f N | 保持段指令 %.2f N | 保持段实测 %.2f N | 抬高 %.1f mm | %s"
          % (closed_tr["peak_cmd"], closed_tr["mean_cmd_hold"], closed_tr["mean_force_hold"],
             closed_tr["rise_end"], "[FAIL] 滑移" if closed_tr["slipped"] else "成功"))

    print("-" * 64)
    if open_tr["slipped"] and not closed_tr["slipped"]:
        print("结论：开环滑了，闭环夹住了 —— 补偿生效")
    elif open_tr["slipped"] and closed_tr["slipped"]:
        print("[FAIL] 结论：两种模式都滑了，把 KP_FORCE / K_SLIP / MAX_CMD 调大再试")
    elif not open_tr["slipped"] and not closed_tr["slipped"]:
        print("结论：两种模式都夹住了，把 TARGET_FORCE 再调低一点才能看出差别")
    else:
        print("[FAIL] 结论：开环夹住了、闭环反而滑了，检查反馈符号")

    plot_compare(open_tr, closed_tr, n_min)
    p.disconnect()
    print("完成。")


if __name__ == "__main__":
    main()
