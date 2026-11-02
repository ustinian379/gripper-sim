# -*- coding: utf-8 -*-
"""03 力控：用「力模式」夹住方块，边抬边测真实接触力，画出设定力 vs 实测力

比 01 / 02 多出来的三件事：
  1. 手指不再是"闭合到某个位置"，而是直接给一个力（Torque/Force 控制）
  2. 每一步用 getContactPoints 读夹爪和方块之间的真实接触力（normalForce 求和）
  3. 每组目标力跑一次，最后画曲线：设定力 vs 实测接触力 + 方块抬升高度（滑移）

关于"力模式"：PyBullet 的 setJointMotorControl2 没有叫 FORCE 的模式，
真正按力施加的是 p.TORQUE_CONTROL —— 对滑动关节（panda 的两根手指）来说，
force 参数的单位就是牛顿（N），所以这里直接写 force=目标夹持力。
手指张开仍然用 POSITION_CONTROL（那时候要的是位置，不是力）。

对应课程：材料力学里的摩擦与滑移；电机控制里的力上限与力跟踪。

运行：
    projects/gripper-sim/.venv/Scripts/python.exe 03_力控.py

可选环境变量：
    GRIPPER_HEADLESS=1   不开仿真窗口（无头跑，只看日志和曲线）
    GRIPPER_NO_SHOW=1    不弹绘图窗口，只保存 PNG
    GRIPPER_FAST=1       不按实时节奏等待，尽快跑完
"""

import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

# ---------------- 可调参数 ----------------
TIME_STEP = 1.0 / 500.0          # 仿真步长（力控要小一点，1/240 容易抖）
CUBE_POS = [0.55, 0.0, 0.02]     # 方块初始位置（x 向前，y 左右，z 高度）
HOME = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]   # 机械臂初始姿态
APPROACH_Z = 0.16                # 先到方块上方这个高度
GRASP_Z = 0.015                  # 下探到的高度
LIFT_Z = 0.30                    # 抬到这个高度
FINGER_OPEN = 0.04               # 两指张开（0~0.04 m）
ARM_FORCE = 500                  # 机械臂关节力上限
FORCE_LEVELS = [0.3, 0.5, 0.8, 1.2, 2.0]   # 每组的「目标夹持力」(N)
T_CLOSE = 0.45                   # 合拢阶段时长 (s)
T_LIFT = 0.70                    # 抬起阶段时长 (s)
T_HOLD = 0.45                    # 保持阶段时长 (s)
SLIP_RISE_MM = 40.0              # 抬升不足这个高度就算滑移
MU = 1.0                         # 夹爪/方块的摩擦系数（Bullet 里两物体相乘，所以有效 1.0*1.0）
T_RAMP = 0.20                    # 目标力从 0 爬到设定值的时间 (s)，别一步砸上去
FINGER_DAMPING = 10.0            # 手指关节阻尼（N·s/m），让接触别是硬撞上去
PRE_CLOSE = 0.032                # 切力模式前先合到离方块约 6 mm，减少接触冲击
OUTPUT_PNG = "力控曲线.png"
# -----------------------------------------

SHOW_PLOT = os.environ.get("GRIPPER_NO_SHOW") != "1"
USE_GUI = os.environ.get("GRIPPER_HEADLESS") != "1"
PAUSE = os.environ.get("GRIPPER_FAST") != "1"

import matplotlib                                  # noqa: E402
if not SHOW_PLOT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------
# 场景
# ---------------------------------------------------------------
def load_scene():
    """重新搭一遍场景，返回 (机械臂, 方块, 手指关节编号列表)"""
    p.resetSimulation()
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")                                        # 地面
    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    cube = p.loadURDF("cube_small.urdf", CUBE_POS)
    p.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])  # 红色，和 02 一致

    # 摩擦系数显式设一遍，免得"为什么夹不住"变成玄学
    p.changeDynamics(cube, -1, lateralFriction=MU, spinningFriction=0.0, rollingFriction=0.0)

    fingers = []
    for j in range(p.getNumJoints(panda)):
        name = p.getJointInfo(panda, j)[1].decode()
        if "finger_joint" in name:
            fingers.append(j)
            p.changeDynamics(panda, j, lateralFriction=MU, jointDamping=FINGER_DAMPING)

    for j, q in zip(range(7), HOME):
        p.resetJointState(panda, j, q)

    return panda, cube, fingers


def grasp_orn():
    """夹爪朝下的四元数"""
    return p.getQuaternionFromEuler([math.pi, 0, 0])


def step_once():
    p.stepSimulation()
    if PAUSE:
        time.sleep(TIME_STEP)


def move_arm(panda, fingers, pos, steps, finger_target=FINGER_OPEN):
    """机械臂走到 pos，手指用位置控制张开/合拢"""
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
    """取夹爪与方块之间的真实接触：返回 (单侧夹持力 N, 总法向力 N, 接触点数)

    一个指头和方块接触时，box-box 最多会给出 4 个接触点，每个点都有自己的
    normalForce。所以要先把同一个指头（同一个 link）上的接触点加起来，
    再对两根手指取平均 —— 得到"每根手指平均夹多少牛"，才能和设定力对比。
    """
    contacts = p.getContactPoints(bodyA=panda, bodyB=cube)
    per_link = {}
    count = 0
    for c in contacts:
        link_a = c[3]                 # 机械臂这一侧的 link 编号
        if link_a in fingers:
            per_link[link_a] = per_link.get(link_a, 0.0) + c[9]   # c[9] = normalForce
            count += 1
    total = sum(per_link.values())
    per_side = total / len(per_link) if per_link else 0.0
    return per_side, total, count


# ---------------------------------------------------------------
# 一组力试验
# ---------------------------------------------------------------
def run_trial(level):
    panda, cube, fingers = load_scene()
    orn = grasp_orn()
    approach_steps = int(0.5 / TIME_STEP)

    # 1. 移到方块上方，手指张开
    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], APPROACH_Z], approach_steps)
    # 2. 下降到抓取高度
    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], GRASP_Z], approach_steps)
    # 2.5 先合到"快碰到但还没碰到"的位置，切力模式时的接触冲击会小很多
    move_arm(panda, fingers, [CUBE_POS[0], CUBE_POS[1], GRASP_Z], approach_steps,
             finger_target=PRE_CLOSE)

    # 3. 手指切成"力模式"：先关掉默认的电机，之后每步直接给力
    for j in fingers:
        p.setJointMotorControl2(panda, j, p.VELOCITY_CONTROL, force=0)

    n_steps = int(round((T_CLOSE + T_LIFT + T_HOLD) / TIME_STEP))
    t_arr = np.arange(n_steps) * TIME_STEP
    force_arr = np.zeros(n_steps)
    target_arr = np.zeros(n_steps)
    total_arr = np.zeros(n_steps)
    count_arr = np.zeros(n_steps, dtype=int)
    rise_arr = np.zeros(n_steps)

    cube_z0 = p.getBasePositionAndOrientation(cube)[0][2]

    for k in range(n_steps):
        t = t_arr[k]
        if t < T_CLOSE:
            target = [CUBE_POS[0], CUBE_POS[1], GRASP_Z]
        else:
            frac = min((t - T_CLOSE) / T_LIFT, 1.0)
            z = GRASP_Z + (LIFT_Z - GRASP_Z) * frac
            target = [CUBE_POS[0], CUBE_POS[1], z]

        # 设定力：前 T_RAMP 秒从 0 线性爬到 level，之后保持
        cmd = level * min(t / T_RAMP, 1.0)
        target_arr[k] = cmd

        joints = p.calculateInverseKinematics(panda, 11, target, orn)
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=ARM_FORCE)
        # 手指的关节坐标 0=合拢、0.04=张开，所以"合拢方向的力"要取负号：
        # force=-cmd 表示每根手指用 cmd 牛往里夹。
        for j in fingers:
            p.setJointMotorControl2(panda, j, p.TORQUE_CONTROL, force=-cmd)

        step_once()

        force_arr[k], total_arr[k], count_arr[k] = read_grip_force(panda, cube, fingers)
        rise_arr[k] = (p.getBasePositionAndOrientation(cube)[0][2] - cube_z0) * 1000.0

    hold_mask = t_arr >= (T_CLOSE + T_LIFT)
    rise_end_mm = float(rise_arr[-1])
    slipped = rise_end_mm < SLIP_RISE_MM

    return {
        "level": level,
        "t": t_arr,
        "force": force_arr,
        "target": target_arr,
        "total": total_arr,
        "counts": count_arr,
        "rise": rise_arr,
        "peak": float(force_arr.max()),
        "mean_hold": float(force_arr[hold_mask].mean()) if hold_mask.any() else 0.0,
        "contacts": int(count_arr.max()),
        "rise_end": rise_end_mm,
        "slipped": slipped,
    }


# ---------------------------------------------------------------
# 画图
# ---------------------------------------------------------------
def plot_results(trials):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(trials)))
    handles = []
    for color, tr in zip(colors, trials):
        label = "实测 %.2f N" % tr["level"]
        ax1.plot(tr["t"], tr["target"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        line, = ax1.plot(tr["t"], tr["force"], color=color, linewidth=1.6, label=label)
        handles.append(line)
        ax2.plot(tr["t"], tr["rise"], color=color, linewidth=1.6,
                 label="%.2f N %s" % (tr["level"], "[FAIL]" if tr["slipped"] else "成功"))

    ax1.set_ylabel("单侧接触力 (N)")
    ax1.set_title("设定力（虚线）vs 实测接触力（实线，单侧平均）")
    ax1.axvline(T_CLOSE, color="gray", linestyle="-.", linewidth=1.0, alpha=0.9)
    ax1.text(T_CLOSE + 0.02, 0.65, "开始抬升", fontsize=9, color="gray",
             transform=ax1.get_xaxis_transform())
    ax1.grid(alpha=0.3)
    ax1.legend(handles=handles, fontsize=9, ncol=2, title="目标夹持力")

    ax2.axhline(SLIP_RISE_MM, color="gray", linestyle=":", linewidth=1.2)
    ax2.text(0.02, SLIP_RISE_MM + 8, "滑移判定线 %.0f mm" % SLIP_RISE_MM,
             fontsize=9, color="gray", transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("仿真时间 (s)")
    ax2.set_ylabel("方块抬升高度 (mm)")
    ax2.set_title("抬升高度：线掉下来就是打滑")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=2, title="目标夹持力")

    fig.tight_layout()
    out = os.path.join(PROJ_DIR, OUTPUT_PNG)
    fig.savefig(out, dpi=130)
    print("曲线已保存：%s" % out)

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
    print("03 力控：目标夹持力 vs 实测接触力")
    print("=" * 64)

    p.connect(p.GUI if USE_GUI else p.DIRECT)

    # 理论阈值：两边各受 N，摩擦 f=mu*N，要 2*mu*N >= m*g
    _, cube, _ = load_scene()
    mass = p.getDynamicsInfo(cube, -1)[0]
    mu_eff = MU * MU                      # Bullet 里两物体摩擦系数相乘
    n_min = mass * 9.8 / (2.0 * mu_eff)   # 两边各 N，摩擦力 2*mu*N 要扛住 m*g
    print("方块质量 %.3f kg，摩擦系数 %.2f（Bullet 里两物体相乘，有效 %.2f）" % (mass, MU, mu_eff))
    print("理论最小夹持力（单侧）≈ %.2f N；低于它应该会滑" % n_min)
    print("试验设定力：" + ", ".join("%.2f N" % f for f in FORCE_LEVELS))
    print("-" * 64)

    trials = []
    for level in FORCE_LEVELS:
        tr = run_trial(level)
        trials.append(tr)
        verdict = "[FAIL] 滑移" if tr["slipped"] else "成功"
        print("目标 %5.2f N | 接触点 %d | 峰值 %5.2f N | 保持段均值 %5.2f N | 抬高 %6.1f mm | %s"
              % (tr["level"], tr["contacts"], tr["peak"], tr["mean_hold"],
                 tr["rise_end"], verdict))

    print("-" * 64)
    ok = [t for t in trials if not t["slipped"]]
    bad = [t for t in trials if t["slipped"]]
    print("成功 %d 组，[FAIL] %d 组" % (len(ok), len(bad)))
    if bad:
        print("滑掉的力：" + ", ".join("%.2f N" % t["level"] for t in bad))
    print("力跟踪偏差（保持段 实测-设定）：" +
          ", ".join("%.2f N→%+.2f N" % (t["level"], t["mean_hold"] - t["level"]) for t in trials))

    plot_results(trials)
    p.disconnect()
    print("完成。")


if __name__ == "__main__":
    main()
