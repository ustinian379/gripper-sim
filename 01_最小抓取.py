# -*- coding: utf-8 -*-
"""01 最小抓取：Franka 机械臂把方块夹起来（无视觉，位置写死）

运行：python 01_最小抓取.py
依赖：pybullet
"""

import time
import pybullet as p
import pybullet_data

# ---------------- 可调参数 ----------------
CUBE_POS = [0.55, 0.0, 0.02]   # 方块位置（x 向前，y 左右，z 高度）
APPROACH_Z = 0.16              # 先移动到方块上方这个高度
GRASP_Z = 0.015                # 实际下探到的高度（越低夹得越实）
FINGER_OPEN = 0.04             # 两指张开（Panda 手指关节 0~0.04）
FINGER_CLOSE = 0.0             # 合拢
FORCE = 200                    # 夹持力上限（新物体用默认 200 就够）
# -----------------------------------------


def main():
    # 1. 起仿真
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")                     # 地面
    p.loadURDF("table/table.urdf", [0.55, 0, -0.65], globalScaling=1.0)  # 桌台

    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    cube = p.loadURDF("cube_small.urdf", CUBE_POS)

    # 让方块颜色显眼一点（方便看）
    p.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.15, 0.15, 1.0])

    # 机械臂回到初始姿态，等它稳定
    home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for j, q in zip(range(7), home):
        p.resetJointState(panda, j, q)
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1 / 240)

    def move_to(pos, gripper=1.0, steps=180):
        """让末端（link 11）移动到 pos；gripper=1 张开，0 合拢"""
        orn = p.getQuaternionFromEuler([3.14159, 0, 0])   # 夹爪朝下
        joints = p.calculateInverseKinematics(panda, 11, pos, orn)
        for j in range(7):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=joints[j], force=500)
        # 手指：位置 0=合拢，0.04=张开
        finger_target = FINGER_OPEN if gripper else FINGER_CLOSE
        for j in (9, 10):
            p.setJointMotorControl2(panda, j, p.POSITION_CONTROL,
                                    targetPosition=finger_target, force=FORCE)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1 / 240)

    # 2. 张开手指
    move_to([CUBE_POS[0], CUBE_POS[1], APPROACH_Z], gripper=1.0)

    # 3. 下降到抓取高度
    move_to([CUBE_POS[0], CUBE_POS[1], GRASP_Z], gripper=1.0)

    # 4. 合拢
    move_to([CUBE_POS[0], CUBE_POS[1], GRASP_Z], gripper=0.0, steps=120)

    # 5. 抬起来
    move_to([CUBE_POS[0], CUBE_POS[1], APPROACH_Z + 0.15], gripper=0.0)

    # 6. 看看夹住没有
    cube_now = p.getBasePositionAndOrientation(cube)[0]
    lifted = cube_now[2] > 0.08
    print("=" * 40)
    print("方块当前高度：%.3f m" % cube_now[2])
    print("结果：%s" % ("夹起来了" if lifted else "[FAIL] 没夹住，掉回去了"))
    print("=" * 40)

    # 保持 3 秒看结果，然后关闭
    for _ in range(720):
        p.stepSimulation()
        time.sleep(1 / 240)
    p.disconnect()


if __name__ == "__main__":
    main()
