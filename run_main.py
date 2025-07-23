# run_main.py 수정안: scene_path + object_path 두 디렉토리에서 ply를 로드

import click
from pathlib import Path
from splatviz import Splatviz
import time
import trimesh, numpy as np, pybullet as p, math, os, pybullet_data
from pybullet_utils import bullet_client
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R

def local_delta_link(p_b, q_b, p_i, q_i, p_t, q_t):
    # 현재 링크 자세
    q_bt = R.from_quat(q_t) * R.from_quat(q_i).inv()
    p_bt = np.asarray(p_t) - q_bt.apply(p_i)

    # 상대값 (링크 기준)
    q_rel = R.from_quat(q_b).inv() * q_bt
    p_rel = R.from_quat(q_b).inv().apply(p_bt - p_b)
    q_rel_xyzw = q_rel.as_quat()
    q_rel_wxyz = (q_rel_xyzw[3], *q_rel_xyzw[:3])

    return tuple(p_rel), tuple(q_rel_wxyz)


@click.command()
@click.option("--data_path", default="./resources/sample_scenes", help="root path for .ply files", metavar="PATH")
@click.option("--scene_path", default="", help="Path to a single scene .ply file", type=click.Path(exists=True))
@click.option("--objects_path", default="", help="Path to a directory of object .ply files", type=click.Path(exists=True))
@click.option("--mode", default="default", type=str, help="viewer mode")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
@click.option("--gan_path", default=None)
def main(data_path, scene_path, objects_path, mode, host, port, gan_path):
    #physics
    p.connect(p.GUI)  # 또는 p.DIRECT
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 100)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf 등

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    p.setPhysicsEngineParameter(
        numSolverIterations=150,
        contactBreakingThreshold=0.02,
        enableConeFriction=1
        # enableSleeping=1  ← 지원 안 되는 버전
    )

    rotation = [210,0,0]
    quat = p.getQuaternionFromEuler([math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])])

    # If scene_path is provided, use it and any .ply files in objects_path
    if scene_path:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port,
                             scene_path=scene_path, objects_path=objects_path, quat=quat)
    else:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port, quat=quat)

    while not splatviz.should_close():
        p.stepSimulation()
        for idx, phys in enumerate(splatviz.physics_initials, start=1):
            uid = phys[0]
            quat_I = phys[1]
            com = phys[2]
            opos, oquat = p.getBasePositionAndOrientation(uid)  # PyBullet position and quaternion (x,y,z,w)
            #quat tuple, initialPos list, quat_I list, opos tuple, oquat tuple
            lopos, loquat = local_delta_link(splatviz.object_transforms[idx][1], quat, com, quat_I, opos, oquat)
            splatviz.set_transform(1, loquat, lopos)
        splatviz.draw_frame()
    splatviz.close()

if __name__ == '__main__':
    main()