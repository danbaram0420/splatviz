# run_main.py 수정안: scene_path + object_path 두 디렉토리에서 ply를 로드

import click
from pathlib import Path
from splatviz import Splatviz
import time
import trimesh, numpy as np, pybullet as p, math, os, pybullet_data
from pybullet_utils import bullet_client
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import cv2




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
    p.setTimeStep(1 / 50)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf 등

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    quat = p.getQuaternionFromEuler([math.radians(210), 0, 0])

    p.setPhysicsEngineParameter(
        numSolverIterations=150,
        contactBreakingThreshold=0.02,
        enableConeFriction=1
        # enableSleeping=1  ← 지원 안 되는 버전
    )
    # 1) 바닥(plane) 추가: 옵션
    # p.loadURDF("plane.urdf")

    # 2) 정적 Scene 메시 ----------------------------------
    scene_mesh = "cube_high/point_cloud_scene/iteration_30100/tsdf_fusion_post_vis.obj"
    sc_col = p.createCollisionShape(p.GEOM_MESH,
                                    fileName=scene_mesh,
                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    sc_vis = p.createVisualShape(p.GEOM_MESH, fileName=scene_mesh)
    scene_uid = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=sc_col,
                                  baseVisualShapeIndex=sc_vis, basePosition=[0, 0, 0], baseOrientation=quat)

    print("scene_uid =", scene_uid, " sc_vis =", sc_vis)  # -1 이면 로드 실패
    assert scene_uid >= 0 and sc_vis >= 0, "scene mesh 로드 실패!"

    # 3) 동적 오브젝트 -------------------------------------
    obj_mesh = "cube_high/point_cloud_obj/iteration_37000/scene_union.obj"
    if not os.path.exists("cube_high/point_cloud_obj/iteration_37000/obj_vhacd.obj"):
        # (권장) 먼저 볼록 분해: 결과 .vhacd.obj 가 자동 저장됨
        ok = p.vhacd(obj_mesh, "cube_high/point_cloud_obj/iteration_37000/obj_vhacd.obj", "log.txt", concavity=0.002, resolution=1_000_000)
        if not ok:
            raise RuntimeError("VHACD 실패—로그 확인")

    convex_path = "cube_high/point_cloud_obj/iteration_37000/obj_vhacd.obj"
    obj_mesh = "cube_high/point_cloud_obj/iteration_37000/obj_vhacd.obj"

    mesh = trimesh.load(convex_path, force='mesh')
    density = 700  # kg/m³  (목적에 맞게 조정)
    mass = mesh.volume * density
    com = mesh.center_mass  # [x,y,z]  (메시 로컬 기준)

    I_tensor = mesh.moment_inertia  # 3×3, CoM 기준
    eigval, eigvec = np.linalg.eigh(I_tensor)  # 주축 분해
    inertia_diag = eigval.tolist()  # Ixx,Iyy,Izz
    # 행렬→쿼터니언

    quat_I = quaternion_from_matrix(np.vstack([np.hstack([eigvec, [[0], [0], [0]]]),
                                               [0, 0, 0, 1]])[:3, :3]).tolist()

    obj_col = p.createCollisionShape(p.GEOM_MESH, fileName=convex_path)
    obj_vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_mesh)

    initialPos = com.tolist()

    obj_uid = p.createMultiBody(
        baseMass=30.0,  # 질량 [kg]
        baseCollisionShapeIndex=obj_col,
        baseVisualShapeIndex=obj_vis,
        basePosition=[0, 0, 0],  # 월드 좌표
        baseOrientation=quat,
        baseInertialFramePosition=initialPos,  # ★ CoM
        baseInertialFrameOrientation=quat_I,
    )
    ply_path = os.path.abspath("cube_high/point_cloud_obj/iteration_37000/point_cloud.ply")
    p.changeDynamics(obj_uid, -1,
                     lateralFriction=0.9,
                     rollingFriction=0.03, spinningFriction=0.03,
                     linearDamping=0.05, angularDamping=0.05,
                     frictionAnchor=1,
                     activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
    print("obj created")

    # If scene_path is provided, use it and any .ply files in objects_path
    if scene_path:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port,
                             scene_path=scene_path, objects_path=objects_path)
    else:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port)

    splatviz.register_dynamic_object(ply_path, obj_uid, com, quat_I, init_world_pos = [0, 0, 0], init_world_quat = quat)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not splatviz.should_close():
        p.stepSimulation()
        splatviz.sync_dynamic_objects(scene_origin_pos=[0, 0, 0], scene_origin_quat=quat)
        splatviz.draw_frame()
        ret, frame = capture.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    splatviz.close()
    capture.release()

if __name__ == '__main__':
    main()