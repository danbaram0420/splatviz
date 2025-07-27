# run_main.py 수정안: scene_path + object_path 두 디렉토리에서 ply를 로드

import click
from pathlib import Path
from splatviz import Splatviz
import time
import threading
import trimesh, numpy as np, pybullet as p, math, os, pybullet_data
from sdgs.sdgs_predictor import SixDGSPredictor
from pybullet_utils import bullet_client
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import cv2


@click.command()
@click.option("--data_path", default="./resources/sample_scenes", help="root path for .ply files", metavar="PATH")
@click.option("--scene_path", default="", help="Path to a single scene .ply file", type=click.Path())
@click.option("--objects_path", default="", help="Path to a directory of object .ply files", type=click.Path())
@click.option("--checkpoint_path", default="", help="Folder of id_module.th", type=str)
@click.option("--mode", default="default", type=str, help="viewer mode")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
@click.option("--gan_path", default=None)
@click.option("--video", default="", help="Video file path (if empty, use webcam 0)", type=str)
def main(data_path, scene_path, objects_path, checkpoint_path, mode, host, port, gan_path, video=0):
    #physics
    p.connect(p.GUI)  # 또는 p.DIRECT
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 50)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf 등

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    quat = p.getQuaternionFromEuler([math.radians(235), 0, 0])

    p.setPhysicsEngineParameter(
        numSolverIterations=150,
        contactBreakingThreshold=0.02,
        enableConeFriction=1
        # enableSleeping=1  ← 지원 안 되는 버전
    )
    # 1) 바닥(plane) 추가: 옵션
    # p.loadURDF("plane.urdf")

    # 2) 정적 Scene 메시 ----------------------------------
    scene_mesh = "realscene_2/point_cloud_scene/iteration_65000/tsdf_fusion_post_vis.obj"
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
    #1
    obj_mesh_1 = "realscene_2/point_cloud_obj/iteration_67000/point_cloud_1.obj"
    if not os.path.exists("realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_1.obj"):
        # (권장) 먼저 볼록 분해: 결과 .vhacd.obj 가 자동 저장됨
        ok = p.vhacd(obj_mesh_1, "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_1.obj", "log.txt", concavity=0.002, resolution=1_000_000)
        if not ok:
            raise RuntimeError("VHACD 실패—로그 확인")

    convex_path_1 = "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_1.obj"
    obj_mesh_1 = "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_1.obj"

    mesh_1 = trimesh.load(convex_path_1, force='mesh')
    com_1 = mesh_1.center_mass  # [x,y,z]  (메시 로컬 기준)

    I_tensor_1 = mesh_1.moment_inertia  # 3×3, CoM 기준
    eigval, eigvec = np.linalg.eigh(I_tensor_1)  # 주축 분해
    # 행렬→쿼터니언

    quat_I_1 = quaternion_from_matrix(np.vstack([np.hstack([eigvec, [[0], [0], [0]]]),
                                               [0, 0, 0, 1]])[:3, :3]).tolist()

    obj_col_1 = p.createCollisionShape(p.GEOM_MESH, fileName=convex_path_1)
    obj_vis_1 = p.createVisualShape(p.GEOM_MESH, fileName=obj_mesh_1)

    initialPos_1 = com_1.tolist()

    obj_uid_1 = p.createMultiBody(
        baseMass=30.0,  # 질량 [kg]
        baseCollisionShapeIndex=obj_col_1,
        baseVisualShapeIndex=obj_vis_1,
        basePosition=[0, 0, 0],  # 월드 좌표
        baseOrientation=quat,
        baseInertialFramePosition=initialPos_1,  # ★ CoM
        baseInertialFrameOrientation=quat_I_1,
    )
    ply_path_1 = os.path.abspath("realscene_2/point_cloud_obj/iteration_67000/point_cloud_1.ply")
    p.changeDynamics(obj_uid_1, -1,
                     lateralFriction=0.9,
                     rollingFriction=0.03, spinningFriction=0.03,
                     linearDamping=0.05, angularDamping=0.05,
                     frictionAnchor=1,
                     activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
    print("obj created")

    #2
    obj_mesh_2 = "realscene_2/point_cloud_obj/iteration_67000/point_cloud_2.obj"
    if not os.path.exists("realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_2.obj"):
        # (권장) 먼저 볼록 분해: 결과 .vhacd.obj 가 자동 저장됨
        ok = p.vhacd(obj_mesh_2, "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_2.obj", "log.txt",
                     concavity=0.002, resolution=1_000_000)
        if not ok:
            raise RuntimeError("VHACD 실패—로그 확인")

    convex_path_2 = "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_2.obj"
    obj_mesh_2 = "realscene_2/point_cloud_obj/iteration_67000/obj_vhacd_2.obj"

    mesh_2 = trimesh.load(convex_path_2, force='mesh')
    com_2 = mesh_2.center_mass  # [x,y,z]  (메시 로컬 기준)

    I_tensor_2 = mesh_2.moment_inertia  # 3×3, CoM 기준
    eigval, eigvec = np.linalg.eigh(I_tensor_2)  # 주축 분해
    # 행렬→쿼터니언

    quat_I_2 = quaternion_from_matrix(np.vstack([np.hstack([eigvec, [[0], [0], [0]]]),
                                                 [0, 0, 0, 1]])[:3, :3]).tolist()

    obj_col_2 = p.createCollisionShape(p.GEOM_MESH, fileName=convex_path_2)
    obj_vis_2 = p.createVisualShape(p.GEOM_MESH, fileName=obj_mesh_2)

    initialPos_2 = com_2.tolist()

    obj_uid_2 = p.createMultiBody(
        baseMass=30.0,  # 질량 [kg]
        baseCollisionShapeIndex=obj_col_2,
        baseVisualShapeIndex=obj_vis_2,
        basePosition=[0, 0, 0],  # 월드 좌표
        baseOrientation=quat,
        baseInertialFramePosition=initialPos_2,  # ★ CoM
        baseInertialFrameOrientation=quat_I_2,
    )
    ply_path_2 = os.path.abspath("realscene_2/point_cloud_obj/iteration_67000/point_cloud_2.ply")
    p.changeDynamics(obj_uid_2, -1,
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

    splatviz.register_dynamic_object(ply_path_1, obj_uid_1, com_1, quat_I_1, init_world_pos = [0, 0, 0], init_world_quat = quat)
    splatviz.register_dynamic_object(ply_path_2, obj_uid_2, com_2, quat_I_2, init_world_pos = [0, 0, 0], init_world_quat = quat)



    model_ply = checkpoint_path + "point_cloud/iteration_30000/point_cloud.ply"
    id_module_ckpt = checkpoint_path + "id_module.th"
    predictor = None
    if os.path.isfile(model_ply) and os.path.isfile(id_module_ckpt):
        predictor = SixDGSPredictor(model_ply, id_module_ckpt)
    else:
        print("Warning: 6DGS model or weights path not provided. External pose prediction disabled.")

    # 영상 입력 초기화 (웹캠 또는 비디오 파일)
    cap = None
    if predictor is not None:
        if video == "":
            cap_index = 0
            cap = cv2.VideoCapture(cap_index)
        else:
            # video가 숫자 문자열이면 웹캠 인덱스로, 아니면 파일 경로로 처리
            if video.isdigit():
                cap_index = int(video)
                cap = cv2.VideoCapture(cap_index)
            else:
                cap = cv2.VideoCapture(video)
        if not cap or not cap.isOpened():
            print(f"Error: cannot open video source `{video}`")
            cap = None

    # 별도 쓰레드로 카메라 프레임 캡처 및 포즈 예측
    from collections import deque

    # -------- 공유 버퍼 ----------
    frame_buf = deque(maxlen=1)  # 길이 1 → 항상 '가장 최신' 프레임만 보관
    pose_buf = dict(mat=None, new=False)
    buf_lock = threading.Lock()
    stop_evt = threading.Event()

    def capture_loop():
        while not stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                break  # EOF or error
            with buf_lock:
                frame_buf.append(frame)  # 최신 프레임 덮어쓰기
            cv2.imshow("Video", frame)  # 실시간 표시
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                stop_evt.set()
                break

    def predict_loop():
        last_seen_id = 0
        while not stop_evt.is_set():
            with buf_lock:
                if not frame_buf:  # 아직 프레임 없음
                    continue
                frame = frame_buf[-1].copy()  # 가장 최신 프레임 스냅샷
                fid = id(frame)  # 객체 ID로 “신규 여부” 판별
            if fid == last_seen_id:
                time.sleep(0.005)  # 이미 처리한 프레임 → 살짝 대기
                continue
            last_seen_id = fid

            pose = predictor.predict_pose(frame)  # ★ 딥러닝 추론 ★

            with buf_lock:
                pose_buf["mat"] = pose
                pose_buf["new"] = True
        stop_evt.set()

    # 쓰레드 시작
    if predictor is not None and cap is not None:
        # 1) 캡처·디스플레이
        cap_th = threading.Thread(target=capture_loop, daemon=True, name="capture")
        # 2) 포즈 예측
        pred_th = threading.Thread(target=predict_loop, daemon=True, name="predict")

        cap_th.start()
        pred_th.start()
    else:
        cap_th = None
        pred_th = None

    while not splatviz.should_close():
        p.stepSimulation()
        splatviz.sync_dynamic_objects(scene_origin_pos=[0, 0, 0], scene_origin_quat=quat)
        if predictor is not None:
            with buf_lock:
                if pose_buf["new"]:
                    splatviz.set_external_camera_pose(pose_buf["mat"])
                    pose_buf["new"] = False
        splatviz.draw_frame()
    # 루프 종료: 자원 해제 및 쓰레드 정리
    if cap_th and pred_th is not None:
        cap_th.join(timeout=1.0)
        pred_th.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
    splatviz.close()

if __name__ == '__main__':
    main()
