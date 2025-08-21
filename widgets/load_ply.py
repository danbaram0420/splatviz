import os
import math
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from tkinter import Tk, filedialog
import pybullet as p
import numpy as np
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
import shutil

def create_physics_object_from_mesh(obj_path, global_quat, spawn_pos):
    """
    obj_path          : *.obj (아무 이름이나 OK)
    global_quat       : scene 전역 회전 (w,x,y,z)
    spawn_pos         : world 좌표계 xyz
    return            : (bullet_id, com(np.ndarray[3]), quat_I(list[4]))
    """
    import trimesh, numpy as np
    from trimesh.transformations import quaternion_from_matrix

    # 1) Convex mesh 경로 결정 -------------------------------
    base, ext = os.path.splitext(obj_path)
    vhacd_path = f"{base}_vhacd.obj"          # obj와 같은 폴더/이름 뒤에 _vhacd
    # 이미 convex 파일이 있거나 원본이 *_vhacd.obj 이면 그대로 사용
    convex = obj_path if obj_path.endswith("_vhacd.obj") or os.path.exists(vhacd_path) else vhacd_path

    # 2) 아직 convex 파일이 없으면 vHACD 실행(한 번만) ---------
    if convex == vhacd_path and not os.path.exists(convex):
        try:
            p.vhacd(obj_path, convex, "vhacd_log.txt",
                    resolution=250_000, concavity=0.002)   # resolution down → 메모리 폭주 방지
        except Exception as e:
            print("[VHACD] 실패, 원본(mesh)으로 대체:", e)
            convex = obj_path          # fallback (크래시 위험 ↓ 위해 mass=0 처리 예정)

    # 3) 물성치, 관성 ----------------------------------------
    mesh      = trimesh.load(convex, force='mesh')
    com       = mesh.center_mass
    eigval, eigvec = np.linalg.eigh(mesh.moment_inertia)
    quat_I    = quaternion_from_matrix(eigvec).tolist()  # baseInertialFrameOrientation (w,x,y,z)

    # 4) Bullet shape/바디 생성 ------------------------------
    #    동적(질량>0) → concave flag 절대 사용 X
    flag = 0
    cid  = p.createCollisionShape(p.GEOM_MESH, fileName=convex, flags=flag)
    vid  = p.createVisualShape   (p.GEOM_MESH, fileName=obj_path)


    bid  = p.createMultiBody(baseMass=20.0,
                             baseCollisionShapeIndex=cid,
                             baseVisualShapeIndex=vid,
                             basePosition=spawn_pos.tolist(),
                             baseOrientation=global_quat,
                             baseInertialFramePosition=com.tolist(),
                             baseInertialFrameOrientation=quat_I)
    return bid, com, quat_I

class LoadWidget(Widget):
    def __init__(self, viz, root, initial_files=None):
        super().__init__(viz, "Load")
        self.root = root
        self.filter = ""
        # If an initial file list is provided (scene + objects), use that.
        if initial_files is not None:
            # Use provided list as available items
            self.items = [os.path.abspath(path) for path in initial_files]
            if len(self.items) == 0:
                raise FileNotFoundError("No .ply files provided in initial_files!")
            # Set all initial files to be loaded
            self.plys: list[str] = list(self.items)
        else:
            # Original behavior: scan directory for .ply files
            self.items = self.list_runs_and_pkls()
            if len(self.items) == 0:
                raise FileNotFoundError(
                    f"No .ply or compression_config.yml found in '{root}' with filter '{self.filter}'")
            self.plys: list[str] = [self.items[0]]
        self.use_splitscreen = False
        self.highlight_border = False
        self.called_objects = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Search Filters (comma separated)")
            _changed, self.filter = imgui.input_text("##Filter", self.filter)
            plys_to_remove = []

            for i, ply in enumerate(self.plys):
                if imgui.begin_popup(f"browse_pkls_popup{i}"):
                    for item in self.items:
                        clicked = imgui.menu_item_simple(os.path.relpath(item, self.root))
                        if clicked:
                            self.plys[i] = item
                    imgui.end_popup()

                if imgui_utils.button(f"Browse {i + 1}", width=viz.button_w):
                    imgui.open_popup(f"browse_pkls_popup{i}")
                    self.items = self.list_runs_and_pkls()
                imgui.same_line()
                if i > 0:
                    if imgui_utils.button(f"Remove {i + 1}", width=viz.button_w):
                        plys_to_remove.append(i)
                    imgui.same_line()
                imgui.text(f"Scene {i + 1}: " + ply[len(self.root) :])

            for i in plys_to_remove[::-1]:
                self.plys.pop(i)
            if imgui_utils.button("Add Scene", width=viz.button_w):
                self.plys.append(self.plys[-1])

            if imgui_utils.button("Insert Object", width=viz.button_w):
                # 파일 다이얼로그 대신 고정 경로의 파일을 자동 로드
                file_path = os.path.abspath("objects/pokeball/point_cloud.ply")
                obj_path = os.path.abspath("objects/pokeball/obj_vhacd.obj")
                if file_path not in self.plys:
                    self.plys.append(file_path)
                else:
                    dir_name, file_name = os.path.split(file_path)
                    base_name, ext = os.path.splitext(file_name)
                    new_file_name = f"{base_name}_{self.called_objects}{ext}"
                    new_file_path = os.path.join(dir_name, new_file_name)
                    shutil.copyfile(file_path, new_file_path)
                    self.plys.append(new_file_path)
                    file_path = new_file_path
                    # 선택된 .ply를 장면 목록에 추가
                if not os.path.exists(file_path) or not os.path.exists(obj_path):
                    print(f"Object files not found in 'objects/pokeball' directory")
                else:
                    cam_widget = self.viz.widgets[1]  # 일반적으로 index 1이 Camera 위젯
                    cam_pos = cam_widget.cam_pos.cpu().numpy()
                    forward = cam_widget.forward.cpu().numpy()
                    global_quat = p.getQuaternionFromEuler([math.radians(viz.rotation), 0, 0])
                    cam_pos_world = R.from_quat(global_quat).apply(cam_pos)
                    forward_world = R.from_quat(global_quat).apply(forward)
                    spawn_pos_world = cam_pos_world + forward_world / np.linalg.norm(forward_world) * 1.0
                    # PyBullet에 오브젝트 생성 (convex mesh 사용)
                    bid, com, quat_I = create_physics_object_from_mesh(obj_path, global_quat, spawn_pos_world)
                    # Splatviz에 동적 오브젝트로 등록 (초기 pose 지정)
                    self.viz.register_dynamic_object(file_path, bid, com, quat_I,
                                                     init_world_pos=spawn_pos_world,
                                                     init_world_quat=global_quat,
                                                     obj_path=obj_path)
                    # 물체에 전방(force) 힘 가하여 살짝 밀기 (물체 초기화면에서 보이도록)
                    forward_world = R.from_quat(global_quat).apply(forward)
                    F = 10000.0  # 힘의 세기
                    com_world = spawn_pos_world + R.from_quat(global_quat).apply(com)
                    p.applyExternalForce(bid, -1, forward_world * F, com_world, p.WORLD_FRAME)
                    self.called_objects += 1

            # [추가] 클릭해서 배치하는 모드
            if imgui_utils.button("Insert Object (Click to place)", width=viz.button_w):
                file_path, obj_path = self.prepare_object_files_for_insertion()
                if file_path and obj_path:
                    # 뷰어에게 "다음 왼쪽클릭으로 이 오브젝트를 소환"하도록 의사표시
                    self.viz.awaiting_spawn_click = True
                    self.viz.pending_spawn_files = (file_path, obj_path)
                    print("Click on the viewport to spawn the object (click-to-place mode enabled).")

            if len(self.plys) > 1:
                use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
                highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]

    def prepare_object_files_for_insertion(self):
        """
        클릭-소환을 위해 사용할 (ply 경로, obj 경로)만 '반환'한다.
        - 여기서는 self.plys에 추가하지 않는다. (실제 소환 성공 시 등록)
        - 이미 동일 ply가 목록에 있으면 _{N} 접미사로 복사본을 생성해 경로를 반환한다.
        """
        import os, shutil
        base_ply = os.path.abspath("objects/pokeball/point_cloud.ply")
        obj_path = os.path.abspath("objects/pokeball/obj_vhacd.obj")

        if not os.path.exists(base_ply) or not os.path.exists(obj_path):
            print("Object files not found in 'objects/pokeball' directory")
            return None, None

        # 기본 경로가 아직 목록에 없다면 우선 기본 경로 사용
        if base_ply not in self.plys:
            return base_ply, obj_path

        # 이미 사용 중이면, _{k} 접미사를 붙여 '목록에 없는' 파일명으로 복사
        dir_name, file_name = os.path.split(base_ply)
        base_name, ext = os.path.splitext(file_name)
        k = int(getattr(self, "called_objects", 0))
        while True:
            new_file_name = f"{base_name}_{k}{ext}"
            candidate = os.path.join(dir_name, new_file_name)
            if candidate not in self.plys:
                shutil.copyfile(base_ply, candidate)
                return candidate, obj_path
            k += 1

    def list_runs_and_pkls(self) -> list[str]:
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".ply") or file.endswith("compression_config.yml"):
                    current_path = os.path.join(root, file)
                    if all([filter in current_path for filter in self.filter.split(",")]):
                        self.items.append(str(current_path))
        return sorted(self.items)


