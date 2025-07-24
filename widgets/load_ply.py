import os
import math
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from tkinter import Tk, filedialog
import pybullet as p
from trimesh.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R

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
    density   = 700.0                               # kg/m³ 임의
    mass      = mesh.volume * density
    com       = mesh.center_mass
    eigval, eigvec = np.linalg.eigh(mesh.moment_inertia)
    quat_I    = quaternion_from_matrix(eigvec).tolist()  # baseInertialFrameOrientation (w,x,y,z)

    # 4) Bullet shape/바디 생성 ------------------------------
    #    동적(질량>0) → concave flag 절대 사용 X
    flag = 0
    cid  = p.createCollisionShape(p.GEOM_MESH, fileName=convex, flags=flag)
    vid  = p.createVisualShape   (p.GEOM_MESH, fileName=obj_path)

    # concave mesh를 fallback 으로 쓸 땐 mass=0으로 강제하여 crash 회피
    if mass <= 0 or flag != 0:
        mass = 0.0

    bid  = p.createMultiBody(baseMass=2.0,
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
                # Open an OS file dialog for the user to select a .ply file
                root = Tk()
                root.withdraw()  # hide the root Tk window
                file_path = filedialog.askopenfilename(filetypes=[("Point Cloud Files", "*.ply")])
                if file_path:
                    file_path = os.path.abspath(file_path)
                    # Add the selected .ply to the list of loaded scenes/objects
                    self.plys.append(file_path)
                    # Find a corresponding .obj file in the same directory (assume one exists)
                    obj_path = None
                    folder = os.path.dirname(file_path)
                    for fname in os.listdir(folder):
                        if fname.lower().endswith(".obj"):
                            obj_path = os.path.join(folder, fname)
                            break
                    if obj_path is None:
                        print(f"No .obj file found in folder: {folder}")
                    else:
                        cam_widget = self.viz.widgets[1]  # Camera widget is typically at index 1
                        cam_pos = cam_widget.cam_pos.cpu().numpy()  # camera position (world coordinates)
                        forward = cam_widget.forward.cpu().numpy()  # camera forward unit vector
                        spawn_pos = cam_pos + forward * 1.0  # spawn 2.0 units in front of camera
                        # Use the global rotation offset (210° about X-axis) for orientation so object aligns with scene
                        global_quat = p.getQuaternionFromEuler([math.radians(210), 0, 0])
                        spawn_pos_world = R.from_quat(global_quat).apply(spawn_pos)
                        bid, com, quat_I = create_physics_object_from_mesh(obj_path,
                                                                           global_quat,
                                                                           spawn_pos_world)
                        # Splatviz 등록
                        self.viz.register_dynamic_object(file_path, bid, com, quat_I,
                                                         init_world_pos=spawn_pos_world,
                                                         init_world_quat=global_quat)
                        forward_world = R.from_quat(global_quat).apply(forward)
                        F = 1000.0  # [N] 원하는 세기
                        com_world = spawn_pos_world + R.from_quat(global_quat).apply(com)
                        p.applyExternalForce(bid,  # bodyUniqueId
                                             -1,  # base link
                                             forward_world * F,
                                             com_world,  # 힘 작용점 (COM)
                                             p.WORLD_FRAME)

            if len(self.plys) > 1:
                use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
                highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]

    def list_runs_and_pkls(self) -> list[str]:
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".ply") or file.endswith("compression_config.yml"):
                    current_path = os.path.join(root, file)
                    if all([filter in current_path for filter in self.filter.split(",")]):
                        self.items.append(str(current_path))
        return sorted(self.items)


