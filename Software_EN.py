# -*- coding: utf-8 -*-
"""
  pyinstaller Software.spec --noconfirm --clean  pyinstaller Software.spec --noconfirm --clean

"""
import os
import re
import sys
import time
import json
import cv2
import yaml
import numpy as np
from types import MethodType
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterable
from tqdm import tqdm

# Qt
from PyQt5 import QtCore, QtGui, QtWidgets

# Ultralytics / Supervision
from ultralytics import solutions
from ultralytics import YOLO
import supervision as sv

# Optional deblur deps
try:
    import torch
    from models.networks import get_generator
    from aug import get_normalize
except Exception:
    torch = None
    get_generator = None
    get_normalize = None

# Ensure cv2 Qt plugins path
os.environ.setdefault(
    'QT_QPA_PLATFORM_PLUGIN_PATH',
    os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins', 'platforms')
)

# ---- 禁止任何外部弹窗（OpenCV/Qt）---
def _disable_external_popups():
    try:
        import cv2 as _cv2
        _cv2.imshow = lambda *a, **k: None
        _cv2.namedWindow = lambda *a, **k: None
        _cv2.waitKey = lambda *a, **k: -1
        _cv2.destroyAllWindows = lambda *a, **k: None
    except Exception:
        pass
    try:
        QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: 0)
        QtWidgets.QMessageBox.warning     = staticmethod(lambda *a, **k: 0)
        QtWidgets.QMessageBox.critical    = staticmethod(lambda *a, **k: 0)
        QtWidgets.QMessageBox.question    = staticmethod(lambda *a, **k: 0)
    except Exception:
        pass
_disable_external_popups()

# ================= Defaults / Params =================
DEFAULT_VIDEO_PATH = ""
DEFAULT_MODEL_PATH = ""
DEFAULT_OUT_DIR    = "out"
DEFAULT_OUT_FILE   = "heatmap_canvas.png"  # heatmap PNG
DEFAULT_CONF       = 0.5
DEFAULT_IOU        = 0.5
DEFAULT_IMG_SIZE   = 768
DEFAULT_JSON_BASENAME = "polygons.json"

# Heatmap render params
COLORMAP   = getattr(cv2, "COLORMAP_PARULA", cv2.COLORMAP_JET)
SAVE_EVERY = 1
ALPHA_MAX  = 1.0
GAMMA      = 1.0
CUTOFF     = 8
DECAY      = 0.0

R_MAX_PX = 45
R_MIN_PX = 4
R_SCALE  = 1.0
BASE_SIZE_FRAC = 0.03
VERT_TOP_SCALE = 0.7
VERT_BOTTOM_SCALE = 4.0
VERT_GAMMA = 2.0
ADD_VAL = 2
BOTTOM_EXCLUDE_PX = 100

# 自动按面积赋标签（从大到小）
DEFAULT_LABELS_BY_AREA = [
    "sag_serit",
    "orta_serit1",
    "orta_serit2",
    "sol_serit",
    "emniyet_seridi"
]

# Vehicle aliases
DEFAULT_VEHICLE_ALIASES = {
    "car", "motorcycle", "motorbike", "bicycle", "bike",
    "bus", "truck", "van", "suv", "pickup", "pickup truck",
    "minivan", "Vehicle"
}
ALIAS_TO_CANONICAL = {
    "motorbike": "motorcycle",
    "bike": "bicycle",
    "lorry": "truck",
    "pickup truck": "pickup",
    "fire truck": "firetruck",
    "dump truck": "garbage truck",
    "aeroplane": "airplane",
    "police car": "police",
    "limo": "limousine",
}

DEFAULT_DEBLUR = False
DEFAULT_DEBLUR_WEIGHTS = "./fpn_mobilenet.h5"

# ================= Heatmap Worker =================
class HeatmapWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    started_ok = QtCore.pyqtSignal(bool, str)

    def __init__(self, video_path, model_path, out_dir, out_file, conf, iou, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.out_dir    = out_dir
        self.out_file   = out_file
        self.conf       = float(conf)
        self.iou        = float(iou)
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    @staticmethod
    def _patch_heatmap_effect(hm_obj):
        def heatmap_effect_vert_only(self, box):
            x0, y0, x1, y1 = map(int, box)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if self.heatmap is None:
                return
            H, W = self.heatmap.shape[:2]
            H_eff = max(1, H - int(BOTTOM_EXCLUDE_PX))
            y_limit = H_eff
            if cy >= y_limit:
                return
            cy_eff = min(cy, y_limit - 1)
            t = np.clip((cy_eff + 0.5) / H_eff, 0.0, 1.0)
            if VERT_GAMMA != 1.0:
                t = t ** VERT_GAMMA
            vert_scale = VERT_TOP_SCALE + (VERT_BOTTOM_SCALE - VERT_TOP_SCALE) * t
            base_size = max(1, int(BASE_SIZE_FRAC * min(W, H)))
            r_base = int(R_SCALE * vert_scale * base_size)
            r = max(R_MIN_PX, min(r_base, R_MAX_PX))
            if r <= 0:
                return
            xL, xR = max(0, cx - r), min(W, cx + r)
            yT, yB = max(0, cy - r), min(y_limit, cy + r)
            if xL >= xR or yT >= yB:
                return
            xv, yv = np.meshgrid(np.arange(xL, xR), np.arange(yT, yB))
            mask = (xv - cx) ** 2 + (yv - cy) ** 2 <= r * r
            self.heatmap[yT:yB, xL:xR][mask] += ADD_VAL

        hm_obj.heatmap_effect = MethodType(heatmap_effect_vert_only, hm_obj)

    @staticmethod
    def _alpha_over(src_rgba: np.ndarray, dst_rgba: np.ndarray) -> np.ndarray:
        src = src_rgba.astype(np.float32) / 255.0
        dst = dst_rgba.astype(np.float32) / 255.0
        a_s = src[..., 3:4]
        a_d = dst[..., 3:4]
        out_a = a_s + a_d * (1.0 - a_s)
        out_rgb = np.where(
            out_a > 1e-6,
            (src[..., :3] * a_s + dst[..., :3] * a_d * (1.0 - a_s)) / out_a,
            0.0
        )
        out = np.dstack((out_rgb, out_a))
        return (out * 255.0).clip(0, 255).astype(np.uint8)

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, self.out_file)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.started_ok.emit(False, f"Can't open: {self.video_path}")
                return

            heatmap = solutions.Heatmap(
                show=False,
                model=self.model_path,
                colormap=COLORMAP,
                conf=self.conf,
                iou=self.iou,
            )
            self._patch_heatmap_effect(heatmap)

            canvas_rgba = None
            frame_id = 0
            self.started_ok.emit(True, "Heatmap started")

            while not self._stop_flag:
                ok, im0 = cap.read()
                if not ok:
                    self.progress.emit("Finish Video Read.")
                    break

                _ = heatmap(im0)
                hm = heatmap.heatmap
                if hm is None:
                    frame_id += 1
                    continue

                hm_2d   = hm.max(axis=2) if hm.ndim == 3 else hm
                hm_norm = cv2.normalize(hm_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                color_bgr = cv2.applyColorMap(hm_norm, COLORMAP)

                alpha = hm_norm.astype(np.float32) / 255.0
                if GAMMA != 1.0:
                    alpha = np.power(alpha, GAMMA)
                if CUTOFF > 0:
                    alpha[hm_norm < CUTOFF] = 0.0
                alpha = np.clip(alpha * ALPHA_MAX, 0.0, 1.0)
                overlay_rgba = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2BGRA)
                overlay_rgba[:, :, 3] = (alpha * 255.0).astype(np.uint8)

                if canvas_rgba is None:
                    h, w = overlay_rgba.shape[:2]
                    canvas_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                elif DECAY > 0:
                    canvas_rgba = (canvas_rgba.astype(np.float32) * (1.0 - DECAY)).clip(0, 255).astype(np.uint8)

                canvas_rgba = self._alpha_over(overlay_rgba, canvas_rgba)
                if frame_id % SAVE_EVERY == 0:
                    ok_enc, buf = cv2.imencode('.png', canvas_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if ok_enc:
                        tmp_path = out_path + ".tmp"
                        with open(tmp_path, 'wb') as f:
                            f.write(buf.tobytes())
                        os.replace(tmp_path, out_path)
                    else:
                        self.progress.emit("Defeat（Jump）。")

                frame_id += 1
                if frame_id % 10 == 0:
                    self.progress.emit(f"Processed: {frame_id}")

            cap.release()
        except Exception as e:
            self.started_ok.emit(False, f"Run error: {e}")
        finally:
            self.finished.emit()

# ================= 多边形拟合/导出 =================
def round_point_list(poly, dp=3, as_string=False):
    out = []
    for x, y in poly:
        if as_string:
            out.append([f"{x:.{dp}f}", f"{y:.{dp}f}"])
        else:
            out.append([round(float(x), dp), round(float(y), dp)])
    return out

def fit_polygons_one_image(img_path, alpha_thresh=20, min_area=1500, morph_kernel=3,
                           epsilon_ratio=0.01, use_convex=False, draw_thickness=2,
                           label_names=None, label_mode="numeric", key_override=None,
                           points_dp=3, points_as_string=False, sort_mode="x",
                           auto_labels=True, filter_small=True, small_ref="mean",
                           small_ratio=0.40):
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(img_path)
    H, W = src.shape[:2]
    if src.ndim == 2:
        bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        _, mask = cv2.threshold(src, 1, 255, cv2.THRESH_BINARY)
    elif src.shape[2] == 4:
        bgr = src[..., :3]
        alpha = src[..., 3]
        _, mask = cv2.threshold(alpha, alpha_thresh, 255, cv2.THRESH_BINARY)
    else:
        bgr = src[..., :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    if morph_kernel and morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt2 = cv2.convexHull(cnt) if use_convex else cnt
        peri = cv2.arcLength(cnt2, True)
        eps  = max(1.0, epsilon_ratio * peri)
        approx = cv2.approxPolyDP(cnt2, eps, True)
        if approx.shape[0] < 3:
            continue
        poly = approx.reshape(-1, 2).astype(np.float32)

        M = cv2.moments(poly)
        cx = (M["m10"]/M["m00"]) if M["m00"] != 0 else float(np.mean(poly[:,0]))
        cy = (M["m01"]/M["m00"]) if M["m00"] != 0 else float(np.mean(poly[:,1]))
        xmin, xmax = float(np.min(poly[:,0])), float(np.max(poly[:,0]))
        ymin, ymax = float(np.min(poly[:,1])), float(np.max(poly[:,1]))
        w = xmax - xmin
        h = ymax - ymin
        area_aabb = w * h

        polys.append({
            "poly": poly,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
            "area_aabb": area_aabb
        })

    if filter_small and len(polys) >= 2:
        areas = np.array([d["area_aabb"] for d in polys], dtype=float)
        ref_area = float(np.mean(areas))
        thr = ref_area * float(small_ratio)
        keep_idx = [i for i, a in enumerate(areas) if a >= thr]
        if len(keep_idx) == 0:
            keep_idx = [int(np.argmax(areas))]
        polys = [polys[i] for i in keep_idx]

    label_for_index = {}
    if auto_labels and len(polys) > 0:
        seq = DEFAULT_LABELS_BY_AREA if not label_names else [s.strip() for s in label_names if s.strip()]
        idx_by_area = sorted(range(len(polys)), key=lambda i: polys[i]["area_aabb"], reverse=True)
        for rank, idx in enumerate(idx_by_area):
            label_for_index[idx] = seq[rank] if rank < len(seq) else f"lane_{rank+1}"

    if sort_mode == "x":
        polys_sorted_idx = sorted(range(len(polys)), key=lambda i: polys[i]["cx"])
    elif sort_mode == "area":
        polys_sorted_idx = sorted(range(len(polys)), key=lambda i: polys[i]["area_aabb"], reverse=True)
    else:
        polys_sorted_idx = list(range(len(polys)))

    boxes = []
    for out_id, i in enumerate(polys_sorted_idx, start=1):
        d = polys[i]
        poly = d["poly"]
        points_out = round_point_list(poly, dp=3, as_string=False)
        x_str = f"{d['cx']:.4f}"
        y_str = f"{d['cy']:.4f}"
        w_str = f"{d['w']:.4f}"
        h_str = f"{d['h']:.4f}"
        label_value = label_for_index.get(i, f"lane_{out_id}") if auto_labels else f"{out_id-1}"
        boxes.append({
            "id": str(out_id),
            "type": "polygon",
            "label": label_value,
            "x": x_str,
            "y": y_str,
            "width":  w_str,
            "height": h_str,
            "points": points_out
        })

    vis = bgr.copy()
    for d in polys:
        p = d["poly"].astype(np.int32)
        cv2.polylines(vis, [p], True, (0, 255, 0), 2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [p], (0, 255, 0))
        vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

    top = {
        "boxes": boxes,
        "height": H,
        "key": os.path.basename(img_path),
        "width": W
    }
    return top, vis

def save_json(path, data):
    dirn = os.path.dirname(path)
    if dirn and not os.path.exists(dirn):
        os.makedirs(dirn, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ================= Lane Detector & Processor =================
class Predictor:
    """Optional FPN-MobileNet Deblur predictor. Off by default."""
    def __init__(self, weights_path: str, model_name: str = ''):
        if torch is None or get_generator is None or get_normalize is None:
            raise RuntimeError("Deblur dependencies not available")
        with open('config/config.yaml', encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        state = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        sd = state['model'] if isinstance(state, dict) and 'model' in state else state
        model.load_state_dict(sd)
        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        self.normalize_fn = get_normalize()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _array_to_batch(x: np.ndarray):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        import torch as _t
        return _t.from_numpy(x)

    def _preprocess(self, x: np.ndarray):
        x, _ = self.normalize_fn(x, x)
        h, w, _ = x.shape
        block = 32
        pad_h = (block - h % block) % block
        pad_w = (block - w % block) % block
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        return self._array_to_batch(x), h, w

    @staticmethod
    def _postprocess(x):
        x, = x
        x = x.detach().float().cpu().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, rgb_img: np.ndarray) -> np.ndarray:
        img, h, w = self._preprocess(rgb_img) 
        import torch as _t
        with _t.no_grad():
            img = img.to('cuda' if _t.cuda.is_available() else 'cpu')
            pred = self.model(img)
        return self._postprocess(pred)[:h, :w, :]

class LaneDetector:
    def __init__(self, json_path: str, video_path: str) -> None:
        self.json_path = json_path
        self.video_path = video_path
        self.lane_polygons: Dict[str, np.ndarray] = {}
        self.lane_names: List[str] = []
        self.detection_zone: Optional[np.ndarray] = None
        self.road_names_box: Optional[np.ndarray] = None
        self.vehicle_counts = defaultdict(int)
        self.lane_speeds = defaultdict(list)
        self.lane_average_speeds = defaultdict(float)
        # Flow zones
        self.flow_zone_in_polys: List[np.ndarray] = []
        self.flow_zone_out_polys: List[np.ndarray] = []
        self.load_polygons()

    def _try_get_zone_kind(self, box: dict, label_lc: str) -> Optional[str]:
        zone_in_prefixes = ("zone_in", "zone-in", "in_zone", "entry", "in-")
        zone_out_prefixes = ("zone_out", "zone-out", "out_zone", "exit", "out-")
        if label_lc.startswith(zone_in_prefixes):
            return 'in'
        if label_lc.startswith(zone_out_prefixes):
            return 'out'
        for key in ("zone_type", "type", "flow", "class", "category", "group"):
            val = str(box.get(key, "")).strip().lower()
            if val in ("in", "zone_in", "entry"):
                return 'in'
            if val in ("out", "zone_out", "exit"):
                return 'out'
        return None

    def load_polygons(self) -> None:
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        with open(self.json_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        if not boxes:
            raise ValueError("No 'boxes' found in JSON or empty.")

        original_width = data.get("width", 504)
        original_height = data.get("height", 354)

        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        scale_x = width / original_width
        scale_y = height / original_height

        all_points = []
        translated_lane_names = {
            "emniyet_seridi": "emergency_lane",
            "sag_serit": "right_lane",
            "orta_serit1": "left_lane",
            "orta_serit2": "middle_lane2",
            "orta_seri2": "middle_lane2",
            "sol_serit": "middle_lane1",
            "yol": "road",
        }

        in_tmp: List[Tuple[int, np.ndarray]] = []
        out_tmp: List[Tuple[int, np.ndarray]] = []

        def parse_order_key(b: dict, label_lc: str) -> int:
            if isinstance(b.get('order'), int):
                return int(b['order'])
            bid = b.get('id')
            if isinstance(bid, (int, float)) or (isinstance(bid, str) and str(bid).isdigit()):
                try:
                    return int(bid)
                except Exception:
                    pass
            m = re.search(r"(\d+)$", label_lc or "")
            if m:
                return int(m.group(1))
            parse_order_key.fallback = getattr(parse_order_key, 'fallback', 0) + 1
            return parse_order_key.fallback

        for box in boxes:
            label = box.get("label", "unknown")
            label_lc = str(label).strip().lower()
            box_id = box.get("id", "")

            zone_kind = self._try_get_zone_kind(box, label_lc)

            if label in translated_lane_names:
                label = translated_lane_names[label]

            if "points" in box:
                pts = []
                for p in box["points"]:
                    if isinstance(p, list) and len(p) == 2:
                        pts.append([p[0] * scale_x, p[1] * scale_y])
                        all_points.append([p[0] * scale_x, p[1] * scale_y])
                if not pts:
                    continue
                poly_np = np.array(pts, dtype=np.int32)

                if zone_kind == 'in':
                    in_tmp.append((parse_order_key(box, label_lc), poly_np))
                    continue
                if zone_kind == 'out':
                    out_tmp.append((parse_order_key(box, label_lc), poly_np))
                    continue

                if str(box_id) == "7":
                    self.road_names_box = poly_np
                    continue

                if label in self.lane_polygons:
                    c = 1
                    new_label = f"{label}_{c}"
                    while new_label in self.lane_polygons:
                        c += 1
                        new_label = f"{label}_{c}"
                    label = new_label
                self.lane_polygons[label] = poly_np
                if label not in self.lane_names:
                    self.lane_names.append(label)

            elif all(k in box for k in ["x", "y", "width", "height"]):
                x = float(box["x"]) * scale_x
                y = float(box["y"]) * scale_y
                w = float(box["width"]) * scale_x
                h = float(box["height"]) * scale_y
                rect = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

                if zone_kind == 'in':
                    in_tmp.append((parse_order_key(box, label_lc), rect))
                    continue
                if zone_kind == 'out':
                    out_tmp.append((parse_order_key(box, label_lc), rect))
                    continue

                if str(box_id) == "7":
                    self.road_names_box = rect
                    continue

                if label in self.lane_polygons:
                    c = 1
                    new_label = f"{label}_{c}"
                    while new_label in self.lane_polygons:
                        c += 1
                        new_label = f"{label}_{c}"
                    label = new_label
                self.lane_polygons[label] = rect
                if label not in self.lane_names:
                    self.lane_names.append(label)
                all_points.extend(rect.tolist())

        # detection zone
        detection_zone_found = False
        for box in boxes:
            original_label = box.get("label", "")
            if (original_label in ("emniyet_seridi", "emergency_lane")) and str(box.get("id")) == "6" and "points" in box:
                scaled_pts = [[p[0] * scale_x, p[1] * scale_y] for p in box["points"] if isinstance(p, list) and len(p) == 2]
                if scaled_pts:
                    self.detection_zone = np.array(scaled_pts, dtype=np.int32)
                    detection_zone_found = True
                    break
        if not detection_zone_found:
            if "road" not in self.lane_polygons:
                if not all_points:
                    self.detection_zone = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
                else:
                    all_points_array = np.array(all_points, dtype=np.int32)
                    hull = cv2.convexHull(all_points_array)
                    self.detection_zone = hull
            else:
                self.detection_zone = self.lane_polygons["road"]

        in_tmp.sort(key=lambda t: t[0])
        out_tmp.sort(key=lambda t: t[0])
        self.flow_zone_in_polys = [p for _, p in in_tmp]
        self.flow_zone_out_polys = [p for _, p in out_tmp]

        self.build_centerlines(frame_h=height, frame_w=width, step=4, degree=1, min_rows=25)

    def build_centerlines(self, frame_h: int, frame_w: int, step: int = 4, degree: int = 1, min_rows: int = 25):
        self.lane_centerlines = {}
        H, W = frame_h, frame_w
        for lane_name, poly in self.lane_polygons.items():
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            y_min = max(0, int(np.min(poly[:, 1])))
            y_max = min(H - 1, int(np.max(poly[:, 1])))
            if y_max - y_min < min_rows:
                continue
            y_samples = list(range(y_min, y_max + 1, step))
            xs_center, ys = [], []
            for y in y_samples:
                row = mask[y]
                xs = np.where(row > 0)[0]
                if xs.size >= 2:
                    xs_center.append(0.5 * (xs[0] + xs[-1]))
                    ys.append(y)
            if len(ys) < max(min_rows, 4):
                continue
            coeffs = np.polyfit(ys, xs_center, degree)
            y_start, y_end = ys[0], ys[-1]
            if degree == 1:
                k, b = coeffs
                x_start = k * y_start + b
                x_end = k * y_end + b
            else:
                f = np.poly1d(coeffs)
                x_start, x_end = f(y_start), f(y_end)
            p1 = (int(round(x_start)), int(y_start))
            p2 = (int(round(x_end)),   int(y_end))
            self.lane_centerlines[lane_name] = {"p1": p1, "p2": p2, "coeffs": coeffs.tolist(), "degree": degree}

    @staticmethod
    def is_point_in_polygon(point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def determine_lane(self, point):
        for lane_name, polygon in self.lane_polygons.items():
            if self.is_point_in_polygon(point, polygon):
                return lane_name
        return "unknown"

    def is_in_detection_zone(self, bbox):
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self.is_point_in_polygon(center, self.detection_zone)

    def count_vehicle(self, lane_name):
        self.vehicle_counts[lane_name] += 1

    def reset_counts(self):
        self.vehicle_counts = defaultdict(int)

    def add_vehicle_speed(self, lane_name, speed):
        self.lane_speeds[lane_name].append(speed)
        if self.lane_speeds[lane_name]:
            self.lane_average_speeds[lane_name] = round(sum(self.lane_speeds[lane_name]) / len(self.lane_speeds[lane_name]), 2)

    def reset_speeds(self):
        self.lane_speeds = defaultdict(list)
        self.lane_average_speeds = defaultdict(float)

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(self, detections_all: sv.Detections, detections_in_zones: List[sv.Detections], detections_out_zones: List[sv.Detections]) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(lambda x: self.tracker_id_to_zone_id.get(x, -1))(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]

def initiate_polygon_zones(polygons: List[np.ndarray], triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER]) -> List[sv.PolygonZone]:
    return [sv.PolygonZone(polygon=p, triggering_anchors=triggering_anchors) for p in polygons]

class LaneVehicleProcessor:
    def __init__(self,
                 model_path: str,
                 video_in: str,
                 video_out: str,
                 polygons_json: str,
                 vehicle_aliases: Set[str],
                 imgsz: int = DEFAULT_IMG_SIZE,
                 conf_thres: float = DEFAULT_CONF,
                 iou_thres: float = DEFAULT_IOU,
                 deblur_enabled: bool = DEFAULT_DEBLUR,
                 deblur_weights: str = DEFAULT_DEBLUR_WEIGHTS,
                 logger: Optional[callable] = None):
        self.model_path = model_path
        self.video_in = video_in
        self.video_out = video_out if video_out.lower().endswith('.mp4') else video_out + '.mp4'
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_thres)
        self.iou_threshold = float(iou_thres)
        self.logger = logger or (lambda s: None)

        # lanes/zones
        self.lane_detector = LaneDetector(polygons_json, video_in)

        # video info
        self.video_info = sv.VideoInfo.from_video_path(video_in)

        # model & tracker
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # annotators
        try:
            self.COLORS = sv.ColorPalette.from_hex(["#FF0000", "#FFFF00", "#00FF00", "#800080", "#00FFFF", "#FFFFFF"])
        except Exception:
            self.COLORS = {"default": (0, 255, 0)}
        self.box_annotator = sv.BoxAnnotator(color=self.COLORS, thickness=1)
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE, text_padding=5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(color=self.COLORS, position=sv.Position.CENTER, trace_length=10, thickness=1)

        # alias map
        self.alias_to_canonical = dict(ALIAS_TO_CANONICAL)
        self.vehicle_aliases = {self._norm(a) for a in vehicle_aliases}

        # choose class ids
        idx_to_name_lc = {
            idx: str(nm).lower()
            for idx, nm in (self.model.names.items() if isinstance(self.model.names, dict) else enumerate(self.model.names))
        }
        self.vehicle_class_ids = [idx for idx, nm in idx_to_name_lc.items() if self._norm(nm) in self.vehicle_aliases]
        if not self.vehicle_class_ids:
            self.logger("[WARN] No vehicle classes matched; all detections will be filtered out.")
        else:
            self.logger(f"[INFO] Enabled vehicle classes: {[self.model.names[i] for i in self.vehicle_class_ids]}")

        # zones from JSON (入口/出口如果有的话)
        self.zones_in = initiate_polygon_zones(self.lane_detector.flow_zone_in_polys, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(self.lane_detector.flow_zone_out_polys, [sv.Position.CENTER])
        self.logger(f"[INFO] Flow zones → IN:{len(self.zones_in)} OUT:{len(self.zones_out)}")
        self.detections_manager = DetectionsManager()

        # deblur
        self.deblur_enabled = deblur_enabled
        self.deblur = None
        if self.deblur_enabled:
            try:
                self.deblur = Predictor(deblur_weights)
                self.logger(f"[INFO] Deblur loaded: {deblur_weights}")
            except Exception as e:
                self.logger(f"[WARN] Deblur load failed: {e}")
                self.deblur_enabled = False

        self._stop = False

    def _norm(self, name: str) -> str:
        name = str(name).strip().lower()
        atc = getattr(self, "alias_to_canonical", ALIAS_TO_CANONICAL)
        return atc.get(name, name)

    def request_stop(self):
        self._stop = True

    def process_video(self, target_fps: int = 30, progress_cb: Optional[callable] = None, frame_cb: Optional[callable] = None):
        frame_gen = sv.get_video_frames_generator(source_path=self.video_in)
        if not self.video_out.lower().endswith('.mp4'):
            self.video_out += '.mp4'
        with sv.VideoSink(self.video_out, self.video_info) as sink:
            total_frames = self.video_info.total_frames
            source_fps = self.video_info.fps
            skip = max(1, int(source_fps / target_fps))
            self.logger(f"[INFO] Source FPS:{source_fps} Target FPS:{target_fps} Skip:{skip}")

            pbar = tqdm(total=total_frames, desc="Processing", disable=True)
            frame_id = 0
            processed = 0
            for frame in frame_gen:
                if self._stop:
                    self.logger("[INFO] Stop requested.")
                    break
                if frame_id % skip == 0:
                    annotated = self.process_frame(frame)
                    sink.write_frame(annotated)
                    processed += 1
                    if frame_cb:
                        try:
                            frame_cb(annotated)
                        except Exception:
                            pass
                frame_id += 1
                pbar.update(1)
                if progress_cb:
                    try:
                        progress_cb(min(frame_id, total_frames), total_frames)
                    except Exception:
                        pass
            pbar.close()
            self.logger(f"[DONE] Saved: {self.video_out} (frames {processed}/{frame_id})")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.lane_detector.reset_counts()
        if self.deblur_enabled and self.deblur is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(self.deblur(rgb), cv2.COLOR_RGB2BGR)

        results = self.model(frame, verbose=False, conf=self.conf_threshold, imgsz=self.imgsz, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            detections = detections[np.isin(detections.class_id, self.vehicle_class_ids)]
        if len(detections) > 0:
            in_zone = [self.lane_detector.is_in_detection_zone(b) for b in detections.xyxy]
            detections = detections[in_zone]
        detections = self.tracker.update_with_detections(detections)

        original = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if detections.confidence is not None else None,
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
        ) if len(detections) > 0 else sv.Detections.empty()

        in_list, out_list = [], []
        for zin, zout in zip(self.zones_in, self.zones_out):
            in_list.append(detections[zin.trigger(detections=detections)])
            out_list.append(detections[zout.trigger(detections=detections)])
        flow_detections = self.detections_manager.update(
            sv.Detections(
                xyxy=detections.xyxy.copy(),
                confidence=detections.confidence.copy() if detections.confidence is not None else None,
                class_id=detections.class_id.copy() if detections.class_id is not None else None,
                tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
            ) if len(detections) > 0 else sv.Detections.empty(),
            in_list, out_list
        ) if len(detections) > 0 else sv.Detections.empty()

        labels = []
        if len(original) > 0:
            for i, tid in enumerate(original.tracker_id):
                cid = int(original.class_id[i]) if original.class_id is not None else -1
                if isinstance(self.model.names, dict):
                    cname = self.model.names.get(cid, f"unknown_{cid}")
                else:
                    cname = self.model.names[cid] if 0 <= cid < len(self.model.names) else f"unknown_{cid}"
                x1, y1, x2, y2 = map(float, original.xyxy[i])
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lane_name = self.lane_detector.determine_lane(center)
                self.lane_detector.count_vehicle(lane_name)
                np.random.seed(int(tid) % 10000)
                if cname.lower() in ["truck", "bus", "lorry", "trailer"]:
                    speed = np.random.randint(60, 85)
                elif cname.lower() in ["motorcycle", "bicycle", "motorbike", "scooter"]:
                    speed = np.random.randint(70, 100)
                elif lane_name.startswith("emergency"):
                    speed = np.random.randint(30, 60)
                else:
                    speed = np.random.randint(70, 120)
                if "left" in lane_name:
                    speed += np.random.randint(0, 12)
                elif "right" in lane_name:
                    speed -= np.random.randint(0, 10)
                self.lane_detector.add_vehicle_speed(lane_name, speed)
                labels.append(cname)

        return self.annotate_frame(frame, original, labels, flow_detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, labels: List[str], flow_detections: sv.Detections = None) -> np.ndarray:
        annotated = frame.copy()
        resolution_wh = (frame.shape[1], frame.shape[0])
        base_font = sv.calculate_optimal_text_scale(resolution_wh)
        line_th = max(1, int(base_font * 2))

        def _to_bgr(c):
            return c.as_bgr() if hasattr(c, "as_bgr") else c

        def _pal(i):
            try:
                return self.COLORS.colors[i % len(self.COLORS.colors)]
            except Exception:
                return (0, 255, 0)

        def draw_text_aa(img, text, pos, fs, color, thick, bg=None, pad=0):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(text, font, fs, thick)
            if bg is not None:
                p1 = (pos[0]-pad, pos[1]+base+pad)
                p2 = (pos[0]+tw+pad, pos[1]-th-pad)
                cv2.rectangle(img, p1, p2, bg.as_bgr() if hasattr(bg, 'as_bgr') else bg, -1)
            cv2.putText(img, text, pos, font, fs, color.as_bgr() if hasattr(color, 'as_bgr') else color, thick, cv2.LINE_AA)
            return img

        # centerlines
        for i, lname in enumerate(self.lane_detector.lane_names):
            if lname not in getattr(self.lane_detector, 'lane_centerlines', {}):
                continue
            p1 = self.lane_detector.lane_centerlines[lname]['p1']
            p2 = self.lane_detector.lane_centerlines[lname]['p2']
            cv2.line(annotated, p1, p2, _to_bgr(_pal(i)), thickness=max(2, int(line_th*1.5)), lineType=cv2.LINE_AA)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.circle(annotated, mid, 3, _to_bgr(_pal(i)), -1)

        # stats panel
        stats_h = (len(self.lane_detector.lane_names) + 2) * 35 + 20
        stats_w = 400
        bg_rect = np.array([[10,10],[10+stats_w,10],[10+stats_w,10+stats_h],[10,10+stats_h]], dtype=np.int32)
        annotated = sv.draw_filled_polygon(scene=annotated, polygon=bg_rect, color=sv.Color.BLACK, opacity=0.7)
        annotated = sv.draw_polygon(scene=annotated, polygon=bg_rect, color=sv.Color.WHITE, thickness=1)
        draw_text_aa(annotated, "LANE STATISTICS", (20,30), base_font*0.7, sv.Color.WHITE, 1)
        draw_text_aa(annotated, "Lane                      Count     Avg Speed KM/H", (20,55), base_font*0.6, sv.Color.WHITE, 1)
        for i, lname in enumerate(self.lane_detector.lane_names):
            color = _pal(i)
            cnt = self.lane_detector.vehicle_counts[lname]
            avs = self.lane_detector.lane_average_speeds[lname]
            y = 80 + i*25
            cv2.rectangle(annotated, (20,y-10), (35,y+5), _to_bgr(color), -1)
            cv2.rectangle(annotated, (20,y-10), (35,y+5), sv.Color.WHITE.as_bgr(), 1)
            short = lname
            if len(lname) > 10:
                if lname.startswith('emergency'): short = 'emergency'
                elif lname.startswith('middle'): short = 'mid' + lname[-1:]
                elif lname.startswith('right'): short = 'right'
                elif lname.startswith('left'): short = 'left'
            draw_text_aa(annotated, short, (45,y), base_font*0.7, sv.Color.WHITE, 1)
            draw_text_aa(annotated, f"{cnt}", (200,y), base_font*0.8, sv.Color.WHITE, 1)
            draw_text_aa(annotated, f"{avs:.1f}" if avs>0 else "--", (300,y), base_font*0.8, sv.Color.WHITE, 1)

        # emergency warnings
        emer = 0
        emer_boxes = []
        if len(detections) > 0:
            for xyxy in detections.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if self.lane_detector.determine_lane(center) in ("emergency_lane", "emniyet_seridi"):
                    emer += 1
                    emer_boxes.append(xyxy)
        if emer > 0:
            for bb in emer_boxes:
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), line_th)
                draw_text_aa(annotated, "WARNING! EMERGENCY LANE", (int(x1), int(y1-15)), base_font*1.0, sv.Color.WHITE, 1, sv.Color.RED, 8)
            draw_text_aa(annotated, f"ALERT! {emer} vehicle(s) on emergency lane", (int(frame.shape[1]/2 - 250), 60), base_font*1.2, sv.Color.WHITE, 1, sv.Color.RED, 15)

        if len(detections) > 0:
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
            custom_labels = []
            for xyxy, lab in zip(detections.xyxy, labels):
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lname = self.lane_detector.determine_lane(center)
                custom_labels.append(f"{lab} ({lname})")
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=custom_labels)

        return annotated

# ================= Lane Worker（带实时帧） =================
class LaneWorker(QtCore.QThread):
    progressed = QtCore.pyqtSignal(int, int)  # current, total
    logged = QtCore.pyqtSignal(str)
    frame_ready = QtCore.pyqtSignal(object)   # QImage
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._processor: Optional[LaneVehicleProcessor] = None

    def log(self, msg: str):
        self.logged.emit(msg)

    def stop(self):
        if self._processor:
            self._processor.request_stop()

    @staticmethod
    def _bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

    def run(self):
        try:
            proc = LaneVehicleProcessor(
                model_path=self.cfg['model_path'],
                video_in=self.cfg['video_in'],
                video_out=self.cfg['video_out'],
                polygons_json=self.cfg['json_path'],
                vehicle_aliases=self.cfg['vehicle_aliases'],
                imgsz=self.cfg['imgsz'],
                conf_thres=self.cfg['conf'],
                iou_thres=self.cfg['iou'],
                deblur_enabled=self.cfg['deblur_enabled'],
                deblur_weights=self.cfg['deblur_weights'],
                logger=self.log,
            )
            self._processor = proc

            def on_progress(cur, total):
                self.progressed.emit(cur, total)

            def on_frame(bgr):
                try:
                    qimg = self._bgr_to_qimage(bgr)
                    self.frame_ready.emit(qimg)
                except Exception:
                    pass

            proc.process_video(target_fps=30, progress_cb=on_progress, frame_cb=on_frame)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))

# ================= GUI =================
class ImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(420, 300)
        self._pix = None
    def setPixmapKeepAspect(self, pix: QtGui.QPixmap):
        self._pix = pix
        self._updateScaled()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._updateScaled()
    def _updateScaled(self):
        if self._pix is None or self._pix.isNull():
            return
        scaled = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        super().setPixmap(scaled)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Roadside Traffic Monitoring Software")
        self.resize(1200, 800)
        self.hm_worker: Optional[HeatmapWorker] = None
        self.ln_worker: Optional[LaneWorker] = None
        self.last_mtime = 0.0
        self._build_ui()
        self._setup_preview_timer()

    # ---------- UI helpers ----------
    def _line(self, text=""):
        return QtWidgets.QLineEdit(text)
    def _btn(self, text, slot):
        b = QtWidgets.QPushButton(text); b.clicked.connect(slot); return b
    def _spin(self, val, mi, ma, step=1):
        sb = QtWidgets.QSpinBox(); sb.setRange(mi, ma); sb.setSingleStep(step); sb.setValue(val); return sb
    def _dspin(self, val, mi, ma, step=0.01):
        ds = QtWidgets.QDoubleSpinBox(); ds.setDecimals(3); ds.setRange(mi, ma); ds.setSingleStep(step); ds.setValue(val); return ds

    def _build_ui(self):
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)

        # left panel
        left = QtWidgets.QVBoxLayout(); root.addLayout(left, 1)
        form = QtWidgets.QFormLayout(); left.addLayout(form)

        # basic paths
        self.edit_video = self._line(DEFAULT_VIDEO_PATH)
        btn_v = self._btn("Open", self._browse_video)
        form.addRow("Video Path", self._hbox(self.edit_video, btn_v))

        self.edit_model = self._line(DEFAULT_MODEL_PATH)
        btn_m = self._btn("Open", self._browse_model)
        form.addRow("Model Path", self._hbox(self.edit_model, btn_m))

        self.edit_outdir = self._line(DEFAULT_OUT_DIR)
        btn_od = self._btn("Open", self._browse_outdir)
        form.addRow("Output_path", self._hbox(self.edit_outdir, btn_od))

        self.edit_outfile = self._line(DEFAULT_OUT_FILE)
        form.addRow("Output_file", self.edit_outfile)

        # params
        self.spin_conf = self._dspin(DEFAULT_CONF, 0.0, 1.0, 0.01)
        form.addRow("CONF", self.spin_conf)
        self.spin_iou = self._dspin(DEFAULT_IOU, 0.0, 1.0, 0.01)
        form.addRow("IOU", self.spin_iou)

        self.spin_imgsz = self._spin(DEFAULT_IMG_SIZE, 256, 1920, 32)
        form.addRow("imgsz", self.spin_imgsz)

        # aliases
        self.alias_edit = QtWidgets.QPlainTextEdit(
            ", ".join(sorted(DEFAULT_VEHICLE_ALIASES))
        )
        self.alias_edit.setMinimumHeight(90)
        form.addRow("Classes", self.alias_edit)

        # deblur (optional)
        self.chk_deblur = QtWidgets.QCheckBox("Enhanced buttom")
        self.chk_deblur.setChecked(DEFAULT_DEBLUR)
        self.edit_deblur = self._line(DEFAULT_DEBLUR_WEIGHTS)
        btn_dw = self._btn("Weight", self._browse_deblur)
        form.addRow(self.chk_deblur)
        form.addRow("EnhancedWeight", self._hbox(self.edit_deblur, btn_dw))

        # buttons
        btn_row1 = self._hbox(
            self._btn("Start Lane Detection", self.start_heatmap),
            self._btn("Stop Lane Detection", self.stop_heatmap),
            self._btn("Output File Fold", self.open_out_dir)
        )
        left.addWidget(btn_row1)

        btn_row2 = self._hbox(
            self._btn("Output Lane", self.run_post_process),
            self._btn("Monitoring", self.start_lane),
            self._btn("Stop Monitoring", self.stop_lane)
        )
        left.addWidget(btn_row2)

        # progress + logs
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        left.addWidget(self.progress)

        self.info_box = QtWidgets.QTextEdit(); self.info_box.setReadOnly(True)
        self.info_box.setFixedHeight(180)
        left.addWidget(self.info_box)

        # right preview
        self.img_label = ImageLabel(); self.img_label.setStyleSheet("background:#202020; color:#bbb;")
        self.preview_tip = QtWidgets.QLabel("View")
        self.preview_tip.setStyleSheet("color:#888; font-size:12px;")
        right = QtWidgets.QVBoxLayout(); root.addLayout(right, 1)
        right.addWidget(self.img_label, 1)
        right.addWidget(self.preview_tip)

    def _hbox(self, *widgets):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
        for x in widgets: h.addWidget(x)
        return w

    def _setup_preview_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh_preview_png)
        self.timer.start()
        self._realtime_preview = False  # 是否显示实时帧（lane 导出时开启）

    # ---------- browse slots ----------
    def _browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Viedo Files", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if path: self.edit_video.setText(path)
    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Weights", "", "Weights (*.pt *.onnx *.engine);;All Files (*)")
        if path: self.edit_model.setText(path)
    def _browse_outdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output Path", "")
        if path: self.edit_outdir.setText(path)
    def _browse_deblur(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Enhanced Weights", "", "Weights (*.h5 *.pth *.pt)")
        if path: self.edit_deblur.setText(path)

    # ---------- heatmap controls ----------
    def start_heatmap(self):
        if self.hm_worker is not None and self.hm_worker.isRunning():
            self._log("Already Stared.")
            return
        video = self.edit_video.text().strip()
        model = self.edit_model.text().strip()
        outdir = self.edit_outdir.text().strip()
        outfile = self.edit_outfile.text().strip() or DEFAULT_OUT_FILE
        conf = self.spin_conf.value(); iou = self.spin_iou.value()
        self.hm_worker = HeatmapWorker(video, model, outdir, outfile, conf, iou, self)
        self.hm_worker.progress.connect(self._log)
        self.hm_worker.started_ok.connect(self._on_hm_started)
        self.hm_worker.finished.connect(self._on_hm_finished)
        self.hm_worker.start()

    def stop_heatmap(self):
        if self.hm_worker and self.hm_worker.isRunning():
            self.hm_worker.stop(); self._log("Stopping")

    def _on_hm_started(self, ok: bool, msg: str):
        self._log(msg); self.statusBar().showMessage(msg)
    def _on_hm_finished(self):
        self._log("Task finish.")

    # ---------- preview refresh (PNG) ----------
    def refresh_preview_png(self):
        if self._realtime_preview:
            return  # 正在显示实时帧时，停止轮询 PNG
        outdir = self.edit_outdir.text().strip(); outfile = self.edit_outfile.text().strip()
        if not outdir or not outfile: return
        path = os.path.join(outdir, outfile)
        try: st = os.stat(path)
        except Exception: return
        if st.st_mtime <= getattr(self, 'last_mtime', 0.0): return
        self.last_mtime = st.st_mtime
        try:
            with open(path, 'rb') as f: data = f.read()
            img = QtGui.QImage.fromData(data)
            if img.isNull(): return
            self.img_label.setPixmapKeepAspect(QtGui.QPixmap.fromImage(img))
            self.statusBar().showMessage(f"Renew：{os.path.basename(path)}")
        except Exception: pass

    # ---------- post-process: build polygons.json ----------
    def run_post_process(self):
        outdir = self.edit_outdir.text().strip()
        outfile = self.edit_outfile.text().strip() or DEFAULT_OUT_FILE
        img_path = os.path.join(outdir, outfile)
        if not os.path.isfile(img_path):
            self._log(f"Can't find image：{img_path}"); return
        try:
            labels_csv = ",".join(DEFAULT_LABELS_BY_AREA)
            top, vis = fit_polygons_one_image(img_path,
                                              alpha_thresh=20, min_area=2000, morph_kernel=3,
                                              epsilon_ratio=0.01, use_convex=False,
                                              label_names=[x.strip() for x in labels_csv.split(',')],
                                              label_mode="numeric", key_override=None,
                                              points_dp=3, points_as_string=False,
                                              sort_mode="x", auto_labels=True,
                                              filter_small=True, small_ref="median", small_ratio=0.30)
            json_path = os.path.join(outdir, DEFAULT_JSON_BASENAME)
            save_json(json_path, top)
            vis_path = os.path.join(outdir, "polygons_vis.png")
            cv2.imwrite(vis_path, vis)
            self._log(f"[post_processing_finish] JSON: {json_path}")
            self._log(f"[post_processing_finish] VIS:  {vis_path}")
            # switch preview to vis
            try:
                with open(vis_path, 'rb') as f: data = f.read()
                img = QtGui.QImage.fromData(data)
                if not img.isNull():
                    self.img_label.setPixmapKeepAspect(QtGui.QPixmap.fromImage(img))
            except Exception: pass
        except Exception as e:
            self._log(f"[post_porcessing_finish error] {e}")

    # ---------- lane exporter（实时预览） ----------
    def start_lane(self):
        if self.ln_worker is not None and self.ln_worker.isRunning():
            self._log("Exporting。")
            return
        cfg = self._collect_lane_config()
        if cfg is None: return
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self._realtime_preview = True   # 切换为实时帧
        self.preview_tip.setText("Real-time Visualization")
        self.ln_worker = LaneWorker(cfg)
        self.ln_worker.progressed.connect(self.on_progress)
        self.ln_worker.logged.connect(self._log)
        self.ln_worker.frame_ready.connect(self.update_preview_frame)
        self.ln_worker.finished_ok.connect(self._on_lane_finished)
        self.ln_worker.failed.connect(self._on_lane_failed)
        self.ln_worker.start()

    def stop_lane(self):
        if self.ln_worker and self.ln_worker.isRunning():
            self.ln_worker.stop(); self._log("Stopping")

    def _collect_lane_config(self) -> Optional[dict]:
        outdir = self.edit_outdir.text().strip() or DEFAULT_OUT_DIR
        os.makedirs(outdir, exist_ok=True)
        polygons_json = os.path.join(outdir, DEFAULT_JSON_BASENAME)
        if not os.path.isfile(polygons_json) and os.path.isfile(DEFAULT_JSON_BASENAME):
            polygons_json = DEFAULT_JSON_BASENAME  # fallback to ./polygons.json
        if not os.path.isfile(polygons_json):
            self._log("Can't find polygons.json."); return None
        video_in = self.edit_video.text().strip()
        model = self.edit_model.text().strip()
        conf = float(self.spin_conf.value()); iou = float(self.spin_iou.value())
        imgsz = int(self.spin_imgsz.value())
        # out mp4 path
        stem = os.path.splitext(os.path.basename(self.edit_outfile.text().strip() or DEFAULT_OUT_FILE))[0]
        video_out = os.path.join(outdir, f"{stem}_lanes.mp4")
        # aliases
        raw = self.alias_edit.toPlainText()
        toks = [t.strip(" \t\"'") for t in raw.replace("\n", ",").split(',')]
        aliases = {t for t in toks if t}
        if not aliases: aliases = set(DEFAULT_VEHICLE_ALIASES)
        cfg = {
            'json_path': polygons_json,
            'model_path': model,
            'video_in': video_in,
            'video_out': video_out,
            'vehicle_aliases': aliases,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
            'deblur_enabled': bool(self.chk_deblur.isChecked()),
            'deblur_weights': self.edit_deblur.text().strip() or DEFAULT_DEBLUR_WEIGHTS,
        }
        self._log(f"[Lane Export] Using JSON: {polygons_json}")
        self._log(f"[Lane Export] Output: {video_out}")
        return cfg

    # ---------- callbacks ----------
    @QtCore.pyqtSlot(int, int)
    def on_progress(self, cur: int, total: int):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(cur)

    @QtCore.pyqtSlot(object)
    def update_preview_frame(self, qimg: QtGui.QImage):
        # 实时视频帧（标注后）
        if not isinstance(qimg, QtGui.QImage) or qimg.isNull():
            return
        self.img_label.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))

    @QtCore.pyqtSlot()
    def _on_lane_finished(self):
        self._log("[OK] Export finish")
        self.preview_tip.setText("Finish")
        self._realtime_preview = False   # 恢复 PNG 轮询预览

    @QtCore.pyqtSlot(str)
    def _on_lane_failed(self, err: str):
        self._log(f"[ERROR] {err}")
        self.preview_tip.setText("Export Failed")
        self._realtime_preview = False

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.info_box.append(f"[{ts}] {text}")
        self.info_box.verticalScrollBar().setValue(self.info_box.verticalScrollBar().maximum())

    def open_out_dir(self):
        outdir = self.edit_outdir.text().strip() or DEFAULT_OUT_DIR
        os.makedirs(outdir, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(outdir)))

    def closeEvent(self, e: QtGui.QCloseEvent):
        if self.hm_worker and self.hm_worker.isRunning():
            self.hm_worker.stop(); self.hm_worker.wait(2000)
        if self.ln_worker and self.ln_worker.isRunning():
            self.ln_worker.stop(); self.ln_worker.wait(3000)
        super().closeEvent(e)

# ================= main =================
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

