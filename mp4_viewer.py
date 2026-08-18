# -*- coding: utf-8 -*-

import sys, os, argparse, time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    """BGR (H,W,3) -> QImage(RGB)"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class ImageLabel(QtWidgets.QLabel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#202020; color:#999;")
        self._pix = None

    def setPixmapKeepAspect(self, pix: QtGui.QPixmap):
        self._pix = pix
        self._updateScaled()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._updateScaled()

    def _updateScaled(self):
        if not self._pix or self._pix.isNull():
            return
        scaled = self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        super().setPixmap(scaled)


class VideoPlayer(QtWidgets.QMainWindow):
    def __init__(self, init_path: str = ""):
        super().__init__()
        self.setWindowTitle("MP4播放器")
        self.resize(1000, 700)

        # Video State
        self.cap = None
        self.video_path = ""
        self.total_frames = 0
        self.fps = 0.0
        self.duration = 0.0
        self.frame_idx = 0
        self.playing = False
        self.speed = 1.0        # 播放倍速
        self.last_tick = 0.0

        # UI
        self._build_ui()

        # Timer（按 fps 与倍速刷新）
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # 若命令行传入了路径，尝试打开
        if init_path:
            self.open_video(init_path)

    # ---------------- UI ----------------
    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        # 顶部工具栏
        tb = QtWidgets.QHBoxLayout()
        root.addLayout(tb)

        self.btn_open = QtWidgets.QPushButton("打开视频…")
        self.btn_open.clicked.connect(self._on_open)
        tb.addWidget(self.btn_open)

        self.btn_play = QtWidgets.QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        tb.addWidget(self.btn_play)

        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop)
        tb.addWidget(self.btn_stop)

        tb.addSpacing(10)
        tb.addWidget(QtWidgets.QLabel("倍速:"))
        self.combo_speed = QtWidgets.QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.currentTextChanged.connect(self._on_speed_change)
        tb.addWidget(self.combo_speed)

        tb.addSpacing(10)
        self.btn_snap = QtWidgets.QPushButton("截图")
        self.btn_snap.clicked.connect(self.snapshot)
        tb.addWidget(self.btn_snap)

        tb.addStretch(1)

        # 画面显示
        self.view = ImageLabel()
        root.addWidget(self.view, 1)

        # 进度条与时间
        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom)

        self.lbl_time = QtWidgets.QLabel("00:00:00 / 00:00:00")
        self.lbl_time.setStyleSheet("color:#ccc;")
        bottom.addWidget(self.lbl_time)

        bottom.addSpacing(12)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self._slider_dragging = False
        bottom.addWidget(self.slider, 1)

        self.lbl_info = QtWidgets.QLabel("帧: - / -   FPS: -")
        self.lbl_info.setStyleSheet("color:#ccc;")
        bottom.addWidget(self.lbl_info)

        # 快捷键
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, self.snapshot)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, self.close)

        # 状态栏
        self.statusBar().showMessage("就绪。空格：播放/暂停；S：截图；左右方向键：微调进度")

    # ---------------- File Ops ----------------
    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if path:
            self.open_video(path)

    def open_video(self, path: str):
        self.close_video()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "打开失败", f"无法打开视频:\n{path}")
            return

        self.cap = cap
        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.frame_idx = 0

        self.slider.setRange(0, max(0, self.total_frames - 1))
        self._update_time_label()
        self._update_info_label()

        # 读取第一帧
        ok, frame = self.cap.read()
        if ok:
            self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._show_frame(frame)
            self._update_time_label()
            self.slider.setValue(self.frame_idx)

        self.statusBar().showMessage(f"已打开: {os.path.basename(path)}")
        # 自动播放
        self.play()

    def close_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.playing = False
        self.timer.stop()

    # ---------------- Playback ----------------
    def play(self):
        if not self.cap:
            return
        self.playing = True
        self.btn_play.setText("暂停")
        self._reset_timer_interval()
        self.last_tick = time.time()
        self.timer.start()

    def pause(self):
        self.playing = False
        self.btn_play.setText("播放")
        self.timer.stop()

    def stop(self):
        self.pause()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            ok, frame = self.cap.read()
            if ok:
                self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self._show_frame(frame)
                self.slider.setValue(self.frame_idx)
                self._update_time_label()

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def _reset_timer_interval(self):
        # 根据 fps 与倍速设置刷新间隔
        fps = self.fps if self.fps > 0 else 30.0
        interval_ms = max(1, int(1000.0 / (fps * self.speed)))
        self.timer.setInterval(interval_ms)

    def _on_speed_change(self, txt: str):
        try:
            self.speed = float(txt.replace("x", ""))
        except Exception:
            self.speed = 1.0
        self._reset_timer_interval()
        self._update_info_label()

    def _on_tick(self):
        """按时钟拉帧；若到末尾则停止。"""
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            self.statusBar().showMessage("播放结束。")
            return
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self._show_frame(frame)
        if not self._slider_dragging:
            self.slider.setValue(self.frame_idx)
        self._update_time_label()

    # ---------------- Slider / Seek ----------------
    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self._seek_to(self.slider.value())

    def _on_slider_moved(self, v: int):
        # 拖动时即时预览（轻量）
        if self.cap and self._slider_dragging:
            self._seek_to(v, preview=True)

    def _seek_to(self, frame_id: int, preview: bool = False):
        if not self.cap:
            return
        frame_id = int(np.clip(frame_id, 0, max(0, self.total_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = self.cap.read()
        if ok:
            self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._show_frame(frame)
            self._update_time_label()
            if not preview:
                self.slider.setValue(self.frame_idx)

    # ---------------- Render & Info ----------------
    def _show_frame(self, frame_bgr: np.ndarray):
        
        txt = f"{os.path.basename(self.video_path) if self.video_path else 'Untitled'}  |  " \
              f"Frame {self.frame_idx+1}/{self.total_frames}  |  {self.fps:.1f} FPS"
        cv2.putText(frame_bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        qimg = bgr_to_qimage(frame_bgr)
        self.view.setPixmapKeepAspect(QtGui.QPixmap.fromImage(qimg))

    def _update_time_label(self):
        cur_s = self.frame_idx / (self.fps if self.fps > 0 else 30.0)
        total_s = self.duration
        def fmt(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        self.lbl_time.setText(f"{fmt(cur_s)} / {fmt(total_s)}")

    def _update_info_label(self):
        self.lbl_info.setText(f"帧: {self.frame_idx+1} / {self.total_frames}   FPS: {self.fps:.1f}   倍速: {self.speed:.2f}x")

    # ---------------- Snapshot ----------------
    def snapshot(self):
        if not self.cap or not self.video_path:
            return
        
        cur = max(0, self.frame_idx - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = self.cap.read()
        if not ok:
            return
        # 保存路径
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        out_dir = os.path.join(os.path.dirname(self.video_path), "screenshots")
        os.makedirs(out_dir, exist_ok=True)
        fn = f"{base}_frame{self.frame_idx:06d}.png"
        out_path = os.path.join(out_dir, fn)
        cv2.imwrite(out_path, frame)
        self.statusBar().showMessage(f"已截图: {out_path}")
        # 恢复到原位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

    # ---------------- Events ----------------
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key_Space:
            self.toggle_play()
        elif e.key() == QtCore.Qt.Key_Left:
            self._seek_to(self.frame_idx - max(1, int(self.fps // 2)))
        elif e.key() == QtCore.Qt.Key_Right:
            self._seek_to(self.frame_idx + max(1, int(self.fps // 2)))
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.timer.stop()
        self.close_video()
        super().closeEvent(e)


def main():
    parser = argparse.ArgumentParser(description="MP4播放器 ")
    parser.add_argument("--video", "-v", type=str, default="", help="视频文件路径 ")
    args = parser.parse_args()

    # 某些环境下 OpenCV 的 Qt 平台插件路径（通常无需设置）
    try:
        import cv2 as _cv2
        os.environ.setdefault(
            'QT_QPA_PLATFORM_PLUGIN_PATH',
            os.path.join(os.path.dirname(_cv2.__file__), 'qt', 'plugins', 'platforms')
        )
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    w = VideoPlayer(args.video)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

