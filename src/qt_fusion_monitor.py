import sys
import json
import socket
import os

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel,
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox
)
from PySide6.QtCore import QSocketNotifier, Qt

SOCKET_PATH = "/tmp/ui_event.sock"


class FusionMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPR Fusion Monitor")

        # ★ 固定窗口宽度 721
        self.setFixedWidth(721)
        self.resize(721, 160)

        self._init_ui()
        self._init_socket()

        # 状态缓存
        self.vision_state = "OK"
        self.fusion_state = "VISION_ONLY"
        self.occl_seq = 0

    # ---------------- UI ----------------
    def _init_ui(self):
        root_layout = QVBoxLayout(self)

        # ================== 顶部：两列布局 ==================
        top_row = QHBoxLayout()

        status_box = QGroupBox("System State")
        data_box = QGroupBox("Current Values")

        top_row.addWidget(status_box)
        top_row.addWidget(data_box)

        # 左右等宽
        top_row.setStretch(0, 1)
        top_row.setStretch(1, 1)

        root_layout.addLayout(top_row)

        # ---------------- System State ----------------
        s = QGridLayout(status_box)

        self.lbl_vision = QLabel("-")
        self.lbl_fusion = QLabel("-")
        self.lbl_occl = QLabel("-")

        s.addWidget(QLabel("Vision State:"), 0, 0)
        s.addWidget(self.lbl_vision, 0, 1)
        s.addWidget(QLabel("Fusion Mode:"), 1, 0)
        s.addWidget(self.lbl_fusion, 1, 1)
        s.addWidget(QLabel("Occlusion Seq:"), 2, 0)
        s.addWidget(self.lbl_occl, 2, 1)

        # ---------------- Current Values ----------------
        d = QGridLayout(data_box)

        self.lbl_cc = QLabel("-")
        self.lbl_depth = QLabel("-")
        self.lbl_press = QLabel("-")
        self.lbl_bpm = QLabel("-")

        d.addWidget(QLabel("CC Index:"), 0, 0)
        d.addWidget(self.lbl_cc, 0, 1)
        d.addWidget(QLabel("Depth (mm):"), 1, 0)
        d.addWidget(self.lbl_depth, 1, 1)
        d.addWidget(QLabel("Pressure:"), 2, 0)
        d.addWidget(self.lbl_press, 2, 1)
        d.addWidget(QLabel("BPM:"), 3, 0)
        d.addWidget(self.lbl_bpm, 3, 1)

        # ---------------- Label Style ----------------
        for lbl in self.findChildren(QLabel):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl.setStyleSheet("font-size: 14px;")

    # ---------------- Socket ----------------
    def _init_socket(self):
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(SOCKET_PATH)
        self.sock.setblocking(False)

        self.notifier = QSocketNotifier(
            self.sock.fileno(),
            QSocketNotifier.Read
        )
        self.notifier.activated.connect(self._on_socket_ready)

    def _on_socket_ready(self):
        try:
            data = self.sock.recv(4096)
        except BlockingIOError:
            return

        msg = json.loads(data.decode("utf-8"))
        self._handle_event(msg)

    # ---------------- Event ----------------
    def _handle_event(self, msg):
        t = msg.get("type", "")

        if t == "peak":
            self.lbl_cc.setText(str(msg.get("idx", "-")))
            self.lbl_depth.setText(f"{msg.get('depth', 0):.2f}")

            bpm = msg.get("bpm")
            if bpm is not None:
                self.lbl_bpm.setText(f"{bpm:.1f}")

        elif t == "pressure_only":
            self.lbl_press.setText(f"{msg.get('press', 0):.0f}")
            self.lbl_depth.setText(f"{msg.get('pred_depth', 0):.2f}")

        elif t == "peak_update":
            if "press" in msg:
                self.lbl_press.setText(f"{msg['press']:.0f}")

        elif t.startswith("occlusion"):
            if t in ("occlusion", "occlusion_lost"):
                self.occl_seq += 1
                self.lbl_occl.setText(str(self.occl_seq))
                self._set_state("HARD", "PRESSURE_ONLY")
            elif t == "occlusion_clear":
                self._set_state("OK", "VISION_ONLY")

        # type == "fit" 被有意忽略（UI 中已无 Fit Model）

    # ---------------- State ----------------
    def _set_state(self, vision, fusion):
        self.vision_state = vision
        self.fusion_state = fusion

        self.lbl_vision.setText(vision)
        self.lbl_fusion.setText(fusion)

        color = {
            "OK": "green",
            "SOFT": "orange",
            "HARD": "red"
        }.get(vision, "black")

        self.lbl_vision.setStyleSheet(
            f"color:{color}; font-weight:bold; font-size:14px;"
        )
        self.lbl_fusion.setStyleSheet(
            f"color:{color}; font-weight:bold; font-size:14px;"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FusionMonitor()
    w.show()
    sys.exit(app.exec())
