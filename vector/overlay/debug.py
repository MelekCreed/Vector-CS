"""Developer view: camera feed with landmarks + live engine internals.

Shows everything needed to tune recognition: landmark IDs, per-pose scores,
current state, confidence breakdowns, cursor coordinates, velocities, FPS,
latency per stage, selected window and the recent gesture history (including
*rejected* gestures and why). Also hosts the dataset recorder.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget)

from vector.vision.landmarks import CONNECTIONS

STYLE = """
QWidget { background: #0A0E13; color: #D8E6EC; font-family: 'Segoe UI'; font-size: 12px; }
QLabel#panel { font-family: 'Cascadia Mono', 'Consolas'; font-size: 11px; color: #BFD3DB; }
QLineEdit { background: #121922; border: 1px solid #1E2A36; border-radius: 6px; padding: 5px 8px; }
QPushButton { background: #12303A; border: 1px solid #1F5563; border-radius: 6px; padding: 6px 12px;
              color: #6EE7F9; font-weight: 600; }
QPushButton:hover { background: #164150; }
"""


def draw_landmarks(img: np.ndarray, hands, show_ids: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for label, primary, lms in hands:
        color = (249, 231, 110) if primary else (220, 220, 220)   # BGR cyan-ish / white
        pts = [(int(x * w), int(y * h)) for x, y, _ in lms]
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(out, pt, 4 if i in (4, 8) else 3, (255, 255, 255), -1, cv2.LINE_AA)
            if show_ids:
                cv2.putText(out, str(i), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                            (180, 200, 210), 1, cv2.LINE_AA)
        wx, wy = pts[0]
        cv2.putText(out, label, (wx - 20, wy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


class DebugWindow(QWidget):
    def __init__(self, shared, metrics, actuator, worker, cfg):
        super().__init__()
        self.shared, self.metrics, self.actuator, self.worker, self.cfg = shared, metrics, actuator, worker, cfg
        self.setWindowTitle("Vector · Debug")
        self.setStyleSheet(STYLE)
        self.resize(1180, 640)
        self.video = QLabel()
        self.video.setMinimumSize(640, 480)
        self.video.setAlignment(Qt.AlignCenter)
        self.left = QLabel(objectName="panel")
        self.left.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.left.setTextFormat(Qt.PlainText)
        self.history = QLabel(objectName="panel")
        self.history.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.history.setTextFormat(Qt.PlainText)
        self.label_edit = QLineEdit(placeholderText="gesture label, e.g. swipe_left")
        rec = QPushButton("Record clip (3s countdown, 2s)")
        rec.clicked.connect(self._record)
        calib = QPushButton("Recalibrate")
        calib.clicked.connect(self.worker.start_calibration)

        right = QVBoxLayout()
        right.addWidget(self.left, 3)
        right.addWidget(self.history, 2)
        row = QHBoxLayout()
        row.addWidget(self.label_edit, 1)
        row.addWidget(rec)
        row.addWidget(calib)
        right.addLayout(row)
        root = QHBoxLayout(self)
        root.addWidget(self.video, 3)
        root.addLayout(right, 2)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)

    def showEvent(self, e):  # noqa: N802
        self.shared.want_frame = True
        self._timer.start(33)
        super().showEvent(e)

    def hideEvent(self, e):  # noqa: N802
        self.shared.want_frame = False
        self._timer.stop()
        super().hideEvent(e)

    def _record(self) -> None:
        label = self.label_edit.text().strip() or "unlabeled"
        self.worker.record(label)

    def _refresh(self) -> None:
        ui = self.shared.read()
        snap = ui.snapshot
        if ui.frame is not None:
            prim = {hv.label for hv in (snap.hands if snap else []) if hv.primary}
            hands = [(lbl, lbl in prim, lms) for lbl, lms in ui.landmarks]
            img = draw_landmarks(ui.frame, hands)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.left.setText(self._panel(ui))
        fb = ui.feedback
        if fb is not None:
            lines = ["GESTURE HISTORY (newest first)"]
            for h in reversed(fb.history[-14:]):
                mark = "✓" if h.accepted else "✗"
                lines.append(f"{mark} {h.t % 1000:7.2f}  {h.gesture:<22} {h.action:<24} "
                             f"c={h.confidence:.2f} {h.reason}")
            self.history.setText("\n".join(lines))

    def _panel(self, ui) -> str:
        snap = ui.snapshot
        s = self.metrics.summary()
        a = self.actuator.stats

        def ms(k):
            return f"{s[k][0]:6.1f} / {s[k][1]:6.1f}" if k in s else "     –"
        lines = [
            f"STATE      {snap.system.value if snap else '-':<9} mode={snap.primary_mode.value if snap else '-'}"
            f"  owner={snap.owner if snap else '-'}  draw={snap.draw_mode if snap else '-'}",
            f"CAMERA     {ui.camera_fps:5.1f} fps   vision {self.metrics.rate('vision'):5.1f} fps"
            f"   actuator {a.tick_hz:5.0f} Hz",
            "LATENCY ms     mean /    p95",
            f"  capture age   {ms('capture_age')}",
            f"  camera read   {ms('capture_read')}",
            f"  inference     {ms('inference')}",
            f"  gesture       {ms('gesture')}",
            f"  intent        {ms('intent')}",
            f"  vision e2e    {ms('end_to_end_vision')}",
            f"  overlay paint {ms('overlay_paint')}",
            f"  OS command    {a.last_cmd_latency_ms:6.1f} (frame→OS)   exec {a.last_cmd_exec_ms:5.2f}",
            f"  dropped cmds  gen={a.dropped_generation} expired={a.dropped_expired}",
        ]
        if snap is not None:
            if snap.cursor is not None:
                lines.append(f"CURSOR     ({snap.cursor[0]:7.1f}, {snap.cursor[1]:7.1f})"
                             f"  visible={snap.cursor_visible}")
            fb = ui.feedback
            if fb is not None:
                lines.append(f"TARGET     {fb.lock_state or 'hover'}: "
                             f"{(fb.lock_title or fb.hover_title or '-')[:48]}")
            feats = {f.track_id: f for f in ui.features}
            for hv in snap.hands:
                f = feats.get(hv.track_id)
                if f is None:
                    continue
                lines.append("")
                lines.append(f"HAND #{hv.track_id} {hv.label:<5} {'PRIMARY' if hv.primary else '':<7}"
                             f" {hv.mode.value:<13} pose={hv.pose}({hv.pose_score:.2f})"
                             f"{' STALE' if hv.stale else ''}")
                lines.append(f"  pinch={f.pinch_ratio:.2f}{' ON' if hv.pinch else '   '}"
                             f"  mid={f.middle_pinch_ratio:.2f}  facing={f.palm_facing:+.2f}"
                             f"  upright={f.upright:+.2f}  roll={np.degrees(f.roll):+5.0f}°")
                lines.append(f"  vel=({hv.velocity[0]:+5.2f},{hv.velocity[1]:+5.2f}) hl/s"
                             f"  speed={hv.speed:4.2f}  scale={f.hand_scale:.3f}")
                lines.append("  ext " + "  ".join(f"{k[:2]}={v:.2f}" for k, v in f.extension.items()))
            if snap.scroll_velocity:
                lines.append(f"SCROLL     {snap.scroll_velocity:+7.0f} wheel/s")
        return "\n".join(lines)
