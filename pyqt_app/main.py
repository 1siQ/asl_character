import sys
import os
import difflib
from collections import deque, Counter
from enum import Enum, auto

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy, QScrollArea, QGridLayout,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette


# ── NLP ────────────────────────────────────────────────────────────────────────

class OfflineAutoCompleter:
    def __init__(self):
        self.vocabulary = [
            "MERHABA", "PROJE", "SİSTEM", "KONTROL", "ROBOT", "YAPAYZEKA",
            "OTOMASYON", "MÜHENDİS", "İTÜ", "BİLGİSAYAR", "GÖRÜNTÜ", "ASL",
        ]

    def suggest(self, partial, top_n=3):
        if not partial:
            return []
        p = partial.upper()
        prefix = [w for w in self.vocabulary if w.startswith(p)]
        fuzzy  = difflib.get_close_matches(p, self.vocabulary, n=top_n, cutoff=0.4)
        return list(dict.fromkeys(prefix + fuzzy))[:top_n]


# ── ASL CONTROLLER (FSM) ──────────────────────────────────────────────────────

class _S(Enum):
    IDLE    = auto()
    LOCKING = auto()
    WRITTEN = auto()


class ASLController:
    """
    Finite-state machine that converts per-frame char detections into words/sentences.

    States
    ------
    IDLE    : no active detection, waiting for a stable signal
    LOCKING : accumulating frames toward a confident character commit
    WRITTEN : a character was just written; same character is latched
               (user must clear hand or show a different char to re-write)

    Gestures
    --------
    YUMRUK  : fist — held for YUMRUK_RESET frames → commits current word + space
    Swipe LEFT  → backspace (remove last character of word, or last char of sentence)
    Swipe RIGHT → accept top NLP suggestion as current word
    """

    BUFFER_SIZE          = 20   # sliding window width (frames)
    MODE_THRESHOLD       = 14   # frames with same char needed to commit (~70 %)
    NULL_BUF_CLEAR       = 8    # null streak to clear buffer when IDLE/LOCKING
    NULL_LATCH_RELEASE   = 20   # null streak to release latch when WRITTEN
    YUMRUK_RESET         = 8    # consecutive YUMRUK frames to trigger word break

    def __init__(self):
        self._state       = _S.IDLE
        self._buf         = deque(maxlen=self.BUFFER_SIZE)
        self._null_streak = 0
        self._locked_char = None   # latch: the char just written
        self._suggestions = []
        self.word         = ""
        self.sentence     = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def push_char(self, char: str):
        """Feed one frame's detected character (empty string = no hand detected)."""
        if not char:
            return self._on_null()
        if char == "YUMRUK":
            return self._on_yumruk()
        return self._on_letter(char)

    def push_swipe(self, direction: str):
        """Handle a swipe gesture: 'LEFT' = backspace, 'RIGHT' = accept suggestion."""
        if direction == "LEFT":
            self._do_backspace()
        elif direction == "RIGHT":
            self._do_accept()
        # Always reset tracking state after a swipe
        self._buf.clear()
        self._null_streak = 0
        self._locked_char = None
        self._state = _S.IDLE
        return self.word, self.sentence

    def set_suggestions(self, suggestions: list):
        self._suggestions = list(suggestions)

    def reset(self):
        self._state       = _S.IDLE
        self._buf.clear()
        self._null_streak = 0
        self._locked_char = None
        self._suggestions = []
        self.word         = ""
        self.sentence     = ""

    # ── Internal state machine ────────────────────────────────────────────────

    def _on_null(self):
        self._null_streak += 1
        if self._state == _S.WRITTEN:
            # Latch stays locked through brief confidence drops (e.g. hand shifts slightly).
            # Only release after a long null streak — the user deliberately removed their hand.
            if self._null_streak >= self.NULL_LATCH_RELEASE:
                self._locked_char = None
                self._state       = _S.IDLE
                self._null_streak = 0
                self._buf.clear()
        else:
            if self._null_streak >= self.NULL_BUF_CLEAR:
                self._buf.clear()
                self._state       = _S.IDLE
                self._null_streak = 0
        return self.word, self.sentence

    def _on_yumruk(self):
        self._null_streak = 0
        self._buf.append("YUMRUK")
        if self._buf.count("YUMRUK") >= self.YUMRUK_RESET:
            if self.word:
                self.sentence += self.word + " "
                self.word = ""
            self._buf.clear()
            self._locked_char = None
            self._state = _S.IDLE
        return self.word, self.sentence

    def _on_letter(self, char: str):
        self._null_streak = 0

        # Latch: same char just written → ignore until hand resets
        if self._state == _S.WRITTEN and char == self._locked_char:
            return self.word, self.sentence

        # Different char seen while latched → start fresh LOCKING cycle
        if self._state == _S.WRITTEN:
            self._state = _S.LOCKING
            self._buf.clear()
            self._locked_char = None

        self._buf.append(char)
        if self._state == _S.IDLE:
            self._state = _S.LOCKING

        # Commit only once we have enough frames with the same dominant char
        if len(self._buf) >= self.MODE_THRESHOLD:
            best, count = Counter(self._buf).most_common(1)[0]
            if count >= self.MODE_THRESHOLD and best != "YUMRUK":
                self.word        += best
                self._locked_char = best
                self._state       = _S.WRITTEN
                self._buf.clear()

        return self.word, self.sentence

    def _do_backspace(self):
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            s = self.sentence.rstrip(" ")
            self.sentence = (s[:-1] + " ") if len(s) > 1 else ""

    def _do_accept(self):
        if self._suggestions:
            self.sentence += self._suggestions[0] + " "
            self.word = ""


# ── SWIPE DETECTION CONSTANTS ─────────────────────────────────────────────────
# Optical flow is computed only over the top-20% most-moving pixels so that
# the static background doesn't dilute the hand-movement signal.

_SWIPE_PEAK      = 4.0    # pixels/frame (active pixels only) — peak threshold
_SWIPE_AVG       = 2.0    # pixels/frame — mean threshold over window
_SWIPE_WIN       = 6      # consecutive frames in the analysis window
_SWIPE_COOLDOWN  = 25     # frames to ignore after a swipe fires (~1.7 s at 15 fps)
_SWIPE_MIN_PX    = 40     # minimum active pixels before we trust the flow value


# ── KAMERA İŞ PARÇACIĞI ───────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready    = pyqtSignal(np.ndarray)
    char_detected  = pyqtSignal(str)
    swipe_detected = pyqtSignal(str)        # "LEFT" or "RIGHT"

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self._running   = False

    def run(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        model  = YOLO(self.model_path)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._running    = True
        n                = 0
        annotated        = None
        prev_gray        = None
        flow_buf         = deque(maxlen=_SWIPE_WIN)
        swipe_cooldown   = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1

            # ── Dense optical flow for swipe (every frame) ───────────────
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 120))

            if prev_gray is not None:
                if swipe_cooldown > 0:
                    swipe_cooldown -= 1
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, small, None,
                        0.5, 3, 15, 3, 5, 1.2, 0,
                    )
                    u   = flow[..., 0]
                    v   = flow[..., 1]
                    mag = np.sqrt(u * u + v * v)

                    # Only average over pixels that are actually moving (top 20% by magnitude).
                    # This removes the static background from the calculation.
                    motion_thresh = float(np.percentile(mag, 80))
                    active = mag > max(motion_thresh, 0.5)
                    if active.sum() >= _SWIPE_MIN_PX:
                        mean_u = float(np.mean(u[active]))
                    else:
                        mean_u = 0.0

                    flow_buf.append(mean_u)

                    if len(flow_buf) == _SWIPE_WIN:
                        avg  = sum(flow_buf) / _SWIPE_WIN
                        peak = max(flow_buf, key=abs)
                        if abs(peak) > _SWIPE_PEAK and abs(avg) > _SWIPE_AVG:
                            direction = "LEFT" if avg < 0 else "RIGHT"
                            self.swipe_detected.emit(direction)
                            flow_buf.clear()
                            swipe_cooldown = _SWIPE_COOLDOWN

            prev_gray = small

            # ── YOLO classification (every 2nd frame) ────────────────────
            if n % 2 == 0:
                results  = model.predict(frame, imgsz=320, verbose=False, device=device)
                detected = ""
                if results[0].probs is not None:
                    top1_idx  = results[0].probs.top1
                    top1_conf = results[0].probs.top1conf.item()
                    if top1_conf >= 0.80:
                        detected = model.names[top1_idx]
                self.char_detected.emit(detected)
                annotated = results[0].plot()

            self.frame_ready.emit(annotated if annotated is not None else frame)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


# ── YARDIMCI WİDGET'LER ───────────────────────────────────────────────────────

def card(content_widget, title=None):
    outer = QFrame()
    outer.setStyleSheet("""
        QFrame {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
        }
    """)
    lay = QVBoxLayout(outer)
    lay.setContentsMargins(20, 18, 20, 18)
    lay.setSpacing(10)
    if title:
        lbl = QLabel(title)
        lbl.setFont(QFont("Inter", 11, QFont.Bold))
        lbl.setStyleSheet("color:#334155; border:none;")
        lay.addWidget(lbl)
    lay.addWidget(content_widget)
    return outer


class InfoBox(QFrame):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:12px;
            }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self._lbl = QLabel(label.upper())
        self._lbl.setFont(QFont("Inter", 8, QFont.Bold))
        self._lbl.setStyleSheet("color:#64748b; letter-spacing:1px; border:none;")

        self._val = QLabel("...")
        self._val.setFont(QFont("Inter", 18, QFont.Bold))
        self._val.setStyleSheet("color:#0f172a; border:none;")
        self._val.setWordWrap(True)

        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set_value(self, text, color="#0f172a"):
        self._val.setText(text or "...")
        self._val.setStyleSheet(f"color:{color}; border:none;")


class ChipRow(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("border:none; background:transparent;")
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._lay.addStretch()

    def set_chips(self, words):
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for w in words:
            c = QLabel(w)
            c.setFont(QFont("Inter", 10, QFont.Medium))
            c.setStyleSheet("""
                background:#eef2ff; color:#4f46e5;
                padding:6px 12px; border-radius:999px; border:none;
            """)
            self._lay.insertWidget(self._lay.count() - 1, c)

        if not words:
            ph = QLabel("Harf girişi bekleniyor...")
            ph.setStyleSheet("color:#94a3b8; font-style:italic; border:none;")
            ph.setFont(QFont("Inter", 10))
            self._lay.insertWidget(0, ph)


# ── ANA PENCERE ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Intelligence  ·  TeknoLAB 2025-2026")
        self.setMinimumSize(1100, 680)
        self._apply_palette()

        self.controller = ASLController()
        self.completer  = OfflineAutoCompleter()
        self.cam_thread = None

        self.model_path = os.path.join(os.path.dirname(__file__), "..", "best_cls.pt")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        self.dataset_train = os.path.join(
            os.path.dirname(__file__), "..", "train", "dataset_clean", "train"
        )

        root.addWidget(self._build_header())
        root.addWidget(self._build_tabs(), stretch=1)

        # Timer to auto-clear the status feedback label
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status_label.setText(""))

    # ── PALETTE ──────────────────────────────────────────────────────────────

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#f8fafc"))
        pal.setColor(QPalette.WindowText, QColor("#0f172a"))
        self.setPalette(pal)
        self.setStyleSheet("QMainWindow { background:#f8fafc; }")

    # ── HEADER ───────────────────────────────────────────────────────────────

    def _build_header(self):
        w = QFrame()
        w.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0f172a, stop:1 #1e293b);
                border-radius: 20px;
            }
        """)
        w.setFixedHeight(110)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32, 18, 32, 18)
        lay.setSpacing(4)

        title = QLabel("ASL Translation Platform")
        title.setFont(QFont("Inter", 22, QFont.Bold))
        title.setStyleSheet("color:white; background:transparent;")

        sub = QLabel("TeknoLAB 2025-2026 Engine Core  ·  Control & Automation Engineering")
        sub.setFont(QFont("Inter", 10))
        sub.setStyleSheet("color:#94a3b8; background:transparent;")

        lay.addWidget(title)
        lay.addWidget(sub)
        return w

    # ── TABS ──────────────────────────────────────────────────────────────────

    def _build_tabs(self):
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #64748b;
                padding: 10px 24px;
                margin-right: 4px;
                border-radius: 10px 10px 0 0;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #4f46e5;
                color: white;
            }
        """)
        tabs.addTab(self._build_body(),  "⚡  Canlı Çeviri")
        tabs.addTab(self._build_guide(), "📋  ASL Kılavuzu")
        return tabs

    # ── BODY ─────────────────────────────────────────────────────────────────

    def _build_body(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(20)
        lay.addWidget(self._build_camera_panel(), stretch=3)
        lay.addWidget(self._build_nlp_panel(),    stretch=2)
        return w

    # ── KILAVUZ SEKMESİ ───────────────────────────────────────────────────────

    def _build_guide(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:#f8fafc; }")

        container = QWidget()
        container.setStyleSheet("background:#f8fafc;")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QLabel("Eğitilen Model Sınıf Yapısı")
        title.setFont(QFont("Inter", 14, QFont.Bold))
        title.setStyleSheet("color:#0f172a;")

        desc = QLabel(
            "Aşağıdaki görseller modelin eğitildiği gerçek dataset fotoğraflarından alınmıştır. "
            "Her harf için 3 örnek gösterilmektedir.  ✊ YUMRUK = boşluk"
        )
        desc.setWordWrap(True)
        desc.setFont(QFont("Inter", 10))
        desc.setStyleSheet("color:#64748b;")

        outer.addWidget(title)
        outer.addWidget(desc)

        grid = QGridLayout()
        grid.setSpacing(12)
        outer.addLayout(grid)
        outer.addStretch()

        classes = [chr(i) for i in range(65, 91)] + ["YUMRUK"]
        COLS = 4

        for idx, cls in enumerate(classes):
            cell = self._build_guide_cell(cls)
            grid.addWidget(cell, idx // COLS, idx % COLS)

        scroll.setWidget(container)
        return scroll

    def _build_guide_cell(self, cls_name):
        cell = QFrame()
        is_special = cls_name in ("YUMRUK", "ONAYLA")
        bg = "#fff1f2" if is_special else "#ffffff"
        cell.setStyleSheet(f"""
            QFrame {{
                background:{bg};
                border:1px solid #e2e8f0;
                border-radius:14px;
            }}
        """)
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        color = "#e11d48" if is_special else "#0f172a"
        lbl = QLabel(f"{'✊ ' if cls_name == 'YUMRUK' else ''}{cls_name}"
                     f"{'  (Boşluk)' if cls_name == 'YUMRUK' else ''}")
        lbl.setFont(QFont("Inter", 12, QFont.Bold))
        lbl.setStyleSheet(f"color:{color}; border:none;")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)

        img_row = QHBoxLayout()
        img_row.setSpacing(6)

        cls_dir = os.path.join(self.dataset_train, cls_name)
        images  = []
        if os.path.isdir(cls_dir):
            all_imgs = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            images = all_imgs[:3]

        if images:
            for fname in images:
                fpath = os.path.join(cls_dir, fname)
                pix = QPixmap(fpath).scaled(
                    100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                img_lbl = QLabel()
                img_lbl.setPixmap(pix)
                img_lbl.setAlignment(Qt.AlignCenter)
                img_lbl.setStyleSheet(
                    "border:1px solid #e2e8f0; border-radius:8px; background:#f8fafc;"
                )
                img_row.addWidget(img_lbl)
        else:
            ph = QLabel("Görsel\nbulunamadı")
            ph.setAlignment(Qt.AlignCenter)
            ph.setStyleSheet("color:#94a3b8; border:none; font-size:11px;")
            img_row.addWidget(ph)

        lay.addLayout(img_row)
        return cell

    # ── KAMERA PANELİ ─────────────────────────────────────────────────────────

    def _build_camera_panel(self):
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            background:#0f172a;
            border-radius:16px;
            color:#475569;
        """)
        self.video_label.setText("Kamera Pasif")
        self.video_label.setFont(QFont("Inter", 13))

        self.toggle_btn = QPushButton("▶  Kamerayı Başlat")
        self.toggle_btn.setFixedHeight(46)
        self.toggle_btn.setFont(QFont("Inter", 11, QFont.Bold))
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background:#4f46e5; color:white;
                border-radius:12px; border:none;
            }
            QPushButton:hover  { background:#4338ca; }
            QPushButton:pressed{ background:#3730a3; }
        """)
        self.toggle_btn.clicked.connect(self._toggle_camera)

        lay.addWidget(card(self.video_label, "Kamera Girişi"), stretch=1)
        lay.addWidget(self.toggle_btn)
        return container

    # ── NLP PANELİ ───────────────────────────────────────────────────────────

    def _build_nlp_panel(self):
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.sentence_box = InfoBox("Oluşan Cümle")
        self.word_box     = InfoBox("Mevcut Kelime")
        self.chip_row     = ChipRow()

        chip_label = QLabel("AKILLI TAHMİNLER  (→ kaydır: onayla)")
        chip_label.setFont(QFont("Inter", 8, QFont.Bold))
        chip_label.setStyleSheet("color:#64748b; letter-spacing:1px;")

        # Visual feedback for swipe gestures
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Inter", 10, QFont.Bold))
        self.status_label.setFixedHeight(28)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color:transparent; border:none;")

        hint = QLabel("← Sil  |  ✊ Boşluk  |  → Öneriyi Onayla")
        hint.setFont(QFont("Inter", 9))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#94a3b8; border:none;")

        clear_btn = QPushButton("🧹  Arabelleği Temizle")
        clear_btn.setFixedHeight(44)
        clear_btn.setFont(QFont("Inter", 10, QFont.Bold))
        clear_btn.setStyleSheet("""
            QPushButton {
                background:#f1f5f9; color:#475569;
                border-radius:12px; border:1px solid #e2e8f0;
            }
            QPushButton:hover { background:#e2e8f0; }
        """)
        clear_btn.clicked.connect(self._clear_buffer)

        nlp_inner = QWidget()
        ni_lay = QVBoxLayout(nlp_inner)
        ni_lay.setContentsMargins(0, 0, 0, 0)
        ni_lay.setSpacing(10)
        ni_lay.addWidget(self.sentence_box)
        ni_lay.addWidget(self.word_box)
        ni_lay.addWidget(chip_label)
        ni_lay.addWidget(self.chip_row)
        ni_lay.addWidget(self.status_label)
        ni_lay.addStretch()
        ni_lay.addWidget(hint)
        ni_lay.addWidget(clear_btn)

        lay.addWidget(card(nlp_inner, "Doğal Dil İşleme (NLP)"), stretch=1)
        return container

    # ── KAMERA KONTROL ────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if not os.path.exists(self.model_path):
            self.video_label.setText("HATA: best.pt bulunamadı!")
            return

        self.cam_thread = CameraThread(self.model_path)
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.char_detected.connect(self._on_char)
        self.cam_thread.swipe_detected.connect(self._on_swipe)
        self.cam_thread.start()

        self.toggle_btn.setText("⏹  Kamerayı Durdur")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background:#e11d48; color:white;
                border-radius:12px; border:none;
            }
            QPushButton:hover  { background:#be123c; }
            QPushButton:pressed{ background:#9f1239; }
        """)

    def _stop_camera(self):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None

        self.video_label.setText("Kamera Pasif")
        self.toggle_btn.setText("▶  Kamerayı Başlat")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background:#4f46e5; color:white;
                border-radius:12px; border:none;
            }
            QPushButton:hover  { background:#4338ca; }
            QPushButton:pressed{ background:#3730a3; }
        """)

    # ── SLOT'LAR ─────────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _on_char(self, char: str):
        word, sentence = self.controller.push_char(char if char else "")
        suggestions    = self.completer.suggest(word)
        self.controller.set_suggestions(suggestions)

        self.sentence_box.set_value(sentence)
        self.word_box.set_value(word, color="#4f46e5")
        self.chip_row.set_chips(suggestions)

    def _on_swipe(self, direction: str):
        # Capture the top suggestion BEFORE push_swipe consumes it
        accepted_word = (self.controller._suggestions or [""])[0]

        word, sentence = self.controller.push_swipe(direction)
        suggestions    = self.completer.suggest(word)
        self.controller.set_suggestions(suggestions)

        self.sentence_box.set_value(sentence)
        self.word_box.set_value(word, color="#4f46e5")
        self.chip_row.set_chips(suggestions)

        if direction == "LEFT":
            self.status_label.setText("← Silindi")
            self.status_label.setStyleSheet(
                "color:#e11d48; font-weight:bold; border:none;"
            )
        elif direction == "RIGHT":
            msg = f"→ Onaylandı: {accepted_word}" if accepted_word else "→ Öneri yok"
            self.status_label.setText(msg)
            self.status_label.setStyleSheet(
                "color:#16a34a; font-weight:bold; border:none;"
            )

        self._status_timer.start(1800)

    def _clear_buffer(self):
        self.controller.reset()
        self.sentence_box.set_value("")
        self.word_box.set_value("")
        self.chip_row.set_chips([])
        self.status_label.setText("")

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)


# ── GİRİŞ ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
