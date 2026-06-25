"""
ASL Character Recognition — Streamlit Web Arayüzü
TeknoGenç TeknoLAB 2025-2026

Çalışma modları:
  1. Canlı Çeviri  — WebRTC webcam akışı (yerel / HF Spaces GPU tier)
  2. Görüntü Testi — tek fotoğraf yükleyip test et (HF Spaces demo)
  3. Performans    — eğitim grafikleri ve metrikler
"""

import os
import threading
import difflib
from collections import deque, Counter
from enum import Enum, auto

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# ── Sayfa ayarları ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ASL Character Recognition",
    page_icon="hand",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_cls.pt")

# ── ASL Controller (FSM) ──────────────────────────────────────────────────────

class _S(Enum):
    IDLE    = auto()
    LOCKING = auto()
    WRITTEN = auto()


class ASLController:
    """
    Kayan pencere + oy sistemi ile harf tespitini kararlı kelime/cümleye dönüştürür.
    Son 20 karenin %70'i aynı sınıfı gösterirse harf yazılır.
    Aynı harf yanlışlıkla tekrar yazılmaması için latch mekanizması kullanılır.
    """
    BUFFER_SIZE        = 20
    MODE_THRESHOLD     = 14
    NULL_BUF_CLEAR     = 8
    NULL_LATCH_RELEASE = 20
    YUMRUK_RESET       = 8

    def __init__(self):
        self._state       = _S.IDLE
        self._buf         = deque(maxlen=self.BUFFER_SIZE)
        self._null_streak = 0
        self._locked_char = None
        self.word         = ""
        self.sentence     = ""

    def push_char(self, char: str):
        if not char:
            return self._on_null()
        if char == "YUMRUK":
            return self._on_yumruk()
        return self._on_letter(char)

    def backspace(self):
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            s = self.sentence.rstrip(" ")
            self.sentence = (s[:-1] + " ") if len(s) > 1 else ""
        self._buf.clear()
        self._null_streak = 0
        self._locked_char = None
        self._state = _S.IDLE
        return self.word, self.sentence

    def reset(self):
        self._state       = _S.IDLE
        self._buf.clear()
        self._null_streak = 0
        self._locked_char = None
        self.word         = ""
        self.sentence     = ""

    def _on_null(self):
        self._null_streak += 1
        if self._state == _S.WRITTEN:
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
        if self._state == _S.WRITTEN and char == self._locked_char:
            return self.word, self.sentence
        if self._state == _S.WRITTEN:
            self._state = _S.LOCKING
            self._buf.clear()
            self._locked_char = None
        self._buf.append(char)
        if self._state == _S.IDLE:
            self._state = _S.LOCKING
        if len(self._buf) >= self.MODE_THRESHOLD:
            best, count = Counter(self._buf).most_common(1)[0]
            if count >= self.MODE_THRESHOLD and best != "YUMRUK":
                self.word        += best
                self._locked_char = best
                self._state       = _S.WRITTEN
                self._buf.clear()
        return self.word, self.sentence


class OfflineAutoCompleter:
    def __init__(self):
        self.vocab = [
            "MERHABA", "PROJE", "SISTEM", "KONTROL", "ROBOT", "YAPAYZEKA",
            "OTOMASYON", "MUHENDIS", "ITU", "BILGISAYAR", "GORUNTU", "ASL",
        ]

    def suggest(self, partial, n=3):
        if not partial:
            return []
        p = partial.upper()
        prefix = [w for w in self.vocab if w.startswith(p)]
        fuzzy  = difflib.get_close_matches(p, self.vocab, n=n, cutoff=0.4)
        return list(dict.fromkeys(prefix + fuzzy))[:n]


# ── Paylaşımlı durum (VideoProcessor ↔ Streamlit ana thread) ─────────────────

_lock       = threading.Lock()
_controller = ASLController()
_completer  = OfflineAutoCompleter()
_live_state = {
    "char": "",
    "conf": 0.0,
    "word": "",
    "sentence": "",
    "suggestions": [],
}


# ── WebRTC Video İşleyici ─────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)


class ASLVideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        model = load_model()
        img   = frame.to_ndarray(format="bgr24")

        results  = model.predict(img, imgsz=320, verbose=False)
        detected = ""
        conf     = 0.0
        if results[0].probs is not None:
            idx  = results[0].probs.top1
            conf = results[0].probs.top1conf.item()
            if conf >= 0.80:
                detected = model.names[idx]

        with _lock:
            word, sentence    = _controller.push_char(detected)
            suggestions       = _completer.suggest(word)
            _live_state["char"]        = detected
            _live_state["conf"]        = round(conf, 3)
            _live_state["word"]        = word
            _live_state["sentence"]    = sentence
            _live_state["suggestions"] = suggestions

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ── Yardımcı: tek görüntü üzerinde sınıflandırma ─────────────────────────────

def classify_image(img_bgr: np.ndarray):
    model   = load_model()
    results = model.predict(img_bgr, imgsz=320, verbose=False)
    if results[0].probs is None:
        return "", 0.0, results[0].plot()
    idx  = results[0].probs.top1
    conf = results[0].probs.top1conf.item()
    name = model.names[idx] if conf >= 0.50 else ""
    return name, conf, results[0].plot()


# ── Arayüz ────────────────────────────────────────────────────────────────────

st.title("ASL Character Recognition")
st.caption("TeknoGenç TeknoLAB 2025-2026  ·  YOLO11m-cls  ·  27 sınıf  ·  Çevrimdışı")

tab_live, tab_image, tab_perf = st.tabs(
    ["Canlı Çeviri", "Görüntü ile Test", "Performans Analizi"]
)

# ── Tab 1: Canlı Çeviri ───────────────────────────────────────────────────────

with tab_live:
    st.markdown(
        "Kameranıza elinizi gösterin. "
        "Son 20 karenin %70'i aynı harfi gösterince otomatik yazılır.  \n"
        "**YUMRUK gesturesi** kelimeyi sonlandırır ve boşluk ekler."
    )

    col_cam, col_nlp = st.columns([3, 2])

    with col_cam:
        ctx = webrtc_streamer(
            key="asl-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ASLVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_nlp:
        st.subheader("Çıktı")

        char_ph       = st.empty()
        word_ph       = st.empty()
        sentence_ph   = st.empty()
        suggest_ph    = st.empty()
        status_ph     = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        backspace_btn  = btn_col1.button("Geri Al (Sil)", use_container_width=True)
        clear_btn      = btn_col2.button("Temizle", use_container_width=True)

        if backspace_btn:
            with _lock:
                _controller.backspace()

        if clear_btn:
            with _lock:
                _controller.reset()
                _live_state.update({"char": "", "conf": 0.0, "word": "", "sentence": "", "suggestions": []})

        if ctx.state.playing:
            status_ph.success("Kamera aktif — YOLO11m-cls çalışıyor")
            while ctx.state.playing:
                with _lock:
                    s = _live_state.copy()
                char_ph.metric(
                    "Anlık Tespit",
                    s["char"] if s["char"] else "—",
                    delta=f"güven: {s['conf']:.0%}" if s["char"] else None,
                )
                word_ph.metric("Mevcut Kelime", s["word"] if s["word"] else "—")
                sentence_ph.text_area(
                    "Oluşan Cümle", s["sentence"], height=80, key="sent_live"
                )
                if s["suggestions"]:
                    suggest_ph.info("Tahminler: " + "  |  ".join(s["suggestions"]))
                else:
                    suggest_ph.empty()
        else:
            status_ph.info("Kamerayı başlatmak için yukarıdaki butona tıklayın.")
            with _lock:
                s = _live_state.copy()
            char_ph.metric("Anlık Tespit", s["char"] if s["char"] else "—")
            word_ph.metric("Mevcut Kelime", s["word"] if s["word"] else "—")
            sentence_ph.text_area("Oluşan Cümle", s["sentence"], height=80, key="sent_idle")

# ── Tab 2: Görüntü ile Test ───────────────────────────────────────────────────

with tab_image:
    st.markdown(
        "Bir ASL el işareti fotoğrafı yükleyin. "
        "Model güven skoru ile birlikte sınıfı tahmin eder."
    )

    uploaded = st.file_uploader(
        "Görüntü seçin (JPG / PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                caption="Yüklenen görüntü",
                use_column_width=True,
            )

        name, conf, annotated = classify_image(img_bgr)

        with col_pred:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Model çıktısı",
                use_column_width=True,
            )

        if name:
            st.success(f"Tahmin: **{name}**  —  Güven: **{conf:.1%}**")
        else:
            st.warning(f"Güven skoru düşük ({conf:.1%}), tahmin yapılmadı (eşik: %80)")

# ── Tab 3: Performans Analizi ─────────────────────────────────────────────────

with tab_perf:
    st.subheader("Model ve Eğitim Bilgileri")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model",            "YOLO11m-cls")
    col_m2.metric("Val Top-1 Acc",    "100%")
    col_m3.metric("Sınıf Sayısı",     "27")
    col_m4.metric("Eğitim Verisi",    "~78.300 görüntü")

    st.divider()

    results_path = os.path.join(os.path.dirname(__file__), "training_results", "results.png")
    cm_path      = os.path.join(os.path.dirname(__file__), "training_results", "confusion_matrix_normalized.png")
    val_path     = os.path.join(os.path.dirname(__file__), "training_results", "val_batch0_pred.jpg")
    train_path   = os.path.join(os.path.dirname(__file__), "training_results", "train_batch0.jpg")

    st.markdown("#### Eğitim Eğrileri (Loss & Accuracy)")
    if os.path.exists(results_path):
        st.image(results_path, use_column_width=True)
    else:
        st.warning("results.png bulunamadı.")

    col_cm, col_val = st.columns(2)
    with col_cm:
        st.markdown("#### Normalize Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)

    with col_val:
        st.markdown("#### Validation Tahminleri (Örnek Batch)")
        if os.path.exists(val_path):
            st.image(val_path, use_column_width=True)

    st.markdown("#### Eğitim Batch Örneği")
    if os.path.exists(train_path):
        st.image(train_path, use_column_width=True)

    st.divider()
    st.subheader("Eğitim Parametreleri")

    params = {
        "Model": "YOLO11m-cls (Medium — Fine-tuning)",
        "Epoch": "50 (erken durdurma ile ~15-20 epochta tamamlandı)",
        "Batch Size": "32",
        "Görüntü Boyutu": "320×320 px",
        "Optimizer": "SGD (momentum=0.937, weight_decay=0.0005)",
        "Patience (Erken Durdurma)": "10 epoch",
        "Donanım": "NVIDIA RTX 4060 Laptop GPU — CUDA 12.6",
        "Eğitim Süresi": "~18 dakika",
        "Veri Seti": "Kaggle grassknoted/asl-alphabet (87.000 görüntü, 29 sınıf)",
        "Kullanılan Sınıflar": "27 (A–Z + YUMRUK; NOTHING ve DELETE dışarıda bırakıldı)",
        "Train / Val Ayrımı": "%90 / %10 (yaklaşık 78.300 eğitim, 8.700 doğrulama)",
    }
    for k, v in params.items():
        st.markdown(f"- **{k}:** {v}")

    st.divider()
    st.subheader("Mimari Kararlar ve Alternatif Karşılaştırması")

    st.markdown("""
| Kriter | YOLO11n-cls (Denendi) | YOLO11m-cls (Seçildi) |
|---|---|---|
| Top-1 Val Acc | ~%60–70 | %100 |
| Inference Süresi | ~8 ms | ~14 ms |
| Model Boyutu | 5 MB | 21 MB |
| GPU Gereksinimi | Düşük | Orta (RTX 4060 yeterli) |

YOLO11n-cls ilk denemelerde yetersiz doğruluk verdi. Gerçek zamanlı kullanım için
14 ms inference süresi kabul edilebilir olduğundan ve RTX 4060 bu yükü rahatça
kaldırdığından YOLO11m-cls seçildi. Doğruluk hız optimizasyonundan önceliklidir.

**Streamlit vs PyQt5:**
Proje hem Streamlit (web erişimi, Hugging Face Spaces) hem PyQt5 (yüksek FPS, yerel)
arayüzü içermektedir. Gerçek zamanlı kamera akışında PyQt5 ~30 FPS, Streamlit WebRTC
~15 FPS sağlar. Sunum için Streamlit, üretim kullanımı için PyQt5 önerilir.
    """)
