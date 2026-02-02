#!/usr/bin/env python3
"""
Karikatür / Sketch Efekti – Mini Proje (OpenCV) – ARGÜMANSIZ SİĞDIRMA SÜRÜMÜ
- CLI: sadece (--input | --webcam), --out, --show, --no-save
- Webcam: pencereye sığdırma (SABİT hedef: 1280x720), kaynak kare küçültme (SABİT: 480px)
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np

# ---------- AYAR SABİTLERİ (gerekirse aşağıdaki 3 değeri değiştir) ----------
TARGET_W = 1280   # pencere hedef genişlik
TARGET_H = 720    # pencere hedef yükseklik
FRAME_MAX = 480   # webcamden gelen karenin uzun kenar limiti (işlem öncesi küçültme)
# ---------------------------------------------------------------------------

# ——— Unicode güvenli başlık yazımı (Pillow varsa TTF, yoksa ASCII fallback) ———
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

def _ascii_fallback(text: str) -> str:
    mapping = str.maketrans({"Ç":"C","Ğ":"G","İ":"I","Ö":"O","Ş":"S","Ü":"U",
                             "ç":"c","ğ":"g","ı":"i","ö":"o","ş":"s","ü":"u"})
    return text.translate(mapping)

def draw_text_utf8(img: np.ndarray, text: str, xy: tuple,
                   font_size: int = 24, color=(50,50,50), bold: bool=False) -> np.ndarray:
    if PIL_AVAILABLE:
        pil = Image.fromarray(img[:, :, ::-1])  # BGR->RGB
        draw = ImageDraw.Draw(pil)
        # basit font arama
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        font = None
        for p in candidates:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        draw.text(xy, text, fill=(color[2], color[1], color[0]), font=font)
        return np.array(pil)[:, :, ::-1]  # RGB->BGR
    else:
        safe = _ascii_fallback(text)
        thickness = 2 if bold else 1
        cv2.putText(img, safe, xy, cv2.FONT_HERSHEY_SIMPLEX, font_size/30.0, color, thickness, cv2.LINE_AA)
        return img

# ——— Yardımcılar ———
def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return img

def resize_max(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def fit_to_window(img: np.ndarray, max_w: int = TARGET_W, max_h: int = TARGET_H) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

# ——— Efektler ———
def to_sketch(img_bgr: np.ndarray, blur_ksize: int = 21, blur_sigma: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), blur_sigma)
    dodge = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)

def to_cartoon(img_bgr: np.ndarray,
               bilateral_iter: int = 5, d: int = 9,
               sigma_color: int = 75, sigma_space: int = 75,
               edge_block_size: int = 9, edge_C: int = 2) -> np.ndarray:
    filtered = img_bgr.copy()
    for _ in range(bilateral_iter):
        filtered = cv2.bilateralFilter(filtered, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=edge_block_size, C=edge_C)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(filtered, edges)

# ——— Önce–sonra kolaj ———
def make_before_after(original: np.ndarray, processed: np.ndarray,
                      label_left: str, label_right: str) -> np.ndarray:
    h = min(original.shape[0], processed.shape[0])
    original = cv2.resize(original, (int(original.shape[1]*h/original.shape[0]), h), interpolation=cv2.INTER_AREA)
    processed = cv2.resize(processed, (int(processed.shape[1]*h/processed.shape[0]), h), interpolation=cv2.INTER_AREA)
    combo = np.hstack([original, processed])
    bar = np.full((48, combo.shape[1], 3), 245, dtype=np.uint8)
    bar = draw_text_utf8(bar, label_left, (15, 32), font_size=24, color=(50,50,50), bold=True)
    bar = draw_text_utf8(bar, label_right, (original.shape[1]+15, 32), font_size=24, color=(50,50,50), bold=True)
    return np.vstack([bar, combo])

def auto_basename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"

# ——— Tek görsel işleme ———
def process_image(path: str, out_dir: str, show: bool, save: bool) -> None:
    ensure_out_dir(out_dir)
    img = read_image(path)
    img = resize_max(img, 1600)
    sketch = to_sketch(img)
    cartoon = to_cartoon(img)
    ba_sketch = make_before_after(img, sketch, "Önce", "Sketch")
    ba_cartoon = make_before_after(img, cartoon, "Önce", "Cartoon")
    if save:
        base = auto_basename(path)
        cv2.imwrite(os.path.join(out_dir, f"{base}_sketch.png"), sketch)
        cv2.imwrite(os.path.join(out_dir, f"{base}_cartoon.png"), cartoon)
        cv2.imwrite(os.path.join(out_dir, f"{base}_before_after_sketch.png"), ba_sketch)
        cv2.imwrite(os.path.join(out_dir, f"{base}_before_after_cartoon.png"), ba_cartoon)
        print(f"Kaydedildi: {out_dir}")
    if show:
        cv2.imshow("Sketch", fit_to_window(ba_sketch))
        cv2.imshow("Cartoon", fit_to_window(ba_cartoon))
        cv2.waitKey(0); cv2.destroyAllWindows()

# ——— Webcam canlı önizleme (ARGÜMANSIZ SİĞDIRMA) ———
def webcam_loop() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Windows'ta bazen dshow daha sağlıklı
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Webcam açılamadı (0). Başka uygulama kamerayı kullanıyor olabilir.")

    win = "Cartoon & Sketch – Önce/Sonra"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)      # pencereyi esnek aç
    cv2.resizeWindow(win, TARGET_W, TARGET_H)    # hedef boyut

    print("Webcam açık. 'q' ile çıkış.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kare okunamadı.")
                break

            # KAYNAK KAREYİ KÜÇÜLT (taşmayı önleyen kritik adım)
            frame = resize_max(frame, FRAME_MAX)

            # iki paneli dikey istifle (ekrana sığdırması en kolay olan)
            top = make_before_after(frame, to_sketch(frame),  "Önce", "Sketch")
            bot = make_before_after(frame, to_cartoon(frame), "Önce", "Cartoon")
            view = np.vstack([top, bot])

            # üst bilgi çubuğu
            bar = np.full((40, view.shape[1], 3), 240, dtype=np.uint8)
            bar = draw_text_utf8(bar, "Canlı: 'q'=çıkış | üst: Sketch | alt: Cartoon", (10, 28),
                                 font_size=24, color=(0,0,0), bold=True)

            final_view = np.vstack([bar, view])

            # SON: pencereye zorla sığdır
            final_view = fit_to_window(final_view, TARGET_W, TARGET_H)

            cv2.imshow(win, final_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ——— CLI ———
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoon/Sketch efekti ile önce-sonra çıktı üreten mini proje")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Giriş görüntü yolu")
    src.add_argument("--webcam", action="store_true", help="Webcam'den canlı önizleme")

    parser.add_argument("--out", type=str, default="out", help="Çıktı klasörü (varsayılan: out)")
    parser.add_argument("--show", action="store_true", help="Sonuç pencerelerini göster")
    parser.add_argument("--no-save", action="store_true", help="Dosyaları kaydetme")

    args = parser.parse_args()
    save = not args.no_save

    if args.webcam:
        webcam_loop()
    else:
        process_image(args.input, args.out, show=args.show, save=save)
