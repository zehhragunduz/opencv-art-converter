#!/usr/bin/env python3
"""
Karikatür / Sketch Efekti – Mini Proje (OpenCV)
------------------------------------------------
Bu script, bir görüntü üzerinde iki farklı çıktı üretir:
  1) Pencil Sketch (kurşun kalem çizimi etkisi)
  2) Cartoon (çizgi film) etkisi
Ayrıca önce–sonra (yan yana) kolajlarını kaydeder.

Kullanım:
  python cartoon_sketch.py --input path/to/image.jpg

Opsiyonel:
  python cartoon_sketch.py --input image.jpg --show           # Pencerede göster
  python cartoon_sketch.py --input image.jpg --no-save        # Kaydetmeden sadece göster
  python cartoon_sketch.py --webcam                           # Webcam canlı önizleme

Gereksinimler:
  pip install opencv-python numpy

Çıktılar (varsayılan out/ klasörüne kaydedilir):
  - *_sketch.png            : Sketch çıktısı
  - *_cartoon.png           : Cartoon çıktısı
  - *_before_after_sketch.png
  - *_before_after_cartoon.png

Notlar:
  - Sketch: "color dodge" yaklaşımı kullanır (invert + blur + divide)
  - Cartoon: bilateral filter + kenar maskesi (adaptive threshold)
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False



def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return img


def resize_max(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_sketch(img_bgr: np.ndarray, blur_ksize: int = 21, blur_sigma: int = 0) -> np.ndarray:
    """Kurşun kalem çizimi etkisi (grayscale)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), blur_sigma)
    # color dodge: divide(gray, 255 - blur)
    dodge = cv2.divide(gray, 255 - blur, scale=256)
    sketch_bgr = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)
    return sketch_bgr


def to_cartoon(img_bgr: np.ndarray,
               bilateral_iter: int = 5,
               d: int = 9,
               sigma_color: int = 75,
               sigma_space: int = 75,
               edge_block_size: int = 9,
               edge_C: int = 2) -> np.ndarray:
    """Çizgi film etkisi: yumuşatma + kenar maskesi."""
    filtered = img_bgr.copy()
    for _ in range(bilateral_iter):
        filtered = cv2.bilateralFilter(filtered, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=edge_block_size, C=edge_C)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    cartoon = cv2.bitwise_and(filtered, edges)
    return cartoon

def _ascii_fallback(text: str) -> str:
    mapping = str.maketrans({"Ç":"C","Ğ":"G","İ":"I","Ö":"O","Ş":"S","Ü":"U",
                             "ç":"c","ğ":"g","ı":"i","ö":"o","ş":"s","ü":"u"})
    return text.translate(mapping)

def draw_text_utf8(img, text, xy, font_size=24, color=(50,50,50), bold=False):
    # PIL varsa TrueType font ile yaz; yoksa ASCII'ye düş.
    if PIL_AVAILABLE:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img[:, :, ::-1])  # BGR->RGB
        draw = ImageDraw.Draw(pil)
        font = None
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
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

def fit_to_window(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(max_w/w, max_h/h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img


def make_before_after(original: np.ndarray, processed: np.ndarray, label_left: str, label_right: str) -> np.ndarray:
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    h = min(h1, h2)
    if h1 != h:
        original = cv2.resize(original, (int(w1 * h / h1), h), interpolation=cv2.INTER_AREA)
    if h2 != h:
        processed = cv2.resize(processed, (int(w2 * h / h2), h), interpolation=cv2.INTER_AREA)

    pad = 50
    w_total = original.shape[1] + processed.shape[1]
    bar = np.full((pad, w_total, 3), 245, dtype=np.uint8)

    def put(text, x, y):
        cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)

    put(label_left, 15, 32)
    put(label_right, original.shape[1] + 15, 32)

    combo = np.hstack([original, processed])
    out = np.vstack([bar, combo])
    return out


def auto_basename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


def process_image(path: str, out_dir: str, show: bool, save: bool) -> None:
    ensure_out_dir(out_dir)
    img = read_image(path)
    img = resize_max(img, 1600)

    sketch = to_sketch(img)
    cartoon = to_cartoon(img)

    before_after_sketch = make_before_after(img, sketch, "Önce", "Sketch")
    before_after_cartoon = make_before_after(img, cartoon, "Önce", "Cartoon")

    if save:
        base = auto_basename(path)
        cv2.imwrite(os.path.join(out_dir, f"{base}_sketch.png"), sketch)
        cv2.imwrite(os.path.join(out_dir, f"{base}_cartoon.png"), cartoon)
        cv2.imwrite(os.path.join(out_dir, f"{base}_before_after_sketch.png"), before_after_sketch)
        cv2.imwrite(os.path.join(out_dir, f"{base}_before_after_cartoon.png"), before_after_cartoon)
        print(f"Kaydedildi: {out_dir}")

    if show:
        cv2.imshow("Sketch", before_after_sketch)
        cv2.imshow("Cartoon", before_after_cartoon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def webcam_loop() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam açılamadı")
    try:
        print("Webcam açık. Çıkmak için 'q' tuşuna basın.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize_max(frame, 960)
            sketch = to_sketch(frame)
            cartoon = to_cartoon(frame)

            view1 = make_before_after(frame, sketch, "Önce", "Sketch")
            view2 = make_before_after(frame, cartoon, "Önce", "Cartoon")

            # İki görünümü yatay olarak yan yana yerleştir
            if view1.shape[0] != view2.shape[0]:
                h = min(view1.shape[0], view2.shape[0])
                view1 = cv2.resize(view1, (int(view1.shape[1] * h / view1.shape[0]), h), interpolation=cv2.INTER_AREA)
                view2 = cv2.resize(view2, (int(view2.shape[1] * h / view2.shape[0]), h), interpolation=cv2.INTER_AREA)

            stacked = np.hstack([view1, view2])

            # Üstte bilgi yazısı ekle
            info_bar = np.full((40, stacked.shape[1], 3), 240, dtype=np.uint8)
            cv2.putText(info_bar, "Cartoon & Sketch – q ile çıkış", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            final_view = np.vstack([info_bar, stacked])

            cv2.imshow("Cartoon & Sketch – Önce/Sonra", final_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartoon/Sketch efekti ile önce-sonra çıktı üreten mini proje")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Giriş görüntü yolu")
    src.add_argument("--webcam", action="store_true", help="Webcam'den canlı önizleme")

    parser.add_argument("--out", type=str, default="out", help="Çıktı klasörü (varsayılan: out)")
    parser.add_argument("--show", action="store_true", help="Sonuç pencerelerini göster")
    parser.add_argument("--no-save", action="store_true", help="Dosyaları kaydetme")
    parser.add_argument("--maxw", type=int, default=1280, help="Pencere için maksimum genişlik")
    parser.add_argument("--maxh", type=int, default=720, help="Pencere için maksimum yükseklik")
    parser.add_argument("--downscale", type=int, default=480, help="Webcam kare uzun kenar limiti")
    parser.add_argument("--layout", type=str, default="vertical",
                    choices=["vertical","horizontal","cartoon","sketch"],
                    help="Görüntü düzeni: dikey, yatay, sadece cartoon veya sadece sketch")


    args = parser.parse_args()
    save = not args.no_save

    if args.webcam:
        webcam_loop(max_w=args.maxw, max_h=args.maxh)
    else:
        process_image(args.input, args.out, show=args.show, save=save)
