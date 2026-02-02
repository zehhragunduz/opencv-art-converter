import cv2
import numpy as np

def as_cartoon(frame):
    # Gürültü azalt + kenar çıkar + bilateral ile düzleştir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def as_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def as_background_blur(frame):
    # Basit kişi/nesne maskeleme: yüksek frekanslı bölgeleri nesne farz etme (çok kaba)
    # İstersen burayı daha iyi bir segmentasyonla geliştirebilirsin.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Kenarları ve dokuyu vurgula
    texture = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    # Otsu ile maske üret
    _, mask = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Arka planı bulanıklaştır
    blurred = cv2.GaussianBlur(frame, (25,25), 0)
    out = (frame*(mask3/255.0) + blurred*(1-mask3/255.0)).astype(np.uint8)
    return out

def put_label(img, text):
    cv2.rectangle(img, (0,0), (img.shape[1], 36), (0,0,0), -1)
    cv2.putText(img, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    mode = 1  # 1: edges, 2: cartoon, 3: bg blur

    print("Kısayollar: [1] Kenar | [2] Cartoon | [3] Arka Plan Blur | [q] Çıkış")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mode == 1:
            out = as_edges(frame)
            put_label(out, "Kenar (1) | Cartoon (2) | Arka Plan Blur (3) | q: Cikis")
        elif mode == 2:
            out = as_cartoon(frame)
            put_label(out, "Cartoon (2) | Kenar (1) | Arka Plan Blur (3) | q: Cikis")
        else:
            out = as_background_blur(frame)
            put_label(out, "Arka Plan Blur (3) | Kenar (1) | Cartoon (2) | q: Cikis")

        cv2.imshow("Live Filters", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
