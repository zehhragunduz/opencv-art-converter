# 🎨 OpenCV Cartoon & Sketch Converter

Bu proje, Python ve OpenCV kullanılarak geliştirilmiş, gerçek zamanlı bir görüntü işleme uygulamasıdır. Webcam görüntüsü veya statik görseller üzerinde **"Cartoon" (Karikatür)** ve **"Pencil Sketch" (Kara Kalem)** efektlerini uygular ve sonuçları "Öncesi - Sonrası" formatında sunar.

## 🚀 Özellikler
* **Canlı Webcam Desteği:** `CAP_DSHOW` backend desteği ile Windows ortamında sorunsuz kamera akışı.
* **İkili Efekt Modu:**
  * **Sketch:** Gri tonlama, ters çevirme ve "Color Dodge" harmanlama modu ile kurşun kalem efekti.
  * **Cartoon:** Bilateral filtreleme (kenar koruma) ve adaptif eşikleme (adaptive thresholding) kombinasyonu.
* **Türkçe Karakter Desteği:** OpenCV'nin standart font eksikliğini gidermek için **Pillow (PIL)** kütüphanesi ile TrueType font entegrasyonu.
* **Akıllı Pencere Yönetimi:** Görüntülerin ekrandan taşmasını engelleyen dinamik ölçekleme (resizing) ve istifleme algoritması.

## 🛠️ Kullanılan Teknolojiler
* **Python 3**
* **OpenCV (cv2):** Görüntü işleme ve filtreleme algoritmaları.
* **NumPy:** Matris işlemleri.
* **Pillow (PIL):** Gelişmiş metin ve font render işlemleri.

## ⚙️ Algoritma Detayları

### 1. Pencil Sketch Efekti
Gri tonlamalı görüntü ters çevrilir (Invert) ve Gauss bulanıklığı uygulanır. Ardından orijinal gri görüntü ile bulanıklaştırılmış ters görüntü **Color Dodge** yöntemiyle bölünerek çizim hatları ortaya çıkarılır.

### 2. Cartoon Efekti
Görüntüdeki gürültüyü azaltmak ancak kenarları keskin tutmak için tekrarlı **Bilateral Filter** uygulanır. Ardından **Adaptive Thresholding** ile kenar maskesi çıkarılır ve renklendirilmiş görüntü ile birleştirilir.

## 🐛 Çözülen Zorluklar (Challenges & Solutions)

Geliştirme sürecinde karşılaşılan ve çözülen kritik problemler:

1.  **Pencere Taşma Sorunu:**
    * *Sorun:* Webcam çözünürlüğünün yan yana eklenince ekranı taşırması.
    * *Çözüm:* Kaynağı işlemden önce küçültme ve en son aşamada hedef pencere boyutuna göre "Force Resize" işlemi uygulandı.

2.  **Türkçe Karakter Sorunu:**
    * *Sorun:* `cv2.putText` fonksiyonunun Türkçe karakterleri (ş, ğ, ü) "?" olarak göstermesi.
    * *Çözüm:* Pillow kütüphanesi ile metinler bir katman (layer) olarak çizdirildi, Pillow olmayan sistemler için ASCII fallback mekanizması eklendi.

## 💻 Kurulum ve Kullanım

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

2.  Uygulamayı başlatın:
    ```bash
    python main.py
    ```
    *(Varsayılan olarak webcam açılacaktır)*

---
*Geliştirici: Zehra Gündüz - AIHEXA Staj Projesi*