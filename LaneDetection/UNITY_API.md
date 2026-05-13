🏎‍🟀 Unity Vehicle Remote API Dökümantasyonu
Bu API, Unity içindeki bir aracın dışarıdan (Python, C++, Browser vb.) kontrol edilmesini ve araç kamerasından anlık görüntü alınmasını sağlar.

📡 Genel Bilgiler
Base URL: http://localhost:8080
Protokol: HTTP/1.1
Görüntü Formatı: JPEG
🛠️ Endpoints (Uç Noktalar)

1. Araç Kontrolü (/control)
   Aracın direksiyon, gaz/fren ve el freni durumunu günceller.

Method: GET
Parametreler:
steer (float): Direksiyon açısı. -1.0 (Tam Sol) ile 1.0 (Tam Sağ) arası. 0.0 tekerleri düzeltir.
forward (float): Gaz ve Fren. 1.0 (Tam Gaz İleri), -1.0 (Tam Gaz Geri/Fren).
park (float): El freni. 1.0 (Çekili), 0.0 (Serbest).
Örnek İstek: http://localhost:8080/control?steer=0.5&forward=1.0&park=0
Yanıt: OK (Hata yoksa) 2. Kamera Görüntüsü (/camera)
Unity'deki hedeflenen kameradan o anki kareyi yakalar ve gönderir.

Method: GET
Yanıt: image/jpeg tipinde binary veri.
Örnek İstek: http://localhost:8080/camera
⚙️ Unity Kurulumu
Script Atama: VehicleAPIController.cs scriptini sahnedeki boş bir objeye veya araca atayın.
Inspector Ayarları:
Port: Varsayılan 8080.
Target Camera: Görüntü almak istediğiniz kamerayı (Örn: Araç ön kamerası) buraya sürükleyin.
Car Controller: Sahnedeki aracın üzerindeki DemoCarController bileşenini buraya sürükleyin.
Image Resolution: Varsayılan 640x360. FPS artırmak için düşürebilirsiniz.
Başlatma: Unity'de Play butonuna basarak sunucuyu aktif hale getirin.
🐍 Python Kullanım Örneği
En hızlı ve kararlı kontrol için requests ve pynput kütüphaneleri önerilir.

python
import requests

# Kontrol Gönderme

def send_command(steer, forward, park):
url = f"http://localhost:8080/control?steer={steer}&forward={forward}&park={park}"
requests.get(url)

# Görüntü Alma (OpenCV ile)

import cv2
import numpy as np
def get_frame():
response = requests.get("http://localhost:8080/camera")
nparr = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
return img
🚀 Performans Notları
FPS: Görüntü FPS'i Unity'nin o anki kare hızına ve network gecikmesine bağlıdır. 640x360 çözünürlük ideal dengedir.
Multi-Key: Birden fazla tuşa aynı anda basıldığında (W+D gibi) sorun yaşanmaması için Python tarafında pynput gibi "key state" tutan kütüphaneler kullanılmalıdır.
