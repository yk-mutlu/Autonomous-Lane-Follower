import cv2
import numpy as np
import requests
import time
import threading
import json
from pynput import keyboard

# Klasik algoritma tabanli serit tespiti (JustNikhill)
import algorithm as algo

BASE_URL = "http://localhost:8080"

# Hedef hiz (km/h) - Ekrandaki butonlarla degistirilebilir
TARGET_SPEED = 70  

state = {
    "steer": 0.0,
    "forward": 0.0,
    "park": 0.0,
    "running": True,
    "auto_mode": False,
    "speed_kmh": 0.0,
    "w": 640  # Goruntu genisligini takip etmek icin
}

pressed_keys = set()

def on_press(key):
    try:
        k = key.char.lower()
        pressed_keys.add(k)
        if k == 'm':
            state["auto_mode"] = not state["auto_mode"]
            print(f"Otonom Surus: {'ACIK' if state['auto_mode'] else 'KAPALI'}")
    except AttributeError:
        if key == keyboard.Key.space:
            pressed_keys.add('space')

def on_release(key):
    try:
        k = key.char.lower()
        if k in pressed_keys:
            pressed_keys.remove(k)
    except AttributeError:
        if key == keyboard.Key.space:
            if 'space' in pressed_keys:
                pressed_keys.remove('space')
    if key == keyboard.Key.esc:
        state["running"] = False
        return False

# Fare tiklama olayini yakalayan fonksiyon (TARGET_SPEED degistirmek icin)
def handle_mouse(event, x, y, flags, param):
    global TARGET_SPEED
    if event == cv2.EVENT_LBUTTONDOWN:
        w = state["w"]
        # + butonu kontrolü (Hedef hizi artir)
        if w - 50 <= x <= w - 5 and 60 <= y <= 95:
            TARGET_SPEED = min(300, TARGET_SPEED + 10)
            print(f"Hedef Hiz Artirildi: {TARGET_SPEED} km/h")
        # - butonu kontrolü (Hedef hizi azalt)
        elif w - 100 <= x <= w - 55 and 60 <= y <= 95:
            TARGET_SPEED = max(20, TARGET_SPEED - 10)
            print(f"Hedef Hiz Azaltildi: {TARGET_SPEED} km/h")

def update_manual_controls():
    while state["running"]:
        if not state["auto_mode"]:
            # Ileri/Geri
            if 'w' in pressed_keys:
                state["forward"] = 1.0
            elif 's' in pressed_keys:
                state["forward"] = -1.0
            else:
                state["forward"] = 0.0
                
            # Sag/Sol
            if 'a' in pressed_keys:
                state["steer"] = -1.0
            elif 'd' in pressed_keys:
                state["steer"] = 1.0
            else:
                state["steer"] = 0.0
                
            # Fren (Space)
            if 'space' in pressed_keys:
                state["park"] = 1.0
            else:
                state["park"] = 0.0
                
        time.sleep(0.01)

def fetch_speed_loop():
    """Unity'den arac hizini periyodik olarak ceker."""
    session = requests.Session()
    while state["running"]:
        try:
            resp = session.get(f"{BASE_URL}/status", timeout=0.2)
            if resp.status_code == 200:
                data = json.loads(resp.text)
                state["speed_kmh"] = data.get("speed_kmh", 0.0)
        except:
            pass
        time.sleep(0.1)

def send_control_loop():
    session = requests.Session()
    while state["running"]:
        # Unity'nin C# float.TryParse fonksiyonunun bilimsel gosterimi (e-15) 
        # yanlis okumasini engellemek icin degerleri sabit noktali string'e ceviriyoruz.
        params = {
            "steer": f"{float(state['steer']):.4f}",
            "forward": f"{float(state['forward']):.4f}",
            "park": f"{float(state['park']):.4f}"
        }
        try:
            session.get(f"{BASE_URL}/control", params=params, timeout=0.1)
        except:
            pass
        time.sleep(0.03) # Saniyede ~30 istek

def main():
    print("--- Klasik Otonom Arac Kontrolcusu (AI Yok) ---")
    print("Kontroller:")
    print("  'M' tusu : Otonom Surusu AC/KAPAT")
    print("  WASD     : Manuel kontrol (Otonom kapaliyken)")
    print("  SPACE    : El freni")
    print("  ESC      : Cikis\n")
    print("NOT: Unity uzerinden Play butonuna basin, ardindan 'M' tusuyla otonomu baslatin!")
    
    cv2.namedWindow("Autonomous Driving (Classic PID)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Autonomous Driving (Classic PID)", handle_mouse)
    session = requests.Session()
    
    # Klavye dinleyicisi
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Threadler
    threading.Thread(target=update_manual_controls, daemon=True).start()
    threading.Thread(target=send_control_loop, daemon=True).start()
    threading.Thread(target=fetch_speed_loop, daemon=True).start()
    
    # PID Kontrolcu Parametreleri
    Kp = 0.0031
    Ki = 0.00005  # Integral katsayisi
    Kd = 0.135    # Dusuk Kd = daha az titresim
    last_error = 9.81  # Kalibrasyon sonucu sabit baslangic
    integral = 1689.09  # Kalibrasyon sonucu sabit baslangic
    smoothed_steer = 0.0
    last_derivative = 0.0
    error_history = []
    ERROR_HISTORY_SIZE = 7
    
    try:
        while state["running"]:
            try:
                response = session.get(f"{BASE_URL}/camera", timeout=0.5)
                if response.status_code == 200:
                    nparr = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        try:
                            # Kullanicinin algoritmasini calistir
                            algo.first_frame = getattr(algo, '_initialized', 1)
                            algo.current_speed = state["speed_kmh"]  # Hizi algoritmaya aktar (akan cizgiler icin)
                            processed = algo.process_image(frame)
                            algo._initialized = 0  # Sonraki kareler icin first_frame = 0 kalmali
                            h, w = frame.shape[:2]
                            state["w"] = w  # Mouse callback icin genisligi guncelle
                            
                            if state["auto_mode"]:
                                # algorithm.cache: [x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r]
                                lane_cx = -1
                                if algo.cache is not None and len(algo.cache) == 8:
                                    x2_l = algo.cache[2]  # sol cizginin ust (ilerideki) x koordinati
                                    x2_r = algo.cache[6]  # sag cizginin ust (ilerideki) x koordinati
                                    lane_cx = (x2_l + x2_r) / 2.0
                                
                                if lane_cx != -1 and not np.isnan(lane_cx):
                                    raw_error = lane_cx - (w / 2)
                                    
                                    # Median filtre
                                    error_history.append(raw_error)
                                    if len(error_history) > ERROR_HISTORY_SIZE:
                                        error_history.pop(0)
                                    error = float(np.median(error_history))
                                    
                                    # Noise Filter
                                    if abs(error - last_error) > 30:
                                        error = last_error + (30 if error > last_error else -30)
                                        
                                    raw_derivative = error - last_error
                                    # Turev filtresi (D teriminin titretmesini onler)
                                    derivative = (0.6 * last_derivative) + (0.4 * raw_derivative)
                                    last_derivative = derivative
                                    
                                    integral += error
                                    
                                    # Integral Anti-Windup
                                    if integral > 2000: integral = 2000
                                    elif integral < -2000: integral = -2000
                                    
                                    last_error = error
                                    
                                    # Kucuk hatalari yoksay (Olu Bant - Deadband) titremeyi engeller
                                    if abs(error) < 5:
                                        error = 0
                                        derivative = 0
                                    
                                    steer_val = (error * Kp) + (integral * Ki) + (derivative * Kd)
                                    steer_val = max(-1.0, min(1.0, steer_val))
                                    
                                    # Direksiyon cikisini daha da yumusat
                                    smoothed_steer = (0.85 * smoothed_steer) + (0.15 * steer_val)
                                    state["steer"] = smoothed_steer
                                    
                                    # Dinamik Hiz Kontrolu (Hataya Bagli)
                                    # Hata buyudukce hedef hizi dusur (virajlarda yavasla)
                                    error_ratio = min(abs(error) / 120.0, 1.0)  # 0-1 arasi normalize
                                    effective_speed = TARGET_SPEED * (1.0 - error_ratio * 0.5)  # Max %50 dusus
                                    effective_speed = max(30, effective_speed)  # En az 30 km/h
                                    
                                    slowing_range = effective_speed * 0.1
                                    slowing_start = effective_speed - slowing_range
                                    
                                    if state["speed_kmh"] >= effective_speed:
                                        state["forward"] = 0.0
                                    elif state["speed_kmh"] <= slowing_start:
                                        state["forward"] = 1.0 # Tam gaz
                                    else:
                                        dynamic_gas = (effective_speed - state["speed_kmh"]) / slowing_range
                                        state["forward"] = max(0.0, min(1.0, dynamic_gas))
                                        
                                    state["park"] = 0.0
                                    
                                    cv2.putText(processed, f"OTO: ACIK | Steer: {smoothed_steer:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    cv2.line(processed, (int(w/2), h), (int(lane_cx), int(h*0.8)), (0, 255, 255), 2)
                                else:
                                    # Serit kaybedildiginde, en son gordugu yone dogru kirmaya devam etmeli
                                    # Boylece kaybettigi seridi arar.
                                    steer_val = (last_error * Kp) + (integral * Ki) # Turev 0, sadece son hataya ve integrale gore don
                                    steer_val = max(-1.0, min(1.0, steer_val))
                                    smoothed_steer = (0.80 * smoothed_steer) + (0.20 * steer_val)
                                    state["steer"] = smoothed_steer
                                    state["forward"] = 0.3 # Serit aranirken yavasla
                                    state["park"] = 0.0
                                    cv2.putText(processed, "OTO: SERIT ARANIYOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(processed, "OTO: KAPALI (Manuel WASD)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                                
                            # Hiz gostergesi (sag ust kose)
                            speed = state["speed_kmh"]
                            speed_text = f"{speed:.0f} km/h"
                            (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                            cv2.rectangle(processed, (w - tw - 25, 10), (w - 5, th + 25), (0, 0, 0), -1)
                            cv2.rectangle(processed, (w - tw - 25, 10), (w - 5, th + 25), (0, 255, 255), 1)
                            cv2.putText(processed, speed_text, (w - tw - 15, th + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            
                            # Maks Hiz Ayar Butonlari (+ / -)
                            # - Butonu
                            cv2.rectangle(processed, (w - 100, 60), (w - 55, 95), (50, 50, 50), -1)
                            cv2.rectangle(processed, (w - 100, 60), (w - 55, 95), (0, 255, 255), 1)
                            cv2.putText(processed, "-", (w - 85, 87), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            # + Butonu
                            cv2.rectangle(processed, (w - 50, 60), (w - 5, 95), (50, 50, 50), -1)
                            cv2.rectangle(processed, (w - 50, 60), (w - 5, 95), (0, 255, 255), 1)
                            cv2.putText(processed, "+", (w - 38, 87), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            # Mevcut Hedef Hiz Degeri (Target)
                            cv2.putText(processed, f"Hedef: {TARGET_SPEED} km/h", (w - 130, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            cv2.imshow("Autonomous Driving (Classic PID)", processed)
                        except Exception as e:
                            # Hata durumunda ekrani kilitlememesi icin sadece orjinal kareyi goster
                            cv2.imshow("Autonomous Driving (Classic PID)", frame)
                            print(f"Algorithm error: {e}")
            except requests.exceptions.RequestException:
                time.sleep(0.5)
            except Exception as e:
                print(f"Network/Decode error: {e}")
                pass
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        cv2.destroyAllWindows()
        print("\nCikiliyor...")
        try:
            requests.get(f"{BASE_URL}/control?steer=0&forward=0&park=1", timeout=1)
        except:
            pass

if __name__ == "__main__":
    main()
