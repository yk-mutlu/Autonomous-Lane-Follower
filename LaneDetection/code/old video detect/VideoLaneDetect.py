"""
Video Lane Detection - Tesla Style
Simülasyon algoritmasının aynısını video dosyalarına uygular.
Kullanım: python VideoLaneDetect.py
"""

import cv2
import numpy as np
import os
import sys
import algorithm as algo

# ===================== AYARLAR =====================
DATA_DIR = "./data"           # Video dosyalarinin bulundugu klasor
RESULT_DIR = "./result"       # Sonuclarin kaydedilecegi klasor
FAKE_SPEED = 80.0             # Akan cizgilerin hizi icin sahte hiz (km/h)
# ===================================================

def list_videos(directory):
    """Klasordeki video dosyalarini listele."""
    supported = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
    videos = []
    if os.path.exists(directory):
        for f in sorted(os.listdir(directory)):
            if f.lower().endswith(supported):
                videos.append(f)
    return videos

def process_video(video_path, output_path=None):
    """Videoyu kare kare isle ve sonucu goster/kaydet."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video acilamadi: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Cozunurluk: {width}x{height} | FPS: {fps:.1f} | Toplam Kare: {total_frames}")
    
    # Video yazici (kayit icin)
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Kayit: {output_path}")
    
    # Algorithm state'ini sifirla
    algo.cache = None
    algo.first_frame = 1
    algo.frame_counter = 0
    algo.scroll_offset = 0.0
    algo.current_speed = FAKE_SPEED
    
    cv2.namedWindow("Lane Detection - Video", cv2.WINDOW_NORMAL)
    
    frame_num = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video sona erdi.")
                break
            
            frame_num += 1
            
            # Algoritmamizi uygula
            try:
                processed = algo.process_image(frame)
            except Exception as e:
                processed = frame.copy()
            
            # Bilgi paneli
            info_text = f"Kare: {frame_num}/{total_frames}"
            cv2.putText(processed, info_text, (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            progress = frame_num / max(total_frames, 1)
            bar_w = width - 40
            cv2.rectangle(processed, (20, height - 25), (20 + bar_w, height - 10), (50, 50, 50), -1)
            cv2.rectangle(processed, (20, height - 25), (20 + int(bar_w * progress), height - 10), (0, 255, 255), -1)
            
            cv2.imshow("Lane Detection - Video", processed)
            
            if writer:
                writer.write(processed)
        
        # Klavye kontrolleri
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space = Duraklat/Devam
            paused = not paused
            print("DURAKLATILDI" if paused else "DEVAM EDILIYOR")
        elif key == ord('r'):  # R = Basa sar
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            algo.cache = None
            algo.first_frame = 1
            algo.frame_counter = 0
            algo.scroll_offset = 0.0
            frame_num = 0
            print("Basa sarildi.")
        elif key == ord('+') or key == ord('='):
            FAKE_SPEED = min(300, algo.current_speed + 10)
            algo.current_speed = FAKE_SPEED
            print(f"Animasyon hizi: {FAKE_SPEED}")
        elif key == ord('-'):
            FAKE_SPEED = max(0, algo.current_speed - 10)
            algo.current_speed = FAKE_SPEED
            print(f"Animasyon hizi: {FAKE_SPEED}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 50)
    print("  TESLA STYLE VIDEO SERIT TESPIT SISTEMI")
    print("=" * 50)
    print(f"\nKontroller:")
    print(f"  SPACE  : Duraklat / Devam")
    print(f"  R      : Basa Sar")
    print(f"  +/-    : Animasyon hizini artir/azalt")
    print(f"  ESC    : Cikis / Sonraki Video\n")
    
    videos = list_videos(DATA_DIR)
    
    if not videos:
        print(f"HATA: '{DATA_DIR}' klasorunde video bulunamadi!")
        print("Lutfen data/ klasorune .mp4 dosyalari ekleyin.")
        return
    
    print(f"{len(videos)} video bulundu:\n")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v}")
    
    print(f"\n  0. Tumunu isle ve kaydet")
    
    try:
        choice = input(f"\nSecim (1-{len(videos)}, 0=Tumu): ").strip()
        
        if choice == '0':
            # Hepsini isle ve kaydet
            for v in videos:
                video_path = os.path.join(DATA_DIR, v)
                name, ext = os.path.splitext(v)
                output_path = os.path.join(RESULT_DIR, f"{name}_detected.mp4")
                process_video(video_path, output_path)
                # Sonraki video icin state sifirla
                algo.cache = None
                algo.first_frame = 1
            print("\nTum videolar islendi!")
        elif choice.isdigit() and 1 <= int(choice) <= len(videos):
            idx = int(choice) - 1
            video_path = os.path.join(DATA_DIR, videos[idx])
            
            save = input("Sonucu kaydetmek ister misin? (e/h): ").strip().lower()
            output_path = None
            if save == 'e':
                name, ext = os.path.splitext(videos[idx])
                output_path = os.path.join(RESULT_DIR, f"{name}_detected.mp4")
            
            process_video(video_path, output_path)
        else:
            print("Gecersiz secim!")
    except KeyboardInterrupt:
        print("\nIptal edildi.")

if __name__ == "__main__":
    main()
