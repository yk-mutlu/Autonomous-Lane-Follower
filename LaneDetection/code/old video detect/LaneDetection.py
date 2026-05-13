# Lane-Line Detection - Tek Dosya
# Developer - JustNikhill
# Once videoyu isle ve kaydet, sonra GUI'de yan yana goster

import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import warnings

# Polyfit RankWarning'i bastir (az noktayla fit yapildiginda ortaya cikar)
warnings.filterwarnings('ignore', message='.*Polyfit.*')


# =============================================
#  LANE DETECTION FONKSIYONLARI
# =============================================

# Son gecerli polinom katsayilari (kararlilik icin)
left_fit_buffer = []
right_fit_buffer = []
last_valid_left = None
last_valid_right = None
BUFFER_SIZE = 5          # Kac karelik tampon (yuksek = daha stabil)
MAX_JUMP = 80            # Piksel cinsinden maks. ani sicrama
SMOOTHING_ALPHA = 0.25   # Dusuk = daha yumusak/stabil gecis

# Her karedeki serit ortasi x-koordinati (video boyutu olceginde)
lane_center_per_frame = []
original_video_width = 0
original_video_height = 0


def reset_lane_state():
    """Serit tespit durumunu sifirlar."""
    global left_fit_buffer, right_fit_buffer
    global last_valid_left, last_valid_right
    global lane_center_per_frame
    left_fit_buffer = []
    right_fit_buffer = []
    last_valid_left = None
    last_valid_right = None
    lane_center_per_frame = []


def region_of_interest(img, vertices):
    """Ilgi alani maskesi uygular."""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2]
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def average_fit(fit_buffer):
    """Tampondaki polinom katsayilarinin ortalamasini alir."""
    if len(fit_buffer) == 0:
        return None
    return np.mean(fit_buffer, axis=0)


def clamp_fit(new_fit, old_fit, max_jump, y_bottom, y_top):
    """Ani sicramalari sinirlar - eski fit'e gore cok farkli ise kisitla."""
    if old_fit is None:
        return new_fit
    # Alt ve ust noktalardaki x farki kontrol
    new_x_bot = np.polyval(new_fit, y_bottom)
    old_x_bot = np.polyval(old_fit, y_bottom)
    new_x_top = np.polyval(new_fit, y_top)
    old_x_top = np.polyval(old_fit, y_top)

    jump_bot = abs(new_x_bot - old_x_bot)
    jump_top = abs(new_x_top - old_x_top)

    if jump_bot > max_jump or jump_top > max_jump:
        # Cok buyuk sicrama - eski degerle karistir
        blended = old_fit * (1 - SMOOTHING_ALPHA) + new_fit * SMOOTHING_ALPHA
        return blended
    return new_fit


def compute_lane_fits(lines, img_shape):
    """Hough cizgilerinden sol ve sag serit polinomlari hesaplar."""
    if lines is None or len(lines) == 0:
        return None, None

    left_x, left_y = [], []
    right_x, right_y = [], []
    mid_x = img_shape[1] / 2

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Egim filtresi: cok dik (korkuluk/bariyer) ve cok yatay cizgileri at
            if abs(slope) < 0.4 or abs(slope) > 2.5:
                continue
            # Egim + konum kontrolu (sol tarafta negatif egim, sag tarafta pozitif)
            center_x = (x1 + x2) / 2
            if slope < 0 and center_x < mid_x * 1.1:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif slope > 0 and center_x > mid_x * 0.9:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    left_fit = None
    right_fit = None

    try:
        if len(left_x) >= 10:
            left_fit = np.polyfit(left_y, left_x, 2)  # 2. derece = egri (en az 10 nokta)
        elif len(left_x) >= 4:
            fit1 = np.polyfit(left_y, left_x, 1)
            left_fit = np.array([0.0, fit1[0], fit1[1]])  # 1. dereceyi 2. dereceye cevir
    except (np.linalg.LinAlgError, ValueError):
        pass

    try:
        if len(right_x) >= 10:
            right_fit = np.polyfit(right_y, right_x, 2)
        elif len(right_x) >= 4:
            fit1 = np.polyfit(right_y, right_x, 1)
            right_fit = np.array([0.0, fit1[0], fit1[1]])
    except (np.linalg.LinAlgError, ValueError):
        pass

    return left_fit, right_fit


def draw_lane(img, left_fit, right_fit, y_bottom, y_top):
    """Serit cizgilerini, yesil dolguyu ve kirmizi orta cizgiyi cizer."""
    # Y noktalari olustur
    num_points = 50
    plot_y = np.linspace(y_top, y_bottom, num_points).astype(int)

    # Sol ve sag x koordinatlari
    left_x = np.polyval(left_fit, plot_y).astype(int)
    right_x = np.polyval(right_fit, plot_y).astype(int)

    # --- Yesil dolgu (seritler arasi) ---
    pts_left = np.column_stack((left_x, plot_y))
    pts_right = np.column_stack((right_x, plot_y))[::-1]
    fill_pts = np.vstack((pts_left, pts_right))
    overlay = img.copy()
    cv2.fillPoly(overlay, [fill_pts], (0, 180, 0))
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # --- Serit cizgileri (mavi, kalin, egri) ---
    left_pts = np.column_stack((left_x, plot_y)).reshape((-1, 1, 2))
    right_pts = np.column_stack((right_x, plot_y)).reshape((-1, 1, 2))
    cv2.polylines(img, [left_pts], False, (255, 100, 0), 5, cv2.LINE_AA)
    cv2.polylines(img, [right_pts], False, (255, 100, 0), 5, cv2.LINE_AA)

    # --- Kirmizi orta cizgi (her noktada ayri gosterilen serit ortasi) ---
    center_x = ((left_x + right_x) / 2).astype(int)
    center_pts = np.column_stack((center_x, plot_y)).reshape((-1, 1, 2))
    # Egri kirmizi cizgi
    cv2.polylines(img, [center_pts], False, (0, 0, 255), 4, cv2.LINE_AA)
    # Her noktada kucuk daire
    for i in range(0, num_points, 2):
        cv2.circle(img, (int(center_x[i]), int(plot_y[i])), 4, (0, 0, 255), -1, cv2.LINE_AA)


def process_image(image):
    """Ana serit tespit fonksiyonu - tek kare isler."""
    global left_fit_buffer, right_fit_buffer
    global last_valid_left, last_valid_right
    global lane_center_per_frame

    h, w = image.shape[:2]
    y_bottom = h
    y_top = int(h * 0.6)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Sari ve beyaz renk maskeleri
    # Beyaz esigi yukselttik (korkuluk/bariyer gibi kirli beyazlari eliyor)
    lower_yellow = np.array([15, 60, 60], dtype="uint8")
    upper_yellow = np.array([40, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 210, 255)  # 170->210: korkulugu elemek icin
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    # Gurultu azaltma ve kenar tespiti
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (7, 7), 0)  # Daha guclu blur
    canny_edges = cv2.Canny(gauss_gray, 60, 160)

    # Ilgi alani (trapez bolge)
    # Sol alt koseyi iceri cektik: korkulugu ROI disinda birakiyor
    vertices = np.array([[
        (int(w * 0.15), h),              # alt-sol (korkuluktan uzak)
        (int(w * 0.42), int(h * 0.60)), # ust-sol
        (int(w * 0.58), int(h * 0.60)), # ust-sag
        (int(w * 0.92), h)              # alt-sag
    ]], dtype=np.int32)
    roi_image = region_of_interest(canny_edges, vertices)

    # Hough cizgi tespiti - minLineLength arttirildi (kesik cizgi gurultusunu azaltir)
    theta = np.pi / 180
    lines = cv2.HoughLinesP(roi_image, 1, theta, 20, np.array([]),
                            minLineLength=40, maxLineGap=30)

    # Polinom hesapla
    left_fit, right_fit = compute_lane_fits(lines, image.shape)

    # Ani sicrama kontrolu
    if left_fit is not None:
        left_fit = clamp_fit(left_fit, last_valid_left, MAX_JUMP, y_bottom, y_top)
    if right_fit is not None:
        right_fit = clamp_fit(right_fit, last_valid_right, MAX_JUMP, y_bottom, y_top)

    # Tampona ekle
    if left_fit is not None:
        left_fit_buffer.append(left_fit)
        if len(left_fit_buffer) > BUFFER_SIZE:
            left_fit_buffer.pop(0)
    if right_fit is not None:
        right_fit_buffer.append(right_fit)
        if len(right_fit_buffer) > BUFFER_SIZE:
            right_fit_buffer.pop(0)

    # Ortalama al (kararlilik)
    avg_left = average_fit(left_fit_buffer)
    avg_right = average_fit(right_fit_buffer)

    # Tespit yoksa son gecerli degeri kullan
    if avg_left is None:
        avg_left = last_valid_left
    else:
        last_valid_left = avg_left.copy()

    if avg_right is None:
        avg_right = last_valid_right
    else:
        last_valid_right = avg_right.copy()

    # Serit ortasi x-koordinatini kaydet (alt kenardaki orta nokta)
    center_x_bottom = -1  # -1 = tespit yok
    if avg_left is not None and avg_right is not None:
        lx = np.polyval(avg_left, y_bottom)
        rx = np.polyval(avg_right, y_bottom)
        center_x_bottom = (lx + rx) / 2.0
    lane_center_per_frame.append(center_x_bottom)

    # Cizim
    result = image.copy()
    if avg_left is not None and avg_right is not None:
        draw_lane(result, avg_left, avg_right, y_bottom, y_top)

    return result


# =============================================
#  VIDEO ISLEME (moviepy yerine cv2 ile)
# =============================================

def process_video(input_path, output_path):
    """Videoyu kare kare isleyip output dosyasina kaydeder."""
    global original_video_width, original_video_height
    reset_lane_state()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_video_width = width
    original_video_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_image(frame)
        out.write(processed)
        frame_count += 1
        # Progress bilgisi
        if frame_count % 30 == 0:
            pct = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            progress_label.config(text=f"Isleniyor... %{pct} ({frame_count}/{total_frames})")
            root.update_idletasks()

    cap.release()
    out.release()
    return True


# =============================================
#  GUI BOLUMU
# =============================================

cap_input = None
cap_output = None
frame_index = 0  # Islenmis video kare indeksi

# Araba referans noktasi (Canvas uzerinde, suruklenebilir)
DISPLAY_W = 600
DISPLAY_H = 500
car_point_x = DISPLAY_W // 2  # Baslangicta ortada
car_point_y = DISPLAY_H - 30  # Alt kisimda
dragging = False


def show_vid1(cap):
    """Orijinal videoyu gosterir (loop)."""
    if cap is None or not cap.isOpened():
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(30, show_vid1, cap)


def show_vid2(cap):
    """Islenmis videoyu Canvas uzerinde gosterir + araba noktasi ve yon (loop)."""
    global frame_index
    if cap is None or not cap.isOpened():
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        ret, frame = cap.read()
        if not ret or frame is None:
            return
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)

    # Onceki karenin tum objelerini sil (performans icin kritik!)
    canvas.delete('all')

    # Canvas uzerine video karesi
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=NW, image=imgtk)

    # Serit ortasi bu karedeki x-koordinati (display olcegine cevir)
    if frame_index < len(lane_center_per_frame) and original_video_width > 0:
        center_x_orig = lane_center_per_frame[frame_index]
        if center_x_orig >= 0:
            # Orijinal video boyutundan display boyutuna olcekle
            scale = DISPLAY_W / original_video_width
            lane_cx = int(center_x_orig * scale)

            # Serit ortasi isaretcisi (kucuk yesil daire)
            canvas.create_oval(lane_cx - 6, car_point_y - 6,
                              lane_cx + 6, car_point_y + 6,
                              fill='#2ecc71', outline='white', width=2, tags="overlay")

            # Araba noktasindan serit ortasina cizgi
            canvas.create_line(car_point_x, car_point_y,
                              lane_cx, car_point_y,
                              fill='yellow', width=2, dash=(4, 4), tags="overlay")

            # Yon hesapla
            offset = car_point_x - lane_cx  # pozitif = araba sagda, sola donmeli
            threshold = 15  # piksel esik degeri

            if offset > threshold:
                direction = "<< SOLA DON"
                dir_color = '#e74c3c'
            elif offset < -threshold:
                direction = "SAGA DON >>"
                dir_color = '#3498db'
            else:
                direction = "DUZ GIT"
                dir_color = '#2ecc71'

            # Yon yazisi
            canvas.create_text(DISPLAY_W // 2, 30,
                              text=direction, font=('Arial', 22, 'bold'),
                              fill=dir_color, tags="overlay")

            # Offset bilgisi
            canvas.create_text(DISPLAY_W // 2, 60,
                              text=f"Sapma: {offset:+d} px",
                              font=('Arial', 12), fill='white', tags="overlay")

    # Araba referans noktasi (turuncu, suruklenebilir)
    r = 10
    canvas.create_oval(car_point_x - r, car_point_y - r,
                      car_point_x + r, car_point_y + r,
                      fill='#f39c12', outline='white', width=3, tags="carpoint")
    canvas.create_text(car_point_x, car_point_y + 20,
                      text="ARABA", font=('Arial', 8, 'bold'),
                      fill='#f39c12', tags="overlay")

    frame_index += 1
    canvas.after(30, show_vid2, cap)


def on_canvas_press(event):
    """Canvas uzerinde tiklandiginda araba noktasina yakin mi kontrol et."""
    global dragging
    dist = ((event.x - car_point_x)**2 + (event.y - car_point_y)**2) ** 0.5
    if dist < 30:  # 30 piksel yakinlik
        dragging = True


def on_canvas_drag(event):
    """Suruklerken araba noktasini guncelle."""
    global car_point_x, car_point_y
    if dragging:
        # x sinirla (0 - DISPLAY_W arasi)
        car_point_x = max(10, min(DISPLAY_W - 10, event.x))
        # y de suruklenebilsin ama alt %30 bolge ile sinirla
        car_point_y = max(int(DISPLAY_H * 0.7), min(DISPLAY_H - 10, event.y))


def on_canvas_release(event):
    """Surukleme bitti."""
    global dragging
    dragging = False


def start_pipeline(video_path):
    """Videoyu isle, kaydet, sonra her ikisini yan yana oynat."""
    global cap_input, cap_output, frame_index

    # Onceki videolari kapat
    if cap_input is not None:
        cap_input.release()
    if cap_output is not None:
        cap_output.release()

    if not os.path.exists(video_path):
        messagebox.showerror("Hata", f"Video dosyasi bulunamadi:\n{video_path}")
        return

    # Output dosya yolu
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "..", "..", "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    output_path = os.path.join(result_dir, name + "_output" + ext)

    # Butonlari devre disi birak
    for btn in buttons:
        btn.config(state=DISABLED)

    progress_label.config(text="Video isleniyor, lutfen bekleyin...")
    root.update_idletasks()

    # Videoyu isle ve kaydet
    success = process_video(video_path, output_path)

    # Butonlari tekrar aktif yap
    for btn in buttons:
        btn.config(state=NORMAL)

    if not success:
        progress_label.config(text="Video islenemedi!")
        messagebox.showerror("Hata", "Video islenirken bir hata olustu.")
        return

    progress_label.config(text="Tamamlandi! Araba noktasini surukle, yonu gor.")

    # Her iki videoyu ac ve yan yana oynat
    cap_input = cv2.VideoCapture(video_path)
    cap_output = cv2.VideoCapture(output_path)
    frame_index = 0

    if not cap_input.isOpened() or not cap_output.isOpened():
        messagebox.showerror("Hata", "Videolar acilamadi.")
        return

    show_vid1(cap_input)
    show_vid2(cap_output)


def open_file():
    """Dosya secme dialogu acar."""
    filepath = filedialog.askopenfilename(
        title="Video Dosyasi Sec",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
    )
    if filepath:
        start_pipeline(filepath)


def use_default_video():
    """data/ klasorundeki varsayilan videoyu kullanir."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # data/ klasoru bir ust dizinde
    data_dir = os.path.join(script_dir, "..", "..", "data")
    for name in ["straight_lane.mp4", "curved_lane.mp4"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            start_pipeline(path)
            return
    messagebox.showwarning("Uyari", "data/ klasorunde video bulunamadi.\nLutfen bir video dosyasi secin.")
    open_file()


def on_closing():
    """Uygulama kapanirken kaynaklari serbest birakir."""
    global cap_input, cap_output
    if cap_input is not None:
        cap_input.release()
    if cap_output is not None:
        cap_output.release()
    root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Lane-Line Detection")
    root.geometry("1280x780+50+10")
    root.configure(bg='#2c3e50')
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Baslik
    heading = Label(root, text="Lane-Line Detection", pady=15,
                    font=('Arial', 36, 'bold'), bg='#2c3e50', fg='#ecf0f1')
    heading.pack()

    # Buton cercevesi
    btn_frame = Frame(root, bg='#2c3e50')
    btn_frame.pack(pady=10)
    buttons = []

    btn1 = Button(btn_frame, text='Varsayilan Video', font=('Arial', 12),
                  fg='white', bg='#27ae60', activebackground='#2ecc71',
                  padx=20, pady=5, command=use_default_video)
    btn1.pack(side=LEFT, padx=10)
    buttons.append(btn1)

    btn2 = Button(btn_frame, text='Video Sec', font=('Arial', 12),
                  fg='white', bg='#2980b9', activebackground='#3498db',
                  padx=20, pady=5, command=open_file)
    btn2.pack(side=LEFT, padx=10)
    buttons.append(btn2)

    btn3 = Button(btn_frame, text='Cikis', font=('Arial', 12),
                  fg='white', bg='#c0392b', activebackground='#e74c3c',
                  padx=20, pady=5, command=on_closing)
    btn3.pack(side=LEFT, padx=10)
    buttons.append(btn3)

    # Progress label
    progress_label = Label(root, text="Bir video secin veya varsayilan videoyu baslatin.",
                           font=('Arial', 11), bg='#2c3e50', fg='#f39c12')
    progress_label.pack(pady=5)

    # Etiketler
    labels_frame = Frame(root, bg='#2c3e50')
    labels_frame.pack(pady=5)
    Label(labels_frame, text="Orijinal Video", font=('Arial', 14, 'bold'),
          bg='#2c3e50', fg='#bdc3c7').pack(side=LEFT, padx=160)
    Label(labels_frame, text="Serit Tespiti", font=('Arial', 14, 'bold'),
          bg='#2c3e50', fg='#bdc3c7').pack(side=RIGHT, padx=160)

    # Video gosterim alanlari
    video_frame = Frame(root, bg='#2c3e50')
    video_frame.pack(pady=5, expand=True, fill=BOTH)

    lmain = tk.Label(master=video_frame, bg='#34495e', width=DISPLAY_W, height=DISPLAY_H)
    lmain.pack(side=LEFT, padx=10, expand=True)

    # Islenmis video icin Canvas (suruklenebilir araba noktasi icin)
    canvas = tk.Canvas(master=video_frame, bg='#34495e',
                       width=DISPLAY_W, height=DISPLAY_H,
                       highlightthickness=0)
    canvas.pack(side=RIGHT, padx=10, expand=True)

    # Canvas surukle olaylari
    canvas.bind("<ButtonPress-1>", on_canvas_press)
    canvas.bind("<B1-Motion>", on_canvas_drag)
    canvas.bind("<ButtonRelease-1>", on_canvas_release)

    root.mainloop()
