"""
Video Lane Detection - Tesla Style GUI
Simülasyon algoritmasının aynısını video dosyalarına uygular.
Kullanım: python VideoLaneDetect.py
"""

import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

# Script'in kendi dizinini sys.path'e ekle (algorithm.py'yi bulabilmek icin)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import algorithm as algo

# ===================== AYARLAR =====================
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
# ===================================================


class VideoLaneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tesla Style - Video Serit Tespit Sistemi")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        
        # Durum degiskenleri
        self.cap = None
        self.playing = False
        self.video_path = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.fake_speed = 80.0
        self.writer = None
        self.recording = False
        self.after_id = None
        
        self._build_ui()
        self._scan_data_folder()
    
    def _build_ui(self):
        """Arayuz olustur."""
        # ---- HEADER ----
        header = tk.Frame(self.root, bg="#16213e", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="⚡ TESLA STYLE", font=("Segoe UI", 18, "bold"),
                 fg="#00e5ff", bg="#16213e").pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="Video Şerit Tespit Sistemi", font=("Segoe UI", 12),
                 fg="#8899aa", bg="#16213e").pack(side=tk.LEFT, padx=5)
        
        # Hiz gostergesi (header sag)
        speed_frame = tk.Frame(header, bg="#16213e")
        speed_frame.pack(side=tk.RIGHT, padx=20)
        
        tk.Button(speed_frame, text="−", font=("Segoe UI", 14, "bold"), width=2,
                  fg="#00e5ff", bg="#0f3460", activebackground="#1a1a2e", activeforeground="#00e5ff",
                  relief=tk.FLAT, command=self._speed_down).pack(side=tk.LEFT, padx=2)
        
        self.speed_label = tk.Label(speed_frame, text="80 km/h", font=("Segoe UI", 13, "bold"),
                                     fg="#00e5ff", bg="#16213e", width=10)
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(speed_frame, text="+", font=("Segoe UI", 14, "bold"), width=2,
                  fg="#00e5ff", bg="#0f3460", activebackground="#1a1a2e", activeforeground="#00e5ff",
                  relief=tk.FLAT, command=self._speed_up).pack(side=tk.LEFT, padx=2)
        
        tk.Label(speed_frame, text="Anim. Hızı", font=("Segoe UI", 8),
                 fg="#556677", bg="#16213e").pack(side=tk.LEFT, padx=(5,0))
        
        # ---- ANA ICERIK ----
        main_area = tk.Frame(self.root, bg="#1a1a2e")
        main_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Sol panel - Video listesi
        left_panel = tk.Frame(main_area, bg="#16213e", width=220)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)
        
        tk.Label(left_panel, text="📁 Videolar", font=("Segoe UI", 12, "bold"),
                 fg="#e0e0e0", bg="#16213e").pack(pady=(10, 5), padx=10, anchor=tk.W)
        
        # Video listesi
        list_frame = tk.Frame(left_panel, bg="#16213e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
        
        self.video_listbox = tk.Listbox(list_frame, font=("Segoe UI", 10),
                                         bg="#0f3460", fg="#e0e0e0",
                                         selectbackground="#00e5ff", selectforeground="#1a1a2e",
                                         relief=tk.FLAT, highlightthickness=0, bd=0)
        self.video_listbox.pack(fill=tk.BOTH, expand=True)
        self.video_listbox.bind("<<ListboxSelect>>", self._on_video_select)
        
        # Dosya ac butonu
        tk.Button(left_panel, text="📂 Dosya Aç...", font=("Segoe UI", 10),
                  fg="#e0e0e0", bg="#0f3460", activebackground="#16213e", activeforeground="#00e5ff",
                  relief=tk.FLAT, cursor="hand2",
                  command=self._open_file).pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Sag panel - Video goruntusu
        right_panel = tk.Frame(main_area, bg="#0d1117")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Label(right_panel, bg="#0d1117",
                                      text="Video seçin veya dosya açın",
                                      font=("Segoe UI", 14), fg="#334455")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # ---- ALT KONTROLLER ----
        controls = tk.Frame(self.root, bg="#16213e", height=90)
        controls.pack(fill=tk.X, pady=(5, 0))
        controls.pack_propagate(False)
        
        # Progress bar
        progress_frame = tk.Frame(controls, bg="#16213e")
        progress_frame.pack(fill=tk.X, padx=15, pady=(8, 3))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Scale(progress_frame, from_=0, to=100,
                                   variable=self.progress_var, orient=tk.HORIZONTAL,
                                   command=self._on_seek)
        self.progress.pack(fill=tk.X)
        
        # Butonlar
        btn_frame = tk.Frame(controls, bg="#16213e")
        btn_frame.pack(pady=5)
        
        btn_style = {
            "font": ("Segoe UI", 12), "fg": "#e0e0e0", "bg": "#0f3460",
            "activebackground": "#1a1a2e", "activeforeground": "#00e5ff",
            "relief": tk.FLAT, "cursor": "hand2", "width": 4
        }
        
        self.btn_restart = tk.Button(btn_frame, text="⏮", command=self._restart, **btn_style)
        self.btn_restart.pack(side=tk.LEFT, padx=3)
        
        self.btn_play = tk.Button(btn_frame, text="▶", command=self._toggle_play, **btn_style)
        self.btn_play.pack(side=tk.LEFT, padx=3)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹", command=self._stop, **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        
        # Kaydet butonu
        self.btn_record = tk.Button(btn_frame, text="⏺ Kaydet", font=("Segoe UI", 11),
                                     fg="#e0e0e0", bg="#8b0000", activebackground="#cc0000",
                                     activeforeground="#ffffff", relief=tk.FLAT, cursor="hand2",
                                     width=8, command=self._toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=(15, 3))
        
        # Frame bilgisi
        self.frame_label = tk.Label(btn_frame, text="0 / 0", font=("Segoe UI", 10),
                                     fg="#556677", bg="#16213e")
        self.frame_label.pack(side=tk.LEFT, padx=15)
    
    def _scan_data_folder(self):
        """data/ klasorundeki videolari listele."""
        self.video_listbox.delete(0, tk.END)
        supported = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        if os.path.exists(DATA_DIR):
            for f in sorted(os.listdir(DATA_DIR)):
                if f.lower().endswith(supported):
                    self.video_listbox.insert(tk.END, f)
    
    def _on_video_select(self, event):
        """Listeden video secildiginde."""
        sel = self.video_listbox.curselection()
        if sel:
            filename = self.video_listbox.get(sel[0])
            self._load_video(os.path.join(DATA_DIR, filename))
    
    def _open_file(self):
        """Dosya secme diyalogu."""
        path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("Tümü", "*.*")]
        )
        if path:
            self._load_video(path)
    
    def _load_video(self, path):
        """Videoyu yukle."""
        self._stop()
        
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", f"Video açılamadı:\n{path}")
            return
        
        self.video_path = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        # Algorithm state sifirla
        algo.cache = None
        algo.first_frame = 1
        algo.frame_counter = 0
        algo.scroll_offset = 0.0
        algo.current_speed = self.fake_speed
        
        self.progress_var.set(0)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        
        # Ilk kareyi goster
        self._show_next_frame()
        
        self.root.title(f"Tesla Style - {os.path.basename(path)}")
    
    def _show_next_frame(self):
        """Sonraki kareyi oku, isle ve goster."""
        if not self.cap or not self.cap.isOpened():
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.btn_play.config(text="▶")
            return False
        
        self.current_frame += 1
        
        # Algoritmamizi uygula
        try:
            algo.current_speed = self.fake_speed
            processed = algo.process_image(frame)
        except Exception:
            processed = frame.copy()
        
        # Kayit yapiliyorsa yaz
        if self.recording and self.writer:
            self.writer.write(processed)
        
        # Tkinter icin goruntuyü cevir
        display = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Canvas boyutuna sigdir
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            h, w = display.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(display, (new_w, new_h))
        
        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.imgtk = imgtk
        self.video_canvas.config(image=imgtk, text="")
        
        # Progress guncelle
        if self.total_frames > 0:
            self.progress_var.set((self.current_frame / self.total_frames) * 100)
        self.frame_label.config(text=f"{self.current_frame} / {self.total_frames}")
        
        return True
    
    def _play_loop(self):
        """Oynatma dongusu."""
        if self.playing:
            success = self._show_next_frame()
            if success:
                delay = max(1, int(1000 / self.fps))
                self.after_id = self.root.after(delay, self._play_loop)
            else:
                self.playing = False
                self.btn_play.config(text="▶")
    
    def _toggle_play(self):
        """Oynat/Duraklat."""
        if not self.cap:
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.config(text="⏸")
            self._play_loop()
        else:
            self.btn_play.config(text="▶")
            if self.after_id:
                self.root.after_cancel(self.after_id)
    
    def _stop(self):
        """Durdur."""
        self.playing = False
        self.btn_play.config(text="▶")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.recording:
            self._toggle_record()
    
    def _restart(self):
        """Basa sar."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            algo.cache = None
            algo.first_frame = 1
            algo.frame_counter = 0
            algo.scroll_offset = 0.0
            self.progress_var.set(0)
            self.frame_label.config(text=f"0 / {self.total_frames}")
            self._show_next_frame()
    
    def _on_seek(self, val):
        """Progress bar ile ileri/geri sar."""
        if self.cap and not self.playing:
            target = int((float(val) / 100) * self.total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_frame = target
            algo.cache = None
            algo.first_frame = 1
            self._show_next_frame()
    
    def _toggle_record(self):
        """Kayit baslat/durdur."""
        if not self.recording:
            if not self.cap:
                return
            os.makedirs(RESULT_DIR, exist_ok=True)
            name = os.path.splitext(os.path.basename(self.video_path))[0]
            out_path = os.path.join(RESULT_DIR, f"{name}_detected.mp4")
            
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
            
            self.recording = True
            self.btn_record.config(text="⏹ Durdur", bg="#cc0000")
            print(f"Kayit basladi: {out_path}")
        else:
            self.recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
            self.btn_record.config(text="⏺ Kaydet", bg="#8b0000")
            print("Kayit durduruldu.")
    
    def _speed_up(self):
        self.fake_speed = min(300, self.fake_speed + 10)
        self.speed_label.config(text=f"{int(self.fake_speed)} km/h")
        algo.current_speed = self.fake_speed
    
    def _speed_down(self):
        self.fake_speed = max(0, self.fake_speed - 10)
        self.speed_label.config(text=f"{int(self.fake_speed)} km/h")
        algo.current_speed = self.fake_speed
    
    def on_close(self):
        """Pencere kapatilirken temizlik yap."""
        self._stop()
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoLaneApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
