# Developer - JustNikhill

#Before detecting lane lines, we masked remaining objects and then identified the line with Hough transformation.

import numpy as np
import cv2

# Global variables for smoothing
cache = None
first_frame = 1
frame_counter = 0
scroll_offset = 0.0  # Akan cizgiler icin yumusak kayma degeri
current_speed = 0.0  # Aracin anlik hizi (km/h) - UnityDrivePID tarafindan guncellenir
lost_frames = 0      # Serit kayboldugunda sayan sayac
wide_hough_lines = None  # Genis ROI Hough sonuclari (yan serit algilama icin)
adj_cache = None     # Yan serit smoothing cache [outer_x_bottom, outer_x_top, side]
adj_lost = 0         # Yan serit kayboldu sayaci
adj_velocity = np.array([0.0, 0.0])  # PD turev terimi icin (son hiz)
lane_vote_history = []   # Son N karenin serit oylari
LANE_VOTE_WINDOW = 30   # Oylama penceresi (30 kare ~ 1 saniye)
confirmed_lane = "TEK SERIT"
ADJ_KP = 0.15   # Oransal katsayi
ADJ_KD = 0.3    # Turev katsayisi


# Frame masking and region of interest
def interested_region(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        mask_color_ignore = (255,) * img.shape[2]
    else:
        mask_color_ignore = 255

    cv2.fillPoly(mask, vertices, mask_color_ignore)
    return cv2.bitwise_and(img, mask)

def lines_drawn(img, lines, color=[0, 0, 255], thickness=4):
    global cache
    global first_frame
    slope_l, slope_r = [],[]
    lane_l,lane_r = [],[]

    α = 0.2

    if lines is None:
        print('no lines detected')
        return

    min_y = img.shape[0]

    all_lefts = []
    all_rights = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                continue
            slope = (y2-y1)/(x2-x1)
            
            # Alt noktadaki (y=img.shape[0]) tahmini x koordinatini hesapla
            x_bottom = ((img.shape[0] - y1) / slope) + x1
            
            if slope > 0.45:
                all_rights.append((x_bottom, slope, line))
            elif slope < -0.45:
                all_lefts.append((x_bottom, slope, line))
                
        min_y = min(y1, y2, min_y)

    center_x = img.shape[1] / 2
    
    # Eger cache varsa hedefleri cache'den al, yoksa arabanin merkezine en yakin cizgiyi referans al
    target_l = None
    target_r = None
    
    if cache is not None:
        target_l = cache[0]
        target_r = cache[4]
    else:
        # Cache yoksa (ilk kare veya serit kaybedilmis), ekranin ortasina en yakin olan cizgileri bul
        if all_lefts:
            target_l = min(all_lefts, key=lambda item: abs(item[0] - center_x))[0]
        if all_rights:
            target_r = min(all_rights, key=lambda item: abs(item[0] - center_x))[0]

    # Simdi sadece referans hedefe yakin olanlari gercek serit olarak kabul et
    # 1. Kontinuite (Işınlanma) Kontrolü: Önceki konumdan max 60 piksel sapabilir
    # 2. Genişlik Kontrolü: Şerit aşırı büyürse (yan şeridi kaparsa) engelle
    
    JUMP_LIMIT = 60 # Piksel
    for x_bottom, slope, line in all_lefts:
        if target_l is not None:
            if abs(x_bottom - target_l) < JUMP_LIMIT:
                slope_l.append(slope)
                lane_l.append(line)
        else: # Cache yoksa merkeze yakınlık kullan (ilk kare)
            if abs(x_bottom - center_x) < 250:
                slope_l.append(slope)
                lane_l.append(line)
            
    for x_bottom, slope, line in all_rights:
        if target_r is not None:
            if abs(x_bottom - target_r) < JUMP_LIMIT:
                slope_r.append(slope)
                lane_r.append(line)
        else: # Cache yoksa merkeze yakınlık kullan (ilk kare)
            if abs(x_bottom - center_x) < 250:
                slope_r.append(slope)
                lane_r.append(line)

    # --- YAN SERIT VERILERINI TOPLA (GENIS ROI'den, ana takipten bagimsiz) ---
    # Genis ROI'den gelen cizgileri isle (process_image'de ayri hesaplaniyor)
    adj_left_lines = []
    adj_right_lines = []
    
    if wide_hough_lines is not None:
        for wline in wide_hough_lines:
            for wx1, wy1, wx2, wy2 in wline:
                if wx2 == wx1:
                    continue
                wslope = (wy2 - wy1) / (wx2 - wx1)
                if abs(wslope) < 0.3:  # Cok yatay cizgileri atla
                    continue
                wx_bottom = ((img.shape[0] - wy1) / wslope) + wx1
                
                # Sol taraf: bizim sol kenarimizdan belirgin sekilde solda
                if target_l is not None and wx_bottom < (target_l - 30):
                    adj_left_lines.append((wx_bottom, wslope, wline))
                # Sag taraf: bizim sag kenarimizdan belirgin sekilde sagda
                elif target_r is not None and wx_bottom > (target_r + 30):
                    adj_right_lines.append((wx_bottom, wslope, wline))
    
    # Debug
    if adj_left_lines or adj_right_lines:
        tl = f"{target_l:.0f}" if target_l else "?"
        tr = f"{target_r:.0f}" if target_r else "?"
        print(f"Yan serit (GENIS ROI): Sol={len(adj_left_lines)} Sag={len(adj_right_lines)} | target_l={tl} target_r={tr}")

    # --- TESLA STYLE: Gorulebilen butun kucuk serit parcalarini ayri ayri ciz (Cyan renginde) ---
    for line in lane_l:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [255, 200, 0], 2)
    for line in lane_r:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [255, 200, 0], 2)

    # Eger hic serit bulunamazsa iptal et
    if len(lane_l) == 0 and len(lane_r) == 0:
        global lost_frames
        lost_frames += 1
        print(f'no lane detected ({lost_frames})')
        
        # 10 kare boyunca bulamazsan pes et ve cache'i sil (yeniden tarama baslasin)
        if lost_frames > 10:
            print("--- SERIT TAMAMEN KAYBOLDU, SIFIRLANIYOR ---")
            cache = None
            first_frame = 1
            lost_frames = 0
            
        return 1

    # Serit bulunduysa sayaci sifirla
    lost_frames = 0

    # Eger sadece SOL serit varsa, sag seridi tahmini olustur
    if len(lane_l) > 0 and len(lane_r) == 0:
        slope_mean_l = np.mean(slope_l, axis=0)
        mean_l = np.mean(np.array(lane_l), axis=0)
        slope_mean_r = -slope_mean_l  # Simetrik egim
        # Sag seridin noktalarini sol seritten 400 piksel saga kaydir
        mean_r = np.array([[mean_l[0][0] + 400, mean_l[0][1], mean_l[0][2] + 400, mean_l[0][3]]])
        
    # Eger sadece SAG serit varsa, sol seridi tahmini olustur
    elif len(lane_r) > 0 and len(lane_l) == 0:
        slope_mean_r = np.mean(slope_r, axis=0)
        mean_r = np.mean(np.array(lane_r), axis=0)
        slope_mean_l = -slope_mean_r  # Simetrik egim
        # Sol seridin noktalarini sag seritten 400 piksel sola kaydir
        mean_l = np.array([[mean_r[0][0] - 400, mean_r[0][1], mean_r[0][2] - 400, mean_r[0][3]]])
        
    # Iki serit de varsa normal sekilde hesapla
    else:
        slope_mean_l = np.mean(slope_l, axis=0)
        slope_mean_r = np.mean(slope_r, axis=0)
        mean_l = np.mean(np.array(lane_l), axis=0)
        mean_r = np.mean(np.array(lane_r), axis=0)

    if slope_mean_r == 0 or slope_mean_l == 0:
        print('dividing by zero')
        return 1

    x1_l = int(((img.shape[0] - mean_l[0][1]) / slope_mean_l) + mean_l[0][0])
    x2_l = int(((min_y - mean_l[0][1]) / slope_mean_l) + mean_l[0][0])
    x1_r = int(((img.shape[0] - mean_r[0][1]) / slope_mean_r) + mean_r[0][0])
    x2_r = int(((min_y - mean_r[0][1]) / slope_mean_r) + mean_r[0][0])

    if x1_l > x1_r:
        x1_l = int((x1_l+x1_r)/2)
        x1_r = x1_l
        y1_l = int((slope_mean_l * x1_l ) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
        y1_r = int((slope_mean_r * x1_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
        y2_l = int((slope_mean_l * x2_l ) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
        y2_r = int((slope_mean_r * x2_r ) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
    else:
        y1_l = img.shape[0]
        y2_l = min_y
        y1_r = img.shape[0]
        y2_r = min_y

    # --- ASIRI GENISLIK KONTROLU ---
    # Eger serit tabanda 650 pikselden genis ciktiysa bir hata vardir (yan seridi yutmustur)
    if abs(x1_r - x1_l) > 650 and cache is not None:
        # Eski veriyi koru, cok ani genislemeyi reddet
        x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r = cache
    
    present_frame = np.array([x1_l,y1_l,x2_l,y2_l,x1_r,y1_r,x2_r,y2_r],dtype ="float32")

    if first_frame == 1 or cache is None:
        next_frame = present_frame        
        first_frame = 0        
    else :
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*present_frame

    # --- TESLA STYLE: Surulebilir Alani (Drivable Space) Yesile Boya ---
    pts = np.array([[
        [int(next_frame[0]), int(next_frame[1])], # Sol Alt
        [int(next_frame[2]), int(next_frame[3])], # Sol Ust
        [int(next_frame[6]), int(next_frame[7])], # Sag Ust
        [int(next_frame[4]), int(next_frame[5])]  # Sag Alt
    ]], dtype=np.int32)
    
    # Yesil transparan dolgu
    cv2.fillPoly(img, pts, [0, 100, 0])

    # Kenarlari kirmizi ile belirginlestir
    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)

    # --- YAN SERIT ALGILAMA VE CIZIMI (Ana takipten bagimsiz) ---
    global adj_cache
    global adj_lost
    lane_status = "TEK SERIT"
    side_text = ""
    
    adj_found = False
    our_width = abs(int(next_frame[4]) - int(next_frame[0]))
    min_adj_width = max(100, our_width * 0.2)  # En az 100px veya kendi genisligimizin %20'si
    
    # Solda yan serit var mi?
    if len(adj_left_lines) >= 1 and not adj_found:
        avg_x = np.mean([x for x, s, l in adj_left_lines])
        avg_slope = np.mean([s for x, s, l in adj_left_lines])
        inner_x = int(next_frame[0])
        inner_top_x = int(next_frame[2])
        
        # CAKISMA KONTROLU: dis kenar kendi seridimizin icinde olamaz
        if target_r is not None and avg_x > target_l:
            avg_x = target_l  # En fazla sol kenarimiza kadar
        
        adj_width = inner_x - avg_x
        if adj_width > min_adj_width and avg_slope != 0:
            outer_top_x = int(avg_x + (min_y - img.shape[0]) / avg_slope)
            
            new_target = np.array([avg_x, outer_top_x], dtype="float32")
            if adj_cache is not None and adj_cache[2] == -1:
                error = new_target - adj_cache[:2]
                adj_velocity[:] = ADJ_KD * adj_velocity + ADJ_KP * error
                adj_cache[:2] += adj_velocity
            else:
                adj_cache = np.array([avg_x, outer_top_x, -1], dtype="float32")
                adj_velocity[:] = 0
            adj_lost = 0
            adj_found = True
            
            ox = int(adj_cache[0])
            otx = int(adj_cache[1])
            
            adj_pts = np.array([[[ox, img.shape[0]], [otx, int(min_y)],
                [inner_top_x, int(min_y)], [inner_x, img.shape[0]]]], dtype=np.int32)
            cv2.fillPoly(img, adj_pts, [100, 50, 0])
            cv2.line(img, (ox, img.shape[0]), (otx, int(min_y)), [255, 100, 0], 3)
            
            lane_status = "SAG SERIT"
            side_text = "SOL"
    
    # Sagda yan serit var mi?
    if len(adj_right_lines) >= 1 and not adj_found:
        avg_x = np.mean([x for x, s, l in adj_right_lines])
        avg_slope = np.mean([s for x, s, l in adj_right_lines])
        inner_x = int(next_frame[4])
        inner_top_x = int(next_frame[6])
        
        # CAKISMA KONTROLU: dis kenar kendi seridimizin icinde olamaz
        if target_l is not None and avg_x < target_r:
            avg_x = target_r  # En fazla sag kenarimiza kadar
        
        adj_width = avg_x - inner_x
        if adj_width > min_adj_width and avg_slope != 0:
            outer_top_x = int(avg_x + (min_y - img.shape[0]) / avg_slope)
            
            new_target = np.array([avg_x, outer_top_x], dtype="float32")
            if adj_cache is not None and adj_cache[2] == 1:
                error = new_target - adj_cache[:2]
                adj_velocity[:] = ADJ_KD * adj_velocity + ADJ_KP * error
                adj_cache[:2] += adj_velocity
            else:
                adj_cache = np.array([avg_x, outer_top_x, 1], dtype="float32")
                adj_velocity[:] = 0
            adj_lost = 0
            adj_found = True
            
            ox = int(adj_cache[0])
            otx = int(adj_cache[1])
            
            adj_pts = np.array([[[inner_x, img.shape[0]], [inner_top_x, int(min_y)],
                [otx, int(min_y)], [ox, img.shape[0]]]], dtype=np.int32)
            cv2.fillPoly(img, adj_pts, [100, 50, 0])
            cv2.line(img, (ox, img.shape[0]), (otx, int(min_y)), [255, 100, 0], 3)
            
            lane_status = "SOL SERIT"
            side_text = "SAG"
    
    # Yan serit bulunamadiysa eski cache'den cizmeye devam et
    if not adj_found:
        adj_lost += 1
        if adj_lost > 30:
            adj_cache = None
        elif adj_cache is not None:
            ox = int(adj_cache[0])
            otx = int(adj_cache[1])
            side = adj_cache[2]
            
            if side == -1:
                inner_x = int(next_frame[0])
                inner_top_x = int(next_frame[2])
                adj_pts = np.array([[[ox, img.shape[0]], [otx, int(min_y)],
                    [inner_top_x, int(min_y)], [inner_x, img.shape[0]]]], dtype=np.int32)
                cv2.fillPoly(img, adj_pts, [100, 50, 0])
                cv2.line(img, (ox, img.shape[0]), (otx, int(min_y)), [255, 100, 0], 3)
                lane_status = "SAG SERIT"
                side_text = "SOL"
            elif side == 1:
                inner_x = int(next_frame[4])
                inner_top_x = int(next_frame[6])
                adj_pts = np.array([[[inner_x, img.shape[0]], [inner_top_x, int(min_y)],
                    [otx, int(min_y)], [ox, img.shape[0]]]], dtype=np.int32)
                cv2.fillPoly(img, adj_pts, [100, 50, 0])
                cv2.line(img, (ox, img.shape[0]), (otx, int(min_y)), [255, 100, 0], 3)
                lane_status = "SOL SERIT"
                side_text = "SAG"

    # --- HAREKETLI YATAY CIZGILER (Tahmin Matrisi) ---
    global frame_counter
    global scroll_offset
    frame_counter += 1
    
    y_bottom = img.shape[0]
    y_top = int(min_y)
    lane_height = y_bottom - y_top
    num_lines = 10
    # Her karede kucuk bir miktar ekle (yumusak gecis)
    scroll_offset += current_speed * 0.04
    if lane_height > 0:
        scroll_offset = scroll_offset % lane_height
    
    offset = scroll_offset
    
    colors = [
        (0, 0, 255),    # Kirmizi
        (0, 127, 255),  # Turuncu
        (0, 255, 255),  # Sari
        (0, 255, 127),  # Acik Yesil
        (0, 255, 0),    # Yesil
        (127, 255, 0),  # Limon Yesili
        (255, 255, 0),  # Cam Gobegi
        (255, 127, 0),  # Acik Mavi
        (255, 0, 0),    # Mavi
        (255, 0, 127)   # Mor
    ]
    
    dx_l = next_frame[2] - next_frame[0]
    dy_l = next_frame[3] - next_frame[1]
    dx_r = next_frame[6] - next_frame[4]
    dy_r = next_frame[7] - next_frame[5]
    
    bottom_most_y = -1
    active_color = (0, 255, 255)
    
    for i in range(num_lines):
        base_y = y_top + i * (lane_height / num_lines)
        y = y_top + (base_y - y_top + offset) % lane_height
        
        # Y noktasina karsilik gelen X noktalari
        if dy_l != 0:
            x_l = next_frame[0] + (y - next_frame[1]) * (dx_l / dy_l)
        else:
            x_l = next_frame[0]
            
        if dy_r != 0:
            x_r = next_frame[4] + (y - next_frame[5]) * (dx_r / dy_r)
        else:
            x_r = next_frame[4]
            
        cv2.line(img, (int(x_l), int(y)), (int(x_r), int(y)), colors[i], 2)
        
        if y > bottom_most_y:
            bottom_most_y = y
            active_color = colors[i]

    # --- TESLA STYLE: Aracin Tahmini Gidis Yonu (Merkez Cizgisi) ---
    center_bottom_x = int((next_frame[0] + next_frame[4]) / 2)
    center_top_x = int((next_frame[2] + next_frame[6]) / 2)
    cv2.line(img, (center_bottom_x, img.shape[0]), (center_top_x, int(min_y)), active_color, 4)
    
    # --- SERIT BILGISI YAZISI (Cogunluk Oyu ile) ---
    global confirmed_lane
    global lane_vote_history
    
    # Mevcut karenin oyunu ekle
    lane_vote_history.append(lane_status)
    if len(lane_vote_history) > LANE_VOTE_WINDOW:
        lane_vote_history.pop(0)
    
    # Son 30 karede en cok hangisi oy aldi?
    from collections import Counter
    vote_counts = Counter(lane_vote_history)
    winner = vote_counts.most_common(1)[0]
    # Kazanan en az %60 oy almissa gecerli say
    if winner[1] >= len(lane_vote_history) * 0.6:
        confirmed_lane = winner[0]
    
    # Ekranda her zaman ONAYLANMIS durumu goster
    display_status = confirmed_lane
    cv2.rectangle(img, (5, 55), (280, 95), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 55), (280, 95), (0, 255, 255), 2)
    cv2.putText(img, f"DRIVE: {display_status}", (20, 83), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- YAN SERIT ORTASINA YAZI ---
    # Mavi koridorun tam ortasini hesapla (adj_cache varsa)
    if side_text and adj_cache is not None:
        ox = int(adj_cache[0])
        otx = int(adj_cache[1])
        side = adj_cache[2]
        # Yazinin y konumu: seridin yuksekliginin %65'i
        text_y = int(img.shape[0] * 0.70)
        # Bu y'deki ic ve dis kenar x degerlerini hesapla (perspektif)
        if side == -1:  # Sol yan serit
            inner_x_at_y = int(next_frame[0] + (text_y - next_frame[1]) * ((next_frame[2] - next_frame[0]) / (next_frame[3] - next_frame[1] + 0.001)))
            frac = (text_y - img.shape[0]) / (int(min_y) - img.shape[0] + 0.001)
            outer_x_at_y = int(ox + frac * (otx - ox))
        else:  # Sag yan serit
            inner_x_at_y = int(next_frame[4] + (text_y - next_frame[5]) * ((next_frame[6] - next_frame[4]) / (next_frame[7] - next_frame[5] + 0.001)))
            frac = (text_y - img.shape[0]) / (int(min_y) - img.shape[0] + 0.001)
            outer_x_at_y = int(ox + frac * (otx - ox))
        
        tx = int((inner_x_at_y + outer_x_at_y) / 2)
        if 30 < tx < img.shape[1] - 30:
            text_size = cv2.getTextSize(side_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            cv2.putText(img, side_text, (tx - text_size[0]//2, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.line(img, (tx - 30, text_y + 12), (tx + 30, text_y + 12), (0, 255, 255), 2)

    cache = next_frame

# pixels to a line
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_drawn(line_img,lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):

    global first_frame
    global wide_hough_lines

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Renk Filtresi: Cimen ve topragi yoksaymak icin sadece doygunlugu (S) cok dusuk alanlari al
    # Burada V (parlaklik) alt siniri 0, boylece golgedeki karanlik ama renksiz alanlar da gecer.
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([255, 45, 255])
    mask_color = cv2.inRange(img_hsv, lower_color, upper_color)

    # 2. Golge Filtresi (Local High-Pass Filter):
    # Goruntuyu cok genis bir sekilde flulastirip orijinalinden cikarirsak, golge veya gunes isigi gibi genel aydinlatma farklari silinir.
    # Geriye sadece kendi "yakin cevresine gore" parlak olan ince detaylar (seritler) kalir.
    blurred_bg = cv2.GaussianBlur(gray_image, (31, 31), 0)
    local_bright = cv2.subtract(gray_image, blurred_bg)
    
    # Cevresinden en az 15 birim daha parlak olan yerleri sec (Golgedeki beyaz serit bile etrafindaki golgeli asfalttan parlaktir)
    _, bright_mask = cv2.threshold(local_bright, 15, 255, cv2.THRESH_BINARY)
    
    # Iki maskeyi birlestir: Hem renksiz (cimensiz) hem de cevresinden belirgin sekilde parlak olan pikseller
    final_mask = cv2.bitwise_and(bright_mask, mask_color)

    # Maskenin dis hatlarini (kenarlarini) bul
    canny_edges = cv2.Canny(final_mask, 50, 150)

    imshape = image.shape
    # En kenarlardaki bariyerleri/kaldirimlari almamak icin alt kismi daha da daraltiyoruz (%15)
    lower_left = [imshape[1]*0.15, imshape[0]]
    lower_right = [imshape[1]*0.85, imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/4, imshape[0]/2+imshape[0]/18]
    top_right = [imshape[1]/2+imshape[1]/4, imshape[0]/2+imshape[0]/18]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = interested_region(canny_edges, vertices)

    theta = np.pi/180

    # Toprak/cimen gibi alanlardaki kucuk noktaciklarin (noise) birlesip cizgi sayilmasini engellemek icin
    # Hough threshold degeri 20'den 50'ye cikarildi.
    # maxLineGap=80: Kesikli serit cizgileri arasindaki buyuk bosluklar da birlestirilir
    # minLineLen=25: Daha kisa serit parcalarini da yakalar
    line_image = hough_lines(roi_image, 2, theta, 50, 25, 80)
    
    # --- 2. GENIS ROI: Yan serit algilama icin (ana takibi etkilemez) ---
    # %8-%92 genislik: bariyerleri ve cimenleri disarida birak
    wide_lower_left = [imshape[1]*0.08, imshape[0]]
    wide_lower_right = [imshape[1]*0.92, imshape[0]]
    wide_top_left = [imshape[1]*0.10, imshape[0]/2+imshape[0]/18]
    wide_top_right = [imshape[1]*0.90, imshape[0]/2+imshape[0]/18]
    wide_vertices = [np.array([wide_lower_left, wide_top_left, wide_top_right, wide_lower_right], dtype=np.int32)]
    wide_roi = interested_region(canny_edges, wide_vertices)
    wide_hough_lines = cv2.HoughLinesP(wide_roi, 2, theta, 45, np.array([]), minLineLength=15, maxLineGap=120)
    
    # Debug: Genis ROI ve uzerindeki tespit edilen cizgileri goster
    wide_debug = cv2.cvtColor(wide_roi, cv2.COLOR_GRAY2BGR)
    if wide_hough_lines is not None:
        for wl in wide_hough_lines:
            for wx1, wy1, wx2, wy2 in wl:
                cv2.line(wide_debug, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)
    cv2.imshow("Wide ROI (Yan Serit)", wide_debug)
    
    # Kullaniciya maskelenmis goruntuyu (Canny + Yesil Filtre + ROI) goster
    cv2.imshow("Masked Canny ROI", roi_image)
    
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    return result

# please give star if you like my project