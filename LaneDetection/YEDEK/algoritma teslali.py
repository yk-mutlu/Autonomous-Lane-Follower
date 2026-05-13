# Developer - JustNikhill

#Before detecting lane lines, we masked remaining objects and then identified the line with Hough transformation.

import numpy as np
import cv2

# Global variables for smoothing
cache = None
first_frame = 1
frame_counter = 0


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

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                continue
            slope = (y2-y1)/(x2-x1)
            
            # Alt noktadaki (y=img.shape[0]) tahmini x koordinatini hesapla
            x_bottom = ((img.shape[0] - y1) / slope) + x1
            
            if slope > 0.45:
                # Eger onceki karede sag serit biliniyorsa, 150 pikselden uzaktaki cizgileri (baska seritleri) yok say
                if cache is not None and abs(x_bottom - cache[4]) > 150:
                    continue
                slope_r.append(slope)
                lane_r.append(line)
            elif slope < -0.45:
                # Eger onceki karede sol serit biliniyorsa, 150 pikselden uzaktaki cizgileri (baska seritleri) yok say
                if cache is not None and abs(x_bottom - cache[0]) > 150:
                    continue
                slope_l.append(slope)
                lane_l.append(line)
        min_y = min(y1, y2, min_y)

    # --- TESLA STYLE: Gorulebilen butun kucuk serit parcalarini ayri ayri ciz (Cyan renginde) ---
    for line in lane_l:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [255, 200, 0], 2)
    for line in lane_r:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [255, 200, 0], 2)

    # Eger hic serit bulunamazsa iptal et
    if len(lane_l) == 0 and len(lane_r) == 0:
        print('no lane detected')
        return 1

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

    # --- HAREKETLI YATAY CIZGILER (Tahmin Matrisi) ---
    global frame_counter
    frame_counter += 1
    
    y_bottom = img.shape[0]
    y_top = int(min_y)
    lane_height = y_bottom - y_top
    num_lines = 10
    scroll_speed = 6 # Animasyon hizi
    
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
    
    offset = (frame_counter * scroll_speed) % lane_height
    bottom_most_y = -1
    active_color = (0, 255, 255)
    
    for i in range(num_lines):
        base_y = y_top + i * (lane_height / num_lines)
        y = y_top + (base_y - y_top + offset) % lane_height
        
        # Y noktasina karsilik gelen X noktalari (dogru denklemi x = x1 + (y - y1) * dx/dy )
        if dy_l != 0:
            x_l = next_frame[0] + (y - next_frame[1]) * (dx_l / dy_l)
        else:
            x_l = next_frame[0]
            
        if dy_r != 0:
            x_r = next_frame[4] + (y - next_frame[5]) * (dx_r / dy_r)
        else:
            x_r = next_frame[4]
            
        # Yatay cizgiyi ciz
        cv2.line(img, (int(x_l), int(y)), (int(x_r), int(y)), colors[i], 2)
        
        # En alta ulasan (sırasi gelen) cizgiyi tespit et
        if y > bottom_most_y:
            bottom_most_y = y
            active_color = colors[i]

    # --- TESLA STYLE: Aracin Tahmini Gidis Yonu (Merkez Cizgisi) ---
    center_bottom_x = int((next_frame[0] + next_frame[4]) / 2)
    center_top_x = int((next_frame[2] + next_frame[6]) / 2)
    
    # Merkez cizgisini sirasi gelen yatik cizginin rengiyle ciz
    cv2.line(img, (center_bottom_x, img.shape[0]), (center_top_x, int(min_y)), active_color, 4)

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

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Toprak ve cimen yansimalarini almamak icin Beyaz filtresini cok daha katilastirdik.
    # Gercek beyaz seritler haricindeki pikseller yok sayilacak. (Doygunluk S cok dusuk, Parlaklik V cok yuksek)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([255, 35, 255])
    mask_lane = cv2.inRange(img_hsv, lower_white, upper_white)
    
    # Sadece serit renklerini iceren gri goruntu
    gray_image_lanes = cv2.bitwise_and(gray_image, gray_image, mask=mask_lane)

    gauss_gray = cv2.GaussianBlur(gray_image_lanes, (5, 5), 0)

    canny_edges = cv2.Canny(gauss_gray, 50, 150)

    imshape = image.shape
    # En kenarlardaki bariyerleri/kaldirimlari almamak icin alt kismi daha da daraltiyoruz (%15)
    lower_left = [imshape[1]*0.15, imshape[0]]
    lower_right = [imshape[1]*0.85, imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/4, imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/4, imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = interested_region(canny_edges, vertices)

    theta = np.pi/180

    # Toprak/cimen gibi alanlardaki kucuk noktaciklarin (noise) birlesip cizgi sayilmasini engellemek icin
    # Hough threshold degeri 20'den 50'ye cikarildi.
    line_image = hough_lines(roi_image, 2, theta, 50, 40, 30)
    
    # Kullaniciya maskelenmis goruntuyu (Canny + Yesil Filtre + ROI) goster
    cv2.imshow("Masked Canny ROI", roi_image)
    
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    return result

# please give star if you like my project