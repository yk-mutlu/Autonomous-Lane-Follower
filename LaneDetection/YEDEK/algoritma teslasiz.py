# Developer - JustNikhill

#Before detecting lane lines, we masked remaining objects and then identified the line with Hough transformation.

import numpy as np
import cv2

# Global variables for smoothing
cache = None
first_frame = 1


# Frame masking and region of interest
def interested_region(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        mask_color_ignore = (255,) * img.shape[2]
    else:
        mask_color_ignore = 255

    cv2.fillPoly(mask, vertices, mask_color_ignore)
    return cv2.bitwise_and(img, mask)

def lines_drawn(img, lines, color=[255, 0, 0], thickness=6):
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

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)

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