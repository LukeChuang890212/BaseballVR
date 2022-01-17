# _*_ coding:utf-8 _*_

#%%
import numpy as np
from numpy.core.defchararray import _center_dispatcher, array

import pandas as pd
import math
from scipy.spatial import distance_matrix
import time 

from ContinuousShot_v2 import Cap
import cv2
import matplotlib.pyplot as plt

from DataProcess import DataProcessor


#%%
def detect(frame,kernel,avg,avg_float,lightness,lowerbH,lowerbS,lowerbV,upperbH,upperbS,upperbV,lowerth,upperth,right_x_grid,left_x_grid,upper_y_grid,lower_y_grid,yellow_grid_len,grid_len,gridok,ball_in,ball_list,record):
    
    fImg = frame.astype(np.float32)
    fImg = fImg / 255.0

    # 調整亮度前轉成HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    # 亮度調整
    # print("lightness:",lightness)
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR 
    frame = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    frame = ((frame * 255).astype(np.uint8))
  
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 將圖片轉為黃
    frame_target = cv2.inRange(frame_hsv, (lowerbH,lowerbS,lowerbV),(upperbH,upperbS,upperbV))
    frame_specifiedColor = cv2.bitwise_and(frame, frame, mask=frame_target)
    frame_specifiedColor = cv2.erode(frame_specifiedColor, kernel, iterations = 1)
    
    gray = cv2.cvtColor(frame_specifiedColor, cv2.COLOR_BGR2GRAY)

    # 模糊處理
    blur = cv2.blur(frame, (4, 4))

    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(avg, blur)
    # diff = cv2.dilate(diff, kernel, iterations = 1)

    frame, ball_in, ball_list, record = detect_ball(diff, frame,left_x_grid,upper_y_grid,grid_len,ball_in,ball_list,record)

    if gridok:
        global record_output_file
        with open(record_output_file+'temp.txt', 'w') as f:
            f.write('%d' % grid_len)
        cv2.circle(frame, (0,0), radius=3, color=(0, 255, ), thickness=-1)
        pass
    else:
        edged = cv2.Canny(gray, lowerth, upperth)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            # 忽略太大或太小的區域  
            if cv2.contourArea(c) > 800 or cv2.contourArea(c) < 80:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.drawContours(frame, c, -1, (255, 255, 255), 2)

            if x > right_x_grid:
                right_x_grid = x+w
            if x < left_x_grid:
                left_x_grid = x
            if y < upper_y_grid:
                upper_y_grid = y+h
            if y+h > lower_y_grid:
                lower_y_grid = y

        cv2.circle(frame, (left_x_grid,upper_y_grid), radius=3, color=(0, 255, 255), thickness=10) 
        yellow_grid_len = int((right_x_grid-left_x_grid)/3)
        grid_len = int((lower_y_grid-upper_y_grid)/3)
    
    for i in range(4):
        for j in range(4):
            cv2.circle(frame, (left_x_grid+i*grid_len,upper_y_grid+j*grid_len), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (left_x_grid+i*yellow_grid_len,upper_y_grid+j*yellow_grid_len), radius=3, color=(0, 255, 0), thickness=-1) 

    # 更新平均影像
    cv2.accumulateWeighted(blur, avg_float, 0.01)
    avg = cv2.convertScaleAbs(avg_float)

    return frame, avg, right_x_grid, left_x_grid, upper_y_grid, lower_y_grid, yellow_grid_len, grid_len, gridok, ball_in, ball_list, record

#%%
def detect_ball(diff,frame,left_x_grid,upper_y_grid,grid_len,ball_in,ball_list,record):
    if not ball_in:
        # 偵測圓形的東西
        arr1 = detect_by_circle(frame,ball_in)
        # 偵測變動大的東西
        arr2 = detect_by_change(diff,ball_in)

        # ==========================================================   
        # 如果arr1、arr2長度皆大於0 
        # 找出兩種方式下，最接近的兩個點，如果夠近，就取該兩點平均為center
        # 否則取arr1自己的平均為center
        # 如果兩者長度皆=0，則pass表示not ball in 
        # ==========================================================
        if (len(arr1) > 0) and (len(arr2) > 0): 
            two_arr_dist = distance_matrix(arr1,arr2)
            smallest_dist_index = np.where(two_arr_dist == np.min(two_arr_dist))
            if two_arr_dist[smallest_dist_index] < 5:
                center = np.mean([arr1[smallest_dist_index[0]][0],arr2[smallest_dist_index[1]][0]],axis=0)
                ball_list.append(center)
                ball_in = True
        elif (len(arr1) > 0) and (len(arr2) == 0):
            center = np.mean(arr1,axis=0)
            ball_list.append(center)
            ball_in = True
        else:
            pass

    elif ball_in:
        trackable = False
        ball_pos = ball_list[-1] # 上個frame偵測到的球位置

        # 偵測圓形的東西
        cnts = detect_by_circle(frame,ball_in)
        trackable, ball_list = try_if_circle_trackable(cnts,ball_list,ball_pos,trackable)
        if not trackable:
            # 偵測變動大的東西
            cnts = detect_by_change(diff,ball_in)
            trackable, ball_list = try_if_change_trackable(cnts,ball_list,ball_pos,trackable)
        else:
            pass
        
        # 確定追蹤不到了，則取最終追蹤位置為落點
        if not trackable: 
            global start_not_track_time
            global record_output_file
            global name

            # cv2.circle(frame, ball_pos, radius=10, color=(0, 0, 255), thickness=-1)
            print(time.time()-start_not_track_time)
            if time.time()-start_not_track_time < 8:
                pass
            else:
                start_not_track_time = time.time()
                # print("ball_list:",ball_list)
                ball_pos = [int(i) for i in ball_pos]
                record.append(tuple(ball_pos))
                pd.DataFrame({'index':[len(record)],"TimeStamp":[time.time()],'x':[ball_pos[0]-left_x_grid],'y':[ball_pos[1]-upper_y_grid]}).to_csv(record_output_file+"temp.csv",index=False,header=False,mode='a')
                print("ball_pos_x:",ball_pos[0]-left_x_grid)
                print("ball_pos_y:",ball_pos[1]-upper_y_grid)

                ball_list = []

            ball_in = False
            # time.sleep(5)    

    return frame, ball_in, ball_list, record

def try_if_change_trackable(cnts,ball_list,ball_pos,trackable):
    for c in cnts:
            # print(c)
            # 計算等高線的外框範圍
            (x, y, w, h) = cv2.boundingRect(c)

            # 計算球的中心
            # M = cv2.moments(c)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # center = (cX,cY)
            center = (int(x+w/2),int(y+h/2))

            # if cv2.contourArea(c) > 200:
            #     continue

            # 只追蹤球，忽略軌跡外的東西
            print("ball_list:",ball_list)
            print("center:",center)
            print("ball_pos:",ball_pos)
            if (center[0] > ball_pos[0]-10 and center[0] < ball_pos[0]+10) and (center[1] > ball_pos[1]-10 and center[1] < ball_pos[1]+10): 

                trackable = True
                ball_list.append(center)
                
                # 畫出外框
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # cv2.drawContours(frame, c, -1, (255, 255, 55), 5)
                # time.sleep(5)
                break
            else:
                continue
    
    return trackable, ball_list

def try_if_circle_trackable(cnts,ball_list,ball_pos,trackable):
    for c in cnts:
            (x,y),r = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))

            if r >= 4 and r <= 8:
                continue

            # 只追蹤球，忽略軌跡外的東西
            print("ball_list:",ball_list)
            print("center:",center)
            print("ball_pos:",ball_pos)
            if (center[0] > ball_pos[0]-10 and center[0] < ball_pos[0]+10) and (center[1] > ball_pos[1]-10 and center[1] < ball_pos[1]+10): 

                trackable = True
                ball_list.append(center)
                
                # 畫出外框
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # cv2.drawContours(frame, c, -1, (255, 255, 55), 5)
                # time.sleep(5)
                break
            else:
                continue
    
    return trackable, ball_list
     
def detect_by_circle(frame, ball_in):
    # frame = detect_circle(frame)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    gray2 = cv2.dilate(gray2, np.ones((5, 5), np.uint8), iterations = 1)
    gray2 = cv2.Canny(gray2,30,100)

    cnts, _ = cv2.findContours(gray2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ball_in:
        return cnts
    else:
        array = []
        # print(len(cnts))
        for c in cnts:
            (x,y),r = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            r = int(r)
            color = frame[center[1],center[0]]
            if (r>=4 and r<=8) and (min(color)>100):
                # 忽略九宮格外的東西
                if (center[0] < left_x_grid or center[0] > left_x_grid+3*grid_len) or (center[1] < upper_y_grid or center[1] > upper_y_grid+3*grid_len):      
                    continue

                print("r:",r)
                cv2.circle(frame,center,r,(255,255,255),8)

                array.append(center)
                print("circle_array:", array)
                # time.sleep(5)
            # print("color:",frame[center[1],center[0]])
        # cv2.imshow("frame", frame)

        return np.array(array)

def detect_by_change(diff, ball_in):
    # 將圖片轉為灰階
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 篩選出變動程度大於門檻值的區域
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # 使用型態轉換函數去除雜訊
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 產生等高線
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ball_in:
        return cnts
    else:
        array = []
        for c in cnts:
                # 計算等高線的外框範圍
                (x, y, w, h) = cv2.boundingRect(c)

                # 計算偵測到的東西的輪廓中心
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX,cY)

                # 忽略九宮格外的東西
                if (center[0] < left_x_grid or center[0] > left_x_grid+3*grid_len) or (center[1] < upper_y_grid or center[1] > upper_y_grid+3*grid_len):      
                    continue

                # color = frame[y,x] # 125 190 # min(color[0],color[1],color[2]) < 100 or 
                # print("color",color[0])
                # if color[0]< 30:
                #     break

                # 畫出外框
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.drawContours(frame, c, -1, (255, 255, 255), 2)
                print("change_array:", array)

                array.append(center)

                # break

        return np.array(array)
#%%
def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(data['img'], (x,y), 3, (255,255,255), 5, 16) 

        # # 改變顯示 window 的內容
        # cv2.imshow("frame", data['img'])
        
        # 顯示 (x,y) 並儲存到 list中
        print("get points: (x, y) = ({}, {})".format(x, y))
        data['points'].append((x,y))

#%%
def revise_last_record(data, record):
    if len(data["points"])== 1:
        if len(record) >= 1:
            record.pop(len(record)-1)
            record.append(data["points"][0])
            data["points"].clear()
        else:
            record.append(data["points"][0])
            data["points"].clear()
        revise_data_last_row(record_output_file, record)
    else: 
        pass
    return data, record

#%%
def revise_data_last_row(record_output_file, record):
    # 刪掉最後一列
    temp = pd.read_csv(record_output_file+"temp.csv")
    if temp.shape[0] >= 1:
        temp = temp[:-1]
        print("del temp:",temp)
        temp.to_csv(record_output_file+"temp.csv",index=False)
    else:
        pass

    # 添加最後一列
    ball_pos = record[-1]
    pd.DataFrame({'index':[len(record)],
                  "TimeStamp":[time.time()],
                  'x':[ball_pos[0]-left_x_grid],
                  'y':[ball_pos[1]-upper_y_grid]}).to_csv(record_output_file+"temp.csv",
                                                          index=False,header=False,mode='a')

#%%

read_mode = 1
record_output_file = "D:/BaseballVR/Camera/Recording Output/"

# ==========================
# read mode = 0 -> 輸入影片
# read mode = 1 -> real time
# ==========================
if read_mode == 0:
    cap = cv2.VideoCapture(record_output_file+'Luke-test5-l.avi')
elif read_mode == 1:
    cap = Cap()

date = "2021-10-31-"
name = "Luke"
suffix = "-t1"

# 棒球落點數據temp檔
data = pd.DataFrame(columns=["index","TimeStamp",'x','y'])
data.to_csv(record_output_file+"temp.csv",index=False)

winName = "rainbow"
def nothing(x):
    pass
    
cv2.namedWindow(winName)

cv2.createTrackbar("Lightness", winName, 0, 100, nothing) 
cv2.createTrackbar("LowerbH", winName, 10, 255, nothing) 
cv2.createTrackbar("LowerbS", winName, 43, 255, nothing) 
cv2.createTrackbar("LowerbV", winName, 46, 255, nothing)
cv2.createTrackbar("UpperbH", winName, 34, 255, nothing)
cv2.createTrackbar("UpperbS", winName, 255, 255, nothing)
cv2.createTrackbar("UpperbV", winName, 255, 255, nothing)
cv2.createTrackbar("Lowerth", winName, 50, 255, nothing)
cv2.createTrackbar("Upperth", winName, 255, 255, nothing)

kernel = np.ones((3,3), np.uint8)

# 初始化平均影像
ret, frame = cap.read()
frame = frame[220:400,320:450]
avg = cv2.blur(frame, (4, 4))
avg_float = np.float32(avg)

# 初始化九宮格邊界、邊長
right_x_grid = 0
left_x_grid = 100000
upper_y_grid = 100000
lower_y_grid = 0

yellow_grid_len = 0
grid_len = 0
gridok = False

# 初始化偵測狀態
detect_method = 0 # 0:detect_by_circle() 1:detect_by_change()
ball_in = False
ball_list = []
record = []
start_not_track_time = 0

# ===============================================================
# 建立影像視窗
# ===============================================================

# 建立 data dict, img:存放圖片, points:存放點
################################
# gridok == F -> data存九宮格座標 
# gridok == T -> data存球點座標
################################
data = {} 
data['img'] = frame.copy()
data['points'] = []

# 建立一個 window
cv2.namedWindow("frame", 0)

# 改變 window 成為適當圖片大小
h, w, dim = frame.shape
print("Img height, width: ({}, {})".format(h, w))
cv2.resizeWindow("frame", w, h)
    
# 顯示圖片在 window 中
cv2.imshow('frame',frame)

# 利用滑鼠回傳值，資料皆保存於 data dict中
cv2.setMouseCallback("frame", mouse_handler, data)
    
# ===============================================================
# while loop : 讀取影像幀
# ===============================================================
while(True):
    lightness = cv2.getTrackbarPos("Lightness",winName)
    lowerbH = cv2.getTrackbarPos("LowerbH", winName)
    lowerbS = cv2.getTrackbarPos("LowerbS", winName)
    lowerbV = cv2.getTrackbarPos("LowerbV", winName)
    upperbH = cv2.getTrackbarPos("UpperbH", winName)
    upperbS = cv2.getTrackbarPos("UpperbS", winName)
    upperbV = cv2.getTrackbarPos("UpperbV", winName)
    lowerth = cv2.getTrackbarPos("Lowerth", winName)
    upperth = cv2.getTrackbarPos("Upperth", winName)

    ret, frame = cap.read()
 
    #裁減中央區塊
    if frame is not None:
        frame = frame[220:400,320:450]
    elif frame is None:
        break

    frame, avg, right_x_grid, left_x_grid, upper_y_grid, lower_y_grid, yellow_grid_len, grid_len, gridok, ball_in, ball_list, record =  detect(frame,kernel,avg, avg_float,lightness,lowerbH,lowerbS,lowerbV,upperbH,upperbS,upperbV,lowerth,upperth,right_x_grid,left_x_grid,upper_y_grid,lower_y_grid,yellow_grid_len,grid_len,gridok,ball_in,ball_list,record)
    data['img'] = frame.copy()

    # 利用滑鼠回傳值，資料皆保存於 data dict中
    cv2.setMouseCallback("frame", mouse_handler, data)   
    print("points",data['points'])

    # 檢查是否需要修正偵測結果
    if gridok == True:
        data, record = revise_last_record(data, record)
    else:
        pass
    
    if len(record) >= 1:
        for center in record:
            cv2.circle(frame, center, radius=5, color=(0,255,255), thickness=-1)
    
    # 顯示圖片在 window 中
    cv2.imshow('frame',frame)
    print("hello")

    k = cv2.waitKey(1)

    if k == 27:
        break
    elif k == 109: #m
        gridok = True
    elif k == 111: #o
        #設定九宮格參數
        left_x_grid = data["points"][0][0]
        right_x_grid = data["points"][1][0]
        lower_y_grid = data["points"][1][1]
        upper_y_grid = data["points"][0][1]
        grid_len = int((lower_y_grid-upper_y_grid)/3)
        
        gridok = True

        data["points"].clear()
    elif k == 110: # n
        gridok = False 

    time.sleep(0.01)

print("lightness:",lightness)
print("lowerbH:",lowerbH)
print("lowerbS:",lowerbS)
print("lowerbV:",lowerbV)
print("upperbH:",upperbH)
print("upperbS:",upperbS)
print("upperbV:",upperbV)

cv2.destroyAllWindows()

# %%
data = pd.read_csv(record_output_file+"temp.csv")

dataProcessor = DataProcessor(data, record_output_file)
data, pos_temps = dataProcessor.process(name, date, grid_len)

data.to_csv(record_output_file+date+name+suffix+".csv")
print(data)

pos_temps.to_csv(record_output_file+date+name+suffix+"_wrist_pos.csv")
print(pos_temps)

#%%
