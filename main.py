import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np

model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cafeV.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()  

area1 = [(227, 131), (374, 101), (386, 142), (256, 182)]
area2 = [(176, 70), (639, 0), (639, 359), (228, 357)]
area3 = [(0, 43), (151, 48), (43, 356), (1, 357)]

frame_counter = 0
skip_frames = 5
area_count={}
counter1=[]
counter2=[]
counter3=[]
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    frame = cv2.resize(frame, (640, 360))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]

        if 'person' in c:
             list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
         x3,y3,x4,y4,id=bbox
         result=cv2.pointPolygonTest(np.array(area2, np.int32),((x4,y4)),False)
         if result>=0:
            area_count[id]=(x4,y4)
            cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 0)
            cvzone.putTextRect(frame, f"{id}", (x3, y3), 1, 1)
            if counter1.count(id)==0:
                counter1.append(id)

         elif 'cup' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
         x3,y3,x4,y4,id=bbox
         result=cv2.pointPolygonTest(np.array(area1, np.int32),((x4,y4)),False)
         if result>=0:
            area_count[id]=(x4,y4)
            cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 0)
            cvzone.putTextRect(frame, f"{id}", (x3, y3), 1, 1)
            if counter2.count(id)==0:
                counter2.append(id)

         elif 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
         x3,y3,x4,y4,id=bbox
         result=cv2.pointPolygonTest(np.array(area3, np.int32),((x4,y4)),False)
         if result>=0:
            area_count[id]=(x4,y4)
            cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 0)
            cvzone.putTextRect(frame, f"{id}", (x3, y3), 1, 1)
            if counter3.count(id)==0:
                counter3.append(id)

    coffee_counter = (len(counter2))
    employee_counter = (len(counter1))
    customer_counter = (len(counter3))

    cvzone.putTextRect(frame, f"CoffeeCup: {coffee_counter}", (50, 30), 1, 1)
    cvzone.putTextRect(frame, f"Employees: {employee_counter}", (185, 30), 1, 1)
    cvzone.putTextRect(frame, f"Customers: {employee_counter}", (320, 30), 1, 1)
    
    cv2.polylines(frame,[np.array(area1, np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area2, np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(area3, np.int32)],True,(255,0,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
