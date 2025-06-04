from ultralytics import YOLO
import cv2
import torch


model = YOLO("yolov8x.pt")


image = cv2.imread("ex2.jpg")

#personのみ抽出
results = model.predict(image, classes = [0], conf=0.1) 

img = results[0].orig_img
boxes = results[0].boxes

for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    
    
    x1 = int(box.data[0][0])
    y1 = int(box.data[0][1])
    x2 = int(box.data[0][2])
    y2 = int(box.data[0][3])

    cutImage = image[y1:y2, x1:x2]

    HSVimage = cv2.cvtColor(cutImage, cv2.COLOR_BGR2HSV)

    blueMask = cv2.inRange(HSVimage, (100, 90, 60), (130, 255, 255))

    blueCount = cv2.countNonZero(blueMask)

    if blueCount > 10:
     cv2.destroyAllWindows()
     cv2.rectangle(img, xy1.to(torch.int).tolist(), xy2.to(torch.int).tolist(), (0, 0, 255), thickness=3,)


    
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()