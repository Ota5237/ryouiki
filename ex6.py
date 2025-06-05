from ultralytics import YOLO
import cv2
import torch


model = YOLO("yolov8x.pt")


image = cv2.imread("ex3.jpg")



#personのみ抽出
results = model.predict(image, classes = [0], conf=0.2) 

img = results[0].orig_img
boxes = results[0].boxes

for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    
    
    x1 = int(box.data[0][0]) +10
    y1 = int(box.data[0][1]) +10
    x2 = int(box.data[0][2]) -10
    y2 = int(box.data[0][3]) -10

    cutImage = image[y1:y2, x1:x2]

    HSVimage = cv2.cvtColor(cutImage, cv2.COLOR_BGR2HSV)


    blueMask = cv2.inRange(HSVimage, (105, 80, 40), (125, 220, 180))
    redMask = cv2.inRange(HSVimage, (160, 100, 40), (180, 255, 150))
    yellowMask = cv2.inRange(HSVimage, (22, 160, 160), (32, 255, 255)) 
    whiteMask = cv2.inRange(HSVimage, (80, 0, 60), (90, 30, 110))

    com = cv2.bitwise_or(blueMask, whiteMask)
    com = cv2.bitwise_or(com, redMask)

    blueCount = cv2.countNonZero(blueMask)
    redCount = cv2.countNonZero(redMask)
    yellowCount = cv2.countNonZero(yellowMask)
    whiteCount = cv2.countNonZero(whiteMask)

    comCount = cv2.countNonZero(com)
    count = abs(x2-x1)*abs(y2-y1)



   # cv2.imshow("A", yellowMask)
   # cv2.imshow("B", com)
    #cv2.imshow("C", cutImage)
   # cv2.waitKey(0)

    if yellowCount > 25 :
        cv2.rectangle(img, xy1.to(torch.int).tolist(), xy2.to(torch.int).tolist(), (0, 0, 255), thickness=3,)
        cv2.destroyAllWindows()
    elif comCount/count > 0.03:
        cv2.rectangle(img, xy1.to(torch.int).tolist(), xy2.to(torch.int).tolist(), (0, 255, 0), thickness=3,)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()


img  = cv2.resize(img, None,  fx = 0.5, fy = 0.5)
    
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()