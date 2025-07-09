from ultralytics import YOLO
import cv2

model = YOLO("bestForEx9.pt")
image = cv2.imread("ex3.jpg")
results = model.predict(image, conf=0.3)

img = results[0].orig_img.copy()
boxes = results[0].boxes

countA = 0
countB = 0

for box in boxes:
    x1 = int(box.data[0][0])
    y1 = int(box.data[0][1])
    x2 = int(box.data[0][2])
    y2 = int(box.data[0][3])

    id = int(box.data[0][5])

    if id == 0:
        color = (255, 255, 255)
        countA += 1
    elif id == 1:
        color = (0, 0, 0)
        countB += 1

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

img  = cv2.resize(img, None,  fx = 0.5, fy = 0.5)

print("A:", countA)
print("B:", countB)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
