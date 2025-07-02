from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
image = cv2.imread("ex4.jpg")
results = model.predict(image, conf=0.25)

img = results[0].orig_img.copy()
boxes = results[0].boxes

countW = 0
countB = 0

for box in boxes:
    x1 = int(box.data[0][0])
    y1 = int(box.data[0][1])
    x2 = int(box.data[0][2])
    y2 = int(box.data[0][3])

    id = int(box.data[0][5])

    if id == 0:
        color = (255, 255, 255)
        countW += 1
    else:
        color = (0, 0, 0)
        countB += 1

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

print("White:", countW)
print("Black:", countB)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
