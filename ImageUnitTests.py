import numpy as np 
import cv2
import time
from ultralytics import YOLO

# define color bounds
LOWER = np.array([5, 80, 50])
UPPER = np.array([35, 255, 255])

# send data to NetworkTables
def sendData(data):
    pass

# algorithm to find notes
def findNote(img, model):
    # analytical filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("imageMask.png",mask)

    # run model
    results = model(mask, verbose=False, conf=0.6)

    # retrieve center from boxes
    centers = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.line(img,(x,int(y1)),(x,int(y2)),color=(255,0,0),thickness=10)
            cv2.line(img,(int(x1),y),(int(x2),y),color=(255,0,0),thickness=10)
            centers.append((x,y))

    # handle center point list data
    sendData(centers)

    return img

if __name__ == '__main__':
    # load model
    conv_model = YOLO('best.pt')
    conv_model.info()

    img = cv2.imread('note2.jpg')

    img = findNote(img, conv_model)

    cv2.imwrite("imageAnnotated.png",img)
    
    