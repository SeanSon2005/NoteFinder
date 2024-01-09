import numpy as np 
import cv2
import time
from ultralytics import YOLO

# define color bounds
LOWER = np.array([60, 35, 140])
UPPER = np.array([180, 255, 255])

# send data to NetworkTables
def sendData(data):
    pass

# algorithm to find notes
def findNote(img, model, show=True):
    # analytical filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    bitwise = cv2.bitwise_and(img,mask=mask)

    # run model
    results = model(bitwise, verbose=False)

    # retrieve center from boxes
    centers = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centers.append((x,y))

    # display intermediates/ results
    if show:
        cv2.imshow('masked', bitwise)
        for center in centers:
            cv2.circle(img,center=center,radius=4,color=(255,0,0))
        cv2.imshow('targets',img)

    # handle center point list data
    sendData(centers)

if __name__ == '__main__':
    # load model
    conv_model = YOLO('best.pth')
    conv_model.info()

    # define camera
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # main loop
    while (True):
        start = time.time() 
        ret, frame = cap.read()
        findNote(frame, conv_model)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
