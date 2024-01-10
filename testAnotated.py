import numpy as np 
import cv2
import time

# define color bounds
LOWER = np.array([5, 80, 50])
UPPER = np.array([35, 255, 255])

# algorithm to find notes
def findNote(img):
    # analytical filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return mask

if __name__ == '__main__':
    # define camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # main loop
    while (True):
        start = time.time() 
        ret, frame = cap.read()

        cv2.imshow('ori',frame)
        img = findNote(frame)
        cv2.imshow('hi',img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
