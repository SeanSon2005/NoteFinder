import numpy as np
import cv2
import math
import sys

def renderNote(img):
    height, width = img.shape
    size = np.random.randint(50,200)
    centerX = np.random.randint(0,width)
    centerY = np.random.randint(0,height)
    cv2.circle(img=img,center=(centerX,centerY),radius=size,color=255,thickness=-1)

    return img, centerX, centerY, size