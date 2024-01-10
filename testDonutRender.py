import numpy as np
import matplotlib.pyplot as plt
import cv2

height = 600
width = 600
center = (300, 300)
radius = 40
image = np.zeros((720, 1080, 3), np.uint8)
homography = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
overlay = np.zeros((height, width, 3), np.uint8)
cv2.circle(overlay, center, radius, (0,0,255), 1, lineType=cv2.LINE_4)
output = cv2.warpPerspective(overlay, homography, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

cv2.imshow("ori", overlay)
cv2.imshow("warp",output)
cv2.waitKey(0)
cv2.destroyAllWindows()