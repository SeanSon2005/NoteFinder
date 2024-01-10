import os
import glob
import shutil
import multiprocessing
import numpy as np
import cv2
import skimage.exposure
from numpy.random import default_rng
from tqdm.auto import tqdm

# Define folder locations
IMAGE_FOLDER = "base_images"
LABELS_FOLDER = "base_labels"
BASE_IMAGES = glob.glob(IMAGE_FOLDER+'/*')
BASE_LABELS = glob.glob(LABELS_FOLDER+'/*')

# Constants
TRAINING = 0.85 # percent of data to be training
N = 300 # total image count will be N * CORE_COUNT!
CORE_COUNT = 24  # the number of logical processors in your system
RES = (1280, 720, 3) # Resolution of OUTPUT images
NOISE_SPARSE = 100 # EVERY 1 in NOISE_SPARSE pixels will be noise
EROSION_KERNEL_SIZE = 5 # Proportionally influences the size of 
                        # the "holes" the noise generator makes
HOLE_BLUR_FACTOR = 21 # Kernel size of gaussian blur after erosion
USE_SEED = True

# Renders a note
def renderNote(img,rng):
    height, width = img.shape
    size = rng.integers(40, 250)
    stretch = rng.random() + 1
    sizeX = int(size * stretch)
    sizeY = int(size / stretch)
    centerX = rng.integers(0, width)
    centerY = rng.integers(0, height)
    cv2.ellipse(img=img,
                center=(centerX,centerY),
                axes=(sizeX, sizeY),
                angle=rng.integers(-45, 46),
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=int(size/2))

    return img, centerX, centerY, sizeX, sizeY

# Generates a black image with white blobs as noise
def generate_noisy_image(noise_intensity, blur_factor, rng):
    # create random noise image
    noise = rng.integers(0, noise_intensity, (RES[1],RES[0]), np.uint8, True)
    # blur the noise
    blur = cv2.GaussianBlur(noise, (0,0), 
                            sigmaX=blur_factor, 
                            sigmaY=blur_factor, 
                            borderType = cv2.BORDER_DEFAULT)
    # stretch the image and threshold
    stretch = skimage.exposure.rescale_intensity(blur, 
                                                 in_range='image', 
                                                 out_range=(0,255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    # masking
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

# Generates the training image
def generate_image(index):
    # set the default rng
    if USE_SEED:
        rng = default_rng(index)
    else:
        rng = default_rng()

    # generate background image
    img = generate_noisy_image(rng.integers(80,120),25,rng)

    # generating NOTES (record labels as well)
    num_notes = rng.integers(0, 4)
    with open(LABELS_FOLDER + '/Image'+str(index)+'.txt','w') as file:
        for i in range(num_notes):
            img,x,y,sizeX,sizeY = renderNote(img,rng)
            file.write('0 ' + str(x/RES[0]) + ' ' + 
                       str(y/RES[1]) + ' ' + 
                       str((sizeX*2)/RES[0]) + ' ' + 
                       str((sizeY*2)/RES[1]) + '\n')
        file.close()

    # add REALISTIC noise to blobs and notes
    noise = rng.integers(0, NOISE_SPARSE, (RES[1],RES[0]), np.uint8, True)
    noise = cv2.threshold(noise, NOISE_SPARSE-1, 255, cv2.THRESH_BINARY)[1]
    img = np.maximum(img - noise, 0)
    img = cv2.erode(img, 
                    np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), 
                                 np.uint8)) 
    img = cv2.GaussianBlur(img, 
                           (HOLE_BLUR_FACTOR,HOLE_BLUR_FACTOR),
                           0)
    img = cv2.threshold(img, 
                        (100 + rng.integers(0,60)), 
                        255, 
                        cv2.THRESH_BINARY)[1]
    # add REALISTIC background grain noise
    mask_noise = rng.integers(0, 255, (RES[1],RES[0]), np.uint8, True)
    mask = cv2.GaussianBlur(mask_noise, (0,0), 
                            sigmaX=20, 
                            sigmaY=20, 
                            borderType = cv2.BORDER_DEFAULT)
    mask = cv2.threshold(mask, 129, 255, cv2.THRESH_BINARY)[1]
    noise = rng.integers(0, 255, (RES[1],RES[0]), np.uint8, True)
    noise = cv2.threshold(noise, 250, 255, cv2.THRESH_BINARY)[1]
    noise_final = cv2.bitwise_and(noise,mask)
    img = np.minimum(img + noise_final, 255)

    # save image
    cv2.imwrite(filename=IMAGE_FOLDER+"/Image"+str(index)+".png",img=img)

# Main Function
if __name__ == '__main__':
    for f in BASE_IMAGES:
        os.remove(f)
    for f in BASE_LABELS:
        os.remove(f)

    try:
        pool = multiprocessing.Pool(processes=CORE_COUNT)

        # Process Images Standard.
        for i in tqdm(range(N), desc="generating data"):
            # Multi Process Images
            index = i * CORE_COUNT
            pool.map(generate_image, range(index,index+CORE_COUNT))
    finally:
        pool.close()
        pool.join()