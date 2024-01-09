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
N = 1 # total image count will be N * CORE_COUNT!
CORE_COUNT = 20  # the number of logical processors in your system
RES = (1280, 720, 3)

def generate_noisy_image():
    # create random noise image
    rng = default_rng()
    noise = rng.integers(0, 70, (RES[1],RES[0]), np.uint8, True)
    # blur the noise
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=25, sigmaY=25, borderType = cv2.BORDER_DEFAULT)
    # stretch the image and threshold
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    # masking
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

def generate_image(index):
    # generate background image
    img = generate_noisy_image()

    # generating NOTES
    num_notes = np.random.randint(0,3)
    for i in range(num_notes):
        pass # bruh I don't want to render a donut...
    
    # save image
    cv2.imwrite(filename=IMAGE_FOLDER+"/Image"+str(index)+".png",img=img)

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

    # spread data
    print("Splitting data...")
    total_num = len(BASE_IMAGES)
    training_num = int(total_num * TRAINING)

    # # clear previous folders
    # files = glob.glob('data/train/images/*')
    # for f in files:
    #     os.remove(f)
    # files = glob.glob('data/valid/images/*')
    # for f in files:
    #     os.remove(f)
    # files = glob.glob('data/train/labels/*')
    # for f in files:
    #     os.remove(f)
    # files = glob.glob('data/valid/labels/*')
    # for f in files:
    #     os.remove(f)

    # for i, file in enumerate(BASE_IMAGES):
    #     if i < training_num:
    #         shutil.move(file, "data/train/images")
    #     else:
    #         shutil.move(file, "data/valid/images")

    # for i, file in enumerate(BASE_LABELS):
    #     if i < training_num:
    #         shutil.move(file, "data/train/labels")
    #     else:
    #         shutil.move(file, "data/valid/labels")