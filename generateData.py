import os
import glob
import shutil
import multiprocessing
import numpy as np
import cv2
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

def generate_image(index):
    pass

if __name__ == '__main__':

    files = glob.glob(IMAGE_FOLDER + "/*")
    for f in files:
        os.remove(f)
    files = glob.glob(LABELS_FOLDER + "/*")
    for f in files:
        os.remove(f)

    try:
        pool = multiprocessing.Pool(processes=CORE_COUNT)

        # Process Images Standard.
        for i in tqdm(range(N), desc="generating data"):
            # Multi Process Images
            index = i * CORE_COUNT
            iter = np.zeros(CORE_COUNT, dtype=np.object_)
            for i in range(CORE_COUNT):
                iter[i] = (index + i)
            pool.starmap(generate_image, iter)

    finally:
        pool.close()
        pool.join()




    # spread data
    print("Splitting data...")
    total_num = len(BASE_IMAGES)
    training_num = int(total_num * TRAINING)

    # clear previous folders
    files = glob.glob('data/train/images/*')
    for f in files:
        os.remove(f)
    files = glob.glob('data/valid/images/*')
    for f in files:
        os.remove(f)
    files = glob.glob('data/train/labels/*')
    for f in files:
        os.remove(f)
    files = glob.glob('data/valid/labels/*')
    for f in files:
        os.remove(f)

    for i, file in enumerate(BASE_IMAGES):
        if i < training_num:
            shutil.move(file, "data/train/images")
        else:
            shutil.move(file, "data/valid/images")

    for i, file in enumerate(BASE_LABELS):
        if i < training_num:
            shutil.move(file, "data/train/labels")
        else:
            shutil.move(file, "data/valid/labels")