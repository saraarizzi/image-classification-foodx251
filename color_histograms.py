import multiprocessing
import time

import cv2 as cv
import os
import numpy as np
import pandas as pd

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_vector(im, bins=32):
    r = cv.calcHist(im, [0], None, [bins], [0, 256])
    g = cv.calcHist(im, [1], None, [bins], [0, 256])
    b = cv.calcHist(im, [2], None, [bins], [0, 256])
    vector = np.concatenate([r, g, b], axis=0)
    vector = vector.reshape(-1)
    return vector


def process_class(class_code):
    data_path = os.path.join(*["data", "clean", "train", str(class_code)])
    im_names = os.listdir(data_path)

    ch_path = os.path.join(*["data", "ch", str(class_code)])
    os.makedirs(ch_path, exist_ok=True)

    images = []
    for i in im_names:
        im = cv.imread(os.path.join(data_path, i))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        images.append(im)

    for bins in [8, 16, 32, 64, 128, 256]:
        # Get histogram vectors
        vectors = []
        for im in images:
            v = get_vector(im, bins=bins)
            vectors.append(np.array(v, dtype=np.uint8))

        pd.DataFrame({
            "image_name": im_names,
            "hist": list(map(lambda x: np.array2string(x, separator=","), vectors))
        }).to_csv(os.path.join(ch_path, f"ch_{bins}.csv"), sep=";", index=False)

    print(f"Done --- {class_code + 1}/251")
    return


if __name__ == "__main__":
    start_time = time.time()
    print(f"Started at: {time.strftime(DATE_FORMAT)}")

    classes_data = pd.read_csv("data/class_list.txt", sep=" ", header=None)
    classes_data.columns = ["code", "label"]
    classes_codes = classes_data.code.tolist()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_class, classes_codes)

    print(f"Ended at: {time.strftime(DATE_FORMAT)}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script total duration: {duration:.2f} seconds")
