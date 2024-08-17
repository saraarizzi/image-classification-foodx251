import multiprocessing
import time

import cv2 as cv
import os
import numpy as np
import pandas as pd

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_vector(im, bins):
    rgb_planes = cv.split(im)
    r = cv.calcHist(rgb_planes, [0], None, [bins], [0, 256], accumulate=False)
    g = cv.calcHist(rgb_planes, [1], None, [bins], [0, 256], accumulate=False)
    b = cv.calcHist(rgb_planes, [2], None, [bins], [0, 256], accumulate=False)

    vector = np.concatenate([
        np.array(r).astype(np.uint32).flatten(),
        np.array(g).astype(np.uint32).flatten(),
        np.array(b).astype(np.uint32).flatten()
    ], axis=0)

    return vector


def process_class(class_code):
    data_path = os.path.join(*["data", "clean", "train", str(class_code)])
    image_info_path = os.path.join(*["data", "raw", "train_info_dirty.csv"])
    image_data = pd.read_csv(image_info_path, header=None, names=["file", "class"])
    im_names = image_data[image_data["class"] == class_code]["file"].values

    ch_path = os.path.join(*["data", "ch", str(class_code)])
    os.makedirs(ch_path, exist_ok=True)

    images = []
    for i in im_names:
        try:
            im = cv.imread(os.path.join(data_path, i))
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            images.append(im)
        except Exception:
            print(i)
            raise Exception(f"Exception in reading images - class {class_code}, image {i}")

    for bins in [8, 16, 32, 64, 128, 256]:
        # Get histogram vectors
        vectors = []
        for im in images:
            v = get_vector(im, bins=bins)
            vectors.append(v)

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
