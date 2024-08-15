import os
import time
import multiprocessing
import joblib

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_desc(names, d_path: str):
    sift = cv.SIFT_create()
    class_desc, to_remove = [], []
    for im in names:
        image = cv.imread(os.path.join(d_path, im))
        bw_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, im_desc = sift.detectAndCompute(bw_image, None)
        if im_desc is not None:
            pass
            # class_desc.append(np.array(im_desc))
        else:
            to_remove.append(im)

    return class_desc, to_remove


def get_codebook(descriptors, k):
    cb, variance = kmeans(np.vstack(descriptors), k, iter=5)

    return cb


def get_vw_vectors(descriptors, cb):
    descriptors_vw = []
    for d in descriptors:

        img_visual_words, distance = [], []
        for i in range(len(d)):
            vw, dist = vq(d[i].reshape(1, 128), cb)
            img_visual_words.append(vw.item())
            distance.append(dist.item())

        descriptors_vw.append(img_visual_words)

    return descriptors_vw


def get_tf_vectors(descriptors_vw, k):
    frequency_vectors = []
    for img_visual_words in descriptors_vw:
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)

    return frequency_vectors


def get_tfidf_vectors(tf, c_code, b_path):
    tfidf_path = os.path.join(b_path, f"{c_code}_tfidf.pkl")

    if os.path.isfile(tfidf_path):
        tfidf = joblib.load(tfidf_path)
    else:
        n = len(tf)
        df = np.sum(np.array(tf) > 0, axis=0)
        idf = np.log(n / df)
        tfidf = tf * idf
        joblib.dump(tfidf, tfidf_path, compress=3)

    return tfidf


def process_class(params):
    class_code = params[0]
    k = params[1]

    bovw_path = os.path.join(*["data", "bovw", f"{k}_vw"])
    os.makedirs(bovw_path, exist_ok=True)

    data_path = os.path.join(*["data", "clean", "train", f"{class_code}"])
    images_names = os.listdir(data_path)

    # Get descriptors for all the images in the class
    images_desc, images_to_remove = get_desc(images_names, data_path)
    for im in images_to_remove:
        images_names.remove(im)

    # Create visual words
    codebook = get_codebook(images_desc, k)

    # Visual words vector
    images_vw = get_vw_vectors(images_desc, codebook)

    # Create tf vectors
    images_tf = get_tf_vectors(images_vw, k)

    # Created tf-idf vectors
    get_tfidf_vectors(images_tf, class_code, bovw_path)

    return


if __name__ == "__main__":

    for k_vw in [5, 10, 20, 30]:
        start_time = time.time()
        print(f"Started {k_vw} at: {time.strftime(DATE_FORMAT)}")

        classes_data = pd.read_csv("data/class_list.txt", sep=" ", header=None)
        classes_data.columns = ["code", "label"]
        classes_codes = classes_data.code.tolist()
        args = [(cl_code, k_vw) for cl_code in classes_codes]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(process_class, args)

        print(f"Ended {k_vw} at: {time.strftime(DATE_FORMAT)}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Script total duration for {k_vw}: {duration:.2f} seconds")
