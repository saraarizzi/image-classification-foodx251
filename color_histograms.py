import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_vector(im, bins=32):
    r = cv.calcHist(im, [0], None, [bins], [0, 256])
    g = cv.calcHist(im, [1], None, [bins], [0, 256])
    b = cv.calcHist(im, [2], None, [bins], [0, 256])
    vector = np.concatenate([r, g, b], axis=0)
    vector = vector.reshape(-1)
    return vector


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ch_analysis(label, bins):
    data_path = os.path.join(*["data", "clean", "train", label])
    im_names = os.listdir(data_path)

    vectors = []
    for i in im_names:
        im = cv.imread(os.path.join(data_path, i))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        v = get_vector(im, bins=bins)
        vectors.append(v)

    sums = []
    for v in vectors:
        similarities = []
        for i in vectors:
            similarities.append(cosine(v, i))
        sums.append(np.sum(similarities))

    df = pd.DataFrame({
        "image_name": im_names,
        "vector": vectors,
        "sum": sums
    })

    im_names = df.sort_values(by="sum", ascending=True).head(10)["image_name"]
    for i in im_names:
        im = cv.imread(os.path.join(data_path, i))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.show()
        plt.close()


if __name__ == "__main__":
    ch_analysis("63", 16)
