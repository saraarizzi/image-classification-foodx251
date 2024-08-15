import os

import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

from numpy.linalg import norm

"""

def show_5_ex(tfidf_vectors, kind="top"):

    # get search image vector
    a = tfidf_vectors[0]
    b = tfidf_vectors  # set search space to the full sample
    # get the cosine distance for the search image `a`
    cosine_similarity = np.dot(a, b.T) / (norm(a) * norm(b, axis=1))

    # get the top k indices for most similar vecs
    if kind == "top":
        k_idx = np.argsort(-cosine_similarity)[:5]
    elif kind == "last":
        k_idx = np.argsort(-cosine_similarity)[-5:]
    else:
        return "Kind must be 'top' or 'last'."

    # display the results
    for i in k_idx:
        print(f"{i}: {round(cosine_similarity[i], 4)}")
        image = cv.imread(os.path.join(DATA_PATH, images_names[i]))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

# Show 5 most and least similar images (cosine similarity)
# show_5_ex(images_tfidf, kind="top")

# show_5_ex(images_tfidf, kind="last")

if show_examples:
    # examples = np.random.choice(range(len(images_names)), size=10)
    examples = range(10)
    for idx in examples:
        plt.subplots(1, 2)

        # Image
        plt.subplot(1, 2, 1)
        img = cv.imread(os.path.join(DATA_PATH, images_names[idx]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        # TFIDF
        plt.subplot(1, 2, 2)
        plt.bar(list(range(K)), images_tfidf[idx])
        plt.title(f"TF-IDF {K}-dim")

        plt.suptitle(classes_data[classes_data["code"] == class_code].label.item().replace("_", " "))
        plt.show()
        plt.close()
        
        
        
if __name__ == "__main__":

    start_time = time.time()
    print("Started at:", time.strftime(DATE_FORMAT))

    show_examples = False
    classes_data = pd.read_csv("data/class_list.txt", sep=" ", header=None)
    classes_data.columns = ["code", "label"]

    K = 10
    for idx, class_code in enumerate(classes_data.code):
        print(f"Progressing ... {(idx + 1)}/251", time.strftime(DATE_FORMAT))

        BOVW_PATH = os.path.join(*["data", "bovw", f"{K}_vw", f"{class_code}"])
        os.makedirs(BOVW_PATH, exist_ok=True)

        DATA_PATH = os.path.join(*["data", "clean", "train", f"{class_code}"])
        images_names = os.listdir(DATA_PATH)

        # Get descriptors for all the images in the class
        images_desc, images_to_remove = get_desc(images_names)
        for im in images_to_remove:
            images_names.remove(im)

        # Create visual words
        codebook = get_codebook(images_desc, k=K)

        # Visual words vector
        images_vw = get_vw_vectors(images_desc, codebook)

        # Create tf vectors
        images_tf = get_tf_vectors(images_vw, k=K)

        # Created tf-idf vectors
        images_tfidf = get_tfidf_vectors(images_tf)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script total duration: {duration:.2f} seconds")

    print("Done :)")
        
"""