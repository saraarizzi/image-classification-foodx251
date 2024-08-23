import ast
import os

import numpy as np
import pandas as pd

CH_PATH = os.path.join("data", "ch")
LABELED_PATH = os.path.join("data", "manual_labelling")


def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_ch_vectors(cl_code, bins):
    if bins not in [8, 16, 32, 64, 128, 256]:
        raise Exception("Bins must be 8, 16, 32, 64, 128, or 256!")

    ch_path = os.path.join(CH_PATH, str(cl_code))
    ch_class = pd.read_csv(os.path.join(ch_path, f"ch_{bins}.csv"), sep=";", header=0)
    ch_class["hist"] = ch_class["hist"].apply(
        lambda x: np.array(ast.literal_eval(x.replace('\n', '').replace('  ', ' ').strip())))

    # Scale into 0-1 by dividing by max value
    maxes = ch_class["hist"].apply(lambda x: max(x))
    ch_class["hist"] = ch_class["hist"] / maxes

    return ch_class


def get_fe_vectors(cl_code, data):
    fe_path = os.path.join(*["data", "fe", str(cl_code)])
    fe_class = pd.read_csv(os.path.join(fe_path, f"vit_{data}.csv"), sep=";", header=0)
    fe_class["vit_feature"] = fe_class["vit_feature"].apply(
        lambda x: np.array(ast.literal_eval(x.replace('\n', '').replace('  ', ' ').strip())))

    # Return all images bovw
    return fe_class


def get_similarities(df_vectors):
    """

    :param df_vectors: contains images names and vector representations
    :type df_vectors: pd.DataFrame
    """
    df_vectors.columns = ["image_name", "vector"]
    images_names = df_vectors["image_name"].to_list()
    vectors = df_vectors["vector"].to_list()

    # Loop over images and calculate cosine similarity between all
    all_sim = []
    for i in vectors:
        image_sim = 0
        for j in vectors:
            image_sim += get_cosine_similarity(i, j)
        all_sim.append(image_sim / len(vectors))

    df_similarities = pd.DataFrame({
        "image_name": images_names,
        "sim": all_sim
    })

    return df_similarities


def get_similarities_chat(df_vectors):
    """

    :param df_vectors: contains images names and vector representations
    :type df_vectors: pd.DataFrame
    """

    df_vectors.columns = ["image_name", "vector"]
    images_names = df_vectors["image_name"].to_list()
    vectors = df_vectors["vector"].to_list()

    vectors = np.array(vectors)

    # Normalize the vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Compute the cosine similarity matrix (dot product)
    cosine_sim_matrix = np.dot(normalized_vectors, normalized_vectors.T)

    # Sum the similarities for each vector
    all_sim = np.sum(cosine_sim_matrix, axis=1) / len(vectors)

    df_similarities = pd.DataFrame({
        "image_name": images_names,
        "sim": all_sim
    })

    return df_similarities


def get_outliers(df_similarities):
    # Calculate threshold
    q1 = np.percentile(df_similarities["sim"].to_list(), 25)
    iqr = np.percentile(df_similarities["sim"].to_list(), 75) - q1
    threshold = q1 - 1.5 * iqr

    # Images with similarity score lte than threshold
    df_similarities["predicted"] = df_similarities["sim"].apply(lambda x: 1 if x <= threshold else 0)

    return df_similarities


def get_first_quartile(df_similarities):
    # Calculate threshold
    q1 = np.percentile(df_similarities["sim"].to_list(), 25)

    # Images with similarity score lte than threshold
    df_similarities["predicted"] = df_similarities["sim"].apply(lambda x: 1 if x <= q1 else 0)

    return df_similarities


def check_accuracy(cl_code, to_rem):

    label_path = os.path.join(LABELED_PATH, f"{cl_code}_to_remove.csv")

    labeled = pd.read_csv(label_path, usecols=range(1, 3))

    # Merge df1 and df2 on 'image_name'
    merged_df = pd.merge(labeled, to_rem, on='image_name')

    actual_positive = merged_df[merged_df["to_remove"] == 1]
    actual_negative = merged_df[merged_df["to_remove"] == 0]

    # Precision and False Positive Rate
    p = np.sum(actual_positive['to_remove'] == actual_positive['predicted']) / len(actual_positive)
    f = np.sum(actual_negative['to_remove'] != actual_negative['predicted']) / len(actual_negative)

    return p, f


if __name__ == "__main__":

    classes = [196, 15, 0, 233, 57, 25]
    classes_names = ["omelette", "seaweed_salad", "macaron", "scotch_egg", "gnocchi", "ramen"]

    for c, c_name in zip(classes, classes_names):
        # vectors = get_ch_vectors(c, bins=256)
        vectors = get_fe_vectors(c, "imagenet")
        similarities = get_similarities(vectors)
        to_remove = get_first_quartile(similarities)

        # print(f"For class {c_name}, to remove {len(to_remove[to_remove['predicted']==1])} out of {len(to_remove)}.")

        # Check correctness
        precision, fpr = check_accuracy(c, to_remove)

        print(f"Precision {np.round(precision*100, 2)}%"
              f"\t FPR {np.round(fpr*100, 2)}%"
              f"\t --> class {c_name} ")
