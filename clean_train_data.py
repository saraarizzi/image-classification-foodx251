import ast
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("error")

DATA_PATH = os.path.join(*["data", "clean", "train"])
LABELED_PATH = os.path.join("data", "manual_labelling")
REMOVED_PATH = os.path.join("data", "removed")


def get_cosine_similarity(a, b):
    try:
        cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cs
    except RuntimeWarning:
        return 0  # a and b are zero vectors


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


def get_bovw_vectors(code, dim):
    if dim not in [10, 20, 40, 80]:
        raise Exception("Dim must be 10, 20, 40, or 80!")

    bovw_path = os.path.join(*["data", "bovw", str(code)])
    bovw_class = pd.read_csv(os.path.join(bovw_path, f"tfidf_{dim}.csv"), sep=";", header=0)
    bovw_class["tfidf"] = bovw_class["tfidf"].apply(
        lambda x: np.array(ast.literal_eval(x.replace('\n', '').replace('  ', ' ').strip())))

    # Return all images bovw
    return bovw_class


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
    if len(df_vectors.columns) == 2:
        df_vectors.columns = ["image_name", "vector"]
    else:
        df_vectors.columns = ["image_name", "d", "vector"]

    images_names = df_vectors["image_name"].to_list()
    vecs = df_vectors["vector"].to_list()

    # Loop over images and calculate cosine similarity between all
    all_sim = []
    for i in vecs:
        image_sim = 0
        for j in vecs:
            cosine_sim = get_cosine_similarity(i, j)
            image_sim += cosine_sim
        all_sim.append(image_sim / len(vecs))

    df_similarities = pd.DataFrame({
        "image_name": images_names,
        "sim": all_sim
    })

    return df_similarities


def get_outliers(df_similarities):
    df = df_similarities.copy(deep=True)
    # Calculate threshold
    q1 = np.percentile(df["sim"].to_list(), 25)
    iqr = np.percentile(df["sim"].to_list(), 75) - q1
    threshold = q1 - 1.5 * iqr

    # Images with similarity score lte than threshold
    df["predicted"] = df["sim"].apply(lambda x: 1 if x <= threshold else 0)

    return df


def get_first_quartile(df_similarities):
    df = df_similarities.copy(deep=True)
    # Calculate threshold
    q1 = np.percentile(df["sim"].to_list(), 25)

    # Images with similarity score lte than threshold
    df["predicted"] = df["sim"].apply(lambda x: 1 if x <= q1 else 0)

    return df


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


def remove_files(code, df_remove):

    folder = os.path.join(DATA_PATH, str(code))

    df_to_remove = df_remove[df_remove["predicted"] == 1]["image_name"]

    # Save files to remove
    df_to_remove.to_csv(os.path.join(REMOVED_PATH, f"{code}_removed.csv"), index=False)

    list_to_remove = df_to_remove.to_list()

    for image_name in list_to_remove:
        try:
            os.remove(os.path.join(folder, image_name))
        except FileNotFoundError:
            continue


if __name__ == "__main__":

    os.makedirs(REMOVED_PATH, exist_ok=True)

    classes_data = pd.read_csv(os.path.join("data", "class_list.txt"), sep=" ", header=None)
    classes_data.columns = ["code", "label"]
    classes_codes = classes_data["code"].tolist()

    for c in classes_codes:
        vectors = get_fe_vectors(c, "imagenet")
        similarities = get_similarities(vectors)
        to_remove = get_first_quartile(similarities)

        remove_files(c, to_remove)

        print(f"Progressing ... {c+1}/251 ... Removed {len(to_remove[to_remove['predicted']==1])}/{len(to_remove)}")

