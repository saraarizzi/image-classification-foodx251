import ast
import os
import warnings
import zipfile

import numpy as np
import pandas as pd

from PIL import Image
import torch
from torchvision.transforms import v2

warnings.filterwarnings("error")

DATA_PATH = os.path.join(*["data", "clean", "train"])
LABELED_PATH = os.path.join("data", "manual_labelling")
REMOVED_PATH = os.path.join("data", "removed")

AUG_TECHS = [
    v2.RandomHorizontalFlip(p=1),
    v2.RandomVerticalFlip(p=1),
    v2.RandomErasing(p=1),
    v2.RandomRotation(degrees=45),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
]


def zip_data():
    with zipfile.ZipFile(os.path.join(*["data", "clean", "train.zip"]), "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk through the directory structure
        for root, dirs, files in os.walk(os.path.join(*["data", "clean", "train"])):
            for file in files:
                # Create the full filepath by joining the root with the file
                full_path = os.path.join(root, file)
                # Add file to zip, removing the base directory from the arc_name to maintain folder structure
                arc_name = os.path.relpath(full_path, start=DATA_PATH)
                zf.write(full_path, arcname=arc_name)


def get_cosine_similarity(a, b):
    try:
        cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cs
    except RuntimeWarning:
        return 0  # a and b are zero vectors


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


def get_first_quartile(df_similarities):
    df = df_similarities.copy(deep=True)
    # Calculate threshold
    q1 = np.percentile(df["sim"].to_list(), 25)

    # Images with similarity score lte than threshold
    df["predicted"] = df["sim"].apply(lambda x: 1 if x <= q1 else 0)

    return df


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


def downsize_class(class_code):

    folder = os.path.join(DATA_PATH, str(class_code))
    file_names = os.listdir(folder)
    rnd_to_remove = np.random.choice(file_names, size=(len(file_names)-350), replace=False)

    for image_name in rnd_to_remove:
        try:
            os.remove(os.path.join(folder, image_name))
        except FileNotFoundError:
            continue


def augment(class_code, img_name, technique):

    image = Image.open(os.path.join(*[DATA_PATH, str(class_code), img_name]))

    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224)),
        technique[0]
    ])

    transformed_image = transform(image)
    transformed_image_pil = v2.ToPILImage()(transformed_image)

    save_path = os.path.join(*[DATA_PATH, str(class_code), f"aug_{technique[0]._get_name()}_{img_name}"])
    transformed_image_pil.save(save_path)


def upsize_class(class_code):
    folder = os.path.join(DATA_PATH, str(class_code))
    file_names = os.listdir(folder)
    to_create = 350 - len(file_names)

    rnd_to_create = np.random.choice(file_names, size=to_create, replace=True)

    unique_transforms = []
    for img_name in rnd_to_create:

        technique = np.random.choice(AUG_TECHS, size=1)
        while (img_name, technique[0]._get_name()) in unique_transforms:
            technique = np.random.choice(AUG_TECHS, size=1)

        augment(class_code, img_name, technique)
        unique_transforms.append((img_name, technique[0]._get_name()))


def balance_train(df_train):
    """

    :param df_train: columns are code (int), name (str), samples (int)
    """

    removed = []
    added = []
    for row in df_train.iterrows():
        class_code = row[1]["code"]
        class_name = row[1]["name"]
        class_samples = row[1]["samples"]

        if class_samples > 350:
            downsize_class(class_code)
            print(f"Downsized class {class_name} ({class_code}): removed {class_samples-350} images")
            removed.append(class_samples-350)
            added.append(0)
        elif class_samples == 350:
            print(f"Class {class_name} ({class_code}): perfectly 350")
            removed.append(0)
            added.append(0)
        else:
            upsize_class(class_code)
            print(f"Upsized class {class_name} ({class_code}): added {350-class_samples} images")
            removed.append(0)
            added.append(350-class_samples)

    return removed, added


if __name__ == "__main__":

    os.makedirs(REMOVED_PATH, exist_ok=True)

    classes_data = pd.read_csv(os.path.join("data", "class_list.txt"), sep=" ", header=None)
    classes_data.columns = ["code", "label"]
    classes_codes = classes_data["code"].tolist()
    classes_names = classes_data["label"].tolist()

    # Remove train images marked as noise
    for c in classes_codes:
        vectors = get_fe_vectors(c, "imagenet")
        similarities = get_similarities(vectors)
        to_remove = get_first_quartile(similarities)

        remove_files(c, to_remove)

        print(f"Progressing ... {c+1}/251 ... Removed {len(to_remove[to_remove['predicted']==1])}/{len(to_remove)}")

    # Check samples distribution over classes
    num_samples = []
    for c in classes_codes:
        num_samples.append(len(os.listdir(os.path.join(DATA_PATH, str(c)))))

    df_samples = pd.DataFrame({
        "code": classes_codes,
        "name": classes_names,
        "samples": num_samples
    })

    # Keep 350 images for each class
    tot_removed, tot_added = balance_train(df_samples)

    pd.DataFrame({
        "code": classes_codes,
        "name": classes_names,
        "samples": num_samples,
        "removed": tot_removed,
        "added": tot_added
    }).to_csv(os.path.join(*["data", "clean", "balancing_info.csv"]), index=False)

    # Zip final data
    zip_data()
