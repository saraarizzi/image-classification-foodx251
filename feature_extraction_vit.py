import multiprocessing
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor
from transformers import ViTModel, ViTConfig
import torch.nn as nn


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
HF_PATH = os.path.join("data", "hf")


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def read_model(dataset):
    if dataset == "food":
        # Load from local folder
        pr = AutoImageProcessor.from_pretrained(os.path.join(HF_PATH, "food"))
        m = AutoModelForImageClassification.from_pretrained(os.path.join(HF_PATH, "food"))

        # Rewrite model as original ViT architecture
        new_model = ViTModel(config=m.config)
        new_model.embeddings = m.base_model.embeddings
        new_model.encoder = m.base_model.encoder
        new_model.layernorm = m.base_model.layernorm
        new_model.pooler = ViTPooler(config=m.config)

        return new_model, pr

    elif dataset == "imagenet":
        # Load from local folder
        pr = ViTImageProcessor.from_pretrained(os.path.join(HF_PATH, "imagenet"))
        m = ViTModel.from_pretrained(os.path.join(HF_PATH, "imagenet"))

        return m, pr

    raise Exception("Param must be in ['food', 'imagenet']")


def process_class(class_code):
    print(f"Start Processing ... {class_code}")
    fe_path = os.path.join(*["data", "fe", f"{class_code}"])
    os.makedirs(fe_path, exist_ok=True)

    data_path = os.path.join(*["data", "clean", "train", f"{class_code}"])
    image_info_path = os.path.join(*["data", "raw", "train_info_dirty.csv"])
    image_data = pd.read_csv(image_info_path, header=None, names=["file", "class"])
    images_names = image_data[image_data["class"] == class_code]["file"].to_list()

    # Load models
    model_food, processor_food = read_model("food")
    model_imagenet, processor_imagenet = read_model("imagenet")

    features_list_food = []
    features_list_imagenet = []
    for image_name in images_names:
        image = Image.open(os.path.join(data_path, image_name))

        # Preprocess
        inputs_food = processor_food(images=image, return_tensors="pt")
        inputs_imagenet = processor_imagenet(images=image, return_tensors="pt")

        # Inference
        outputs_food = model_food(**inputs_food).pooler_output
        outputs_imagenet = model_imagenet(**inputs_imagenet).pooler_output

        # Save Food-101 features
        features_food = outputs_food.detach().numpy().reshape(-1)  # 768-dim
        features_list_food.append(features_food)

        # Save ImageNet21k features
        features_imagenet = outputs_imagenet.detach().numpy().reshape(-1)  # 768-dim
        features_list_imagenet.append(features_imagenet)

    pd.DataFrame({
        "image_name": images_names,
        "vit_feature": list(
            map(
                lambda x: np.array2string(x, separator=",", formatter={'float_kind': lambda num: "%.6f" % num}),
                features_list_imagenet
            )
        )
    }).to_csv(os.path.join(fe_path, f"vit_food.csv"), sep=";", index=False)

    pd.DataFrame({
        "image_name": images_names,
        "vit_feature": list(
            map(
                lambda x: np.array2string(x, separator=",", formatter={'float_kind': lambda num: "%.6f" % num}),
                features_list_imagenet
            )
        )
    }).to_csv(os.path.join(fe_path, f"vit_imagenet.csv"), sep=";", index=False)


def download_model(processor, model, name, folder):
    # Download from HF
    hf_pre = processor.from_pretrained(name)
    hf_mod = model.from_pretrained(name)

    # Save locally
    os.makedirs(os.path.join(HF_PATH, folder), exist_ok=True)
    hf_pre.save_pretrained(os.path.join(HF_PATH, folder))
    hf_mod.save_pretrained(os.path.join(HF_PATH, folder))


if __name__ == "__main__":

    # ViT trained on Food-101
    download_model(
        AutoImageProcessor, AutoModelForImageClassification, "nateraw/food", "food"
    )

    # ViT trained on ImageNet-21k
    download_model(
        ViTImageProcessor, ViTModel, "google/vit-base-patch16-224-in21k", "imagenet"
    )

    # Start processing
    start_time = time.time()
    print(f"Started at: {time.strftime(DATE_FORMAT)}")

    classes_data = pd.read_csv(os.path.join("data", "class_list.txt"), sep=" ", header=None)
    classes_data.columns = ["code", "label"]
    classes_codes = classes_data.code.tolist()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_class, classes_codes)

    print(f"Ended at: {time.strftime(DATE_FORMAT)}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script total duration: {duration:.2f} seconds")
