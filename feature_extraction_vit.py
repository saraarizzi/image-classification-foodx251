import multiprocessing
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTModel, ViTConfig
import torch.nn as nn


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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


def read_model():
    # Load local files
    pr = AutoImageProcessor.from_pretrained("models")
    m = AutoModelForImageClassification.from_pretrained("models")

    # Add pooler
    new_model = ViTModel(config=m.config)
    new_model.embeddings = m.base_model.embeddings
    new_model.encoder = m.base_model.encoder
    new_model.layernorm = m.base_model.layernorm
    new_model.pooler = ViTPooler(config=m.config)  # no weights

    return new_model, pr


def process_class(class_code):
    s_time = time.time()
    print(f"Start Processing ... {class_code}")
    fe_path = os.path.join(*["data", "fe", f"{class_code}"])
    os.makedirs(fe_path, exist_ok=True)

    data_path = os.path.join(*["data", "clean", "train", f"{class_code}"])
    image_info_path = os.path.join(*["data", "raw", "train_info_dirty.csv"])
    image_data = pd.read_csv(image_info_path, header=None, names=["file", "class"])
    images_names = image_data[image_data["class"] == class_code]["file"].to_list()

    # Load model
    model, processor = read_model()

    features_list = []
    for image_name in images_names:
        image = Image.open(os.path.join(data_path, image_name))

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")

        # Inference
        outputs = model(**inputs).pooler_output

        features = outputs.detach().numpy().reshape(-1)  # 768-dim
        features_list.append(features)

    pd.DataFrame({
        "image_name": images_names,
        "vit_feature": list(
            map(lambda x: np.array2string(x, separator=",", formatter={'float_kind': lambda num: "%.6f" % num}),
                features_list)
        )
    }).to_csv(os.path.join(fe_path, f"vit.csv"), sep=";", index=False)

    print(f"End Processing ... {class_code} ---> Duration {(time.time() - s_time):.2f} seconds")
    return


if __name__ == "__main__":
    # Download from HF
    hf_pre = AutoImageProcessor.from_pretrained("nateraw/food")
    hf_mod = AutoModelForImageClassification.from_pretrained("nateraw/food")

    # Save locally
    os.makedirs(os.path.join("data", "hf_vit_food"), exist_ok=True)
    hf_pre.save_pretrained(os.path.join("data", "hf_vit_food"))
    hf_mod.save_pretrained(os.path.join("data", "hf_vit_food"))

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
