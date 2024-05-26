import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import streamlit as st
import streamlit_authenticator as stauth

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- SETTING CONFIGURATION ---
st.set_page_config(page_title="Recipes Project")

# --- USER AUTHENTICATION ---
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names,
    usernames,
    hashed_passwords,
    "recipes_project",
    "adcdef",
    cookie_expiry_days=30,
)

name, authenticaton_status, username = authenticator.login("Login", "main")

# --- DETECTION MODEL  ---
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()


@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()

    return model


obj_model = load_model()


def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = obj_model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction


def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=[
            "red" if label == "person" else "green" for label in prediction["labels"]
        ],
        width=2,
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np


# --- GENERATIVE MODEL  ---
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


def generate_recipe(ingredients):
    input_text = (
        "Your job is to assist a user with cooking some meal for a dinner. Here are some ingredients, that he has:"
        + ", ".join(ingredients)
        + ". Provide at least 3 different recipes for the dinner with given ingredients. "
        + "Make sure that recipes are full and detailed"
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = gen_model.generate(input_ids, max_new_tokens=200)
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return recipe


if authenticaton_status is False:
    st.error("Username/password is incorrect")
if authenticaton_status is None:
    st.warning("Please enter your username and password")
if authenticaton_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    st.title("Object Detector :tea: :coffee:")
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload is not None:
        img = Image.open(upload)
        img.save("food.jpg")

        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate"):

            prediction = make_prediction(img)
            img_with_bbox = create_image_with_bboxes(
                np.array(img).transpose(2, 0, 1), prediction
            )

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([], [])
            plt.yticks([], [])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            del prediction["boxes"]
            st.header("Predicted Probabilities")
            st.write([x for x in prediction["labels"]])

            ingredients = prediction["labels"]
            recipe = generate_recipe(ingredients)
            st.write(recipe)
