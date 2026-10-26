"""Plant disease identification app. Run with: streamlit run app.py"""
import torch
import streamlit as st
from PIL import Image

from app_model import BEST_MODEL_NAME, CHECKPOINT_PATH, INDEX_PATH, NORM, image_features, load_model, predict_scores
from common.config import IMAGE_DIR
from common.dataset import get_transforms

TOP_PAIRS = 5
TOP_SIMILAR = 5

st.set_page_config(page_title="Plant Disease Identifier", page_icon="🌿", layout="wide")


@st.cache_resource
def load_resources():
    """Load the model and training-image index once, not on every click."""
    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=False)
    return load_model(), index


def pretty(name: str) -> str:
    return str(name).replace("___", " | ").replace("_", " ")


def show_top_pairs(scores, index):
    """Top-5 crop-disease pairs with confidence in percent."""
    crop_names, disease_names = index["crop_and_disease_names"]
    confidence = scores.softmax(dim=0)
    top = confidence.topk(TOP_PAIRS)
    st.subheader("Top 5 predictions")
    for rank, (share, pair_row) in enumerate(zip(top.values.tolist(), top.indices.tolist()), start=1):
        crop_id, disease_id = index["pairs"][pair_row].tolist()
        tag = "seen in training" if pair_row < index["num_seen_pairs"] else "UNSEEN pair (zero-shot)"
        st.markdown(f"**{rank}. {pretty(crop_names[crop_id])} – {pretty(disease_names[disease_id])}**  "
                    f"<small>({tag})</small>", unsafe_allow_html=True)
        st.progress(share, text=f"{100 * share:.1f}%")
    return top.indices[0].item()


def show_similar_images(features, index):
    """Training images closest to the upload by cosine similarity of backbone features."""
    similarity = index["features"].float() @ features
    top = similarity.topk(TOP_SIMILAR)
    st.subheader("Most similar training images")
    columns = st.columns(TOP_SIMILAR)
    for column, score, image_number in zip(columns, top.values.tolist(), top.indices.tolist()):
        pair_name = pretty(index["pair_names"][index["pair_rows"][image_number].item()])
        image_path = IMAGE_DIR / index["image_files"][image_number]
        with column:
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.info("Image file not found locally")
            st.caption(f"{pair_name}\n\nsimilarity {score:.2f}")


def main():
    st.title("🌿 Plant Disease Identifier")
    st.write("Upload a leaf photo. The model names the crop and the disease, "
             "including crop-disease pairs it never saw during training.")
    st.sidebar.header("About")
    st.sidebar.write(f"Model: `{BEST_MODEL_NAME}`")
    st.sidebar.write("Dataset: DPD-Small (323 seen pairs, 10 unseen pairs)")
    st.sidebar.write("COS30082 Applied Machine Learning")

    if not CHECKPOINT_PATH.exists() or not INDEX_PATH.exists():
        st.error(f"Missing files. Put the weights in `{CHECKPOINT_PATH.name}` and run "
                 "`python build_index.py` first. See the README.")
        st.stop()
    model, index = load_resources()

    upload = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        return
    image = Image.open(upload).convert("RGB")
    left, right = st.columns([1, 2])
    left.image(image, caption="Your photo", width="stretch")
    if not right.button("Predict", type="primary"):
        return

    _, test_transform = get_transforms(NORM)  # exactly the transform used in testing
    batch = test_transform(image).unsqueeze(0)
    with st.spinner("Identifying..."):
        scores = predict_scores(model, batch, index["pairs"])[0]
        features = image_features(model, batch)[0]
    with right:
        show_top_pairs(scores, index)
    show_similar_images(features, index)


main()
