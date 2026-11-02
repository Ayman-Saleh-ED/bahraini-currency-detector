import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Bahraini Currency Detector", page_icon="💵")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


def pretty(name):
    # class names look like "0_5", "1", "10" -> "0.5 BHD", "1 BHD", "10 BHD"
    return name.replace("_", ".") + " BHD"


model = load_model()

st.title("Bahraini Currency Detector")
st.write("Upload a photo of Bahraini banknotes and the model will find and label each one.")

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.caption("Higher = fewer detections, but more certain ones.")

file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    results = model(image, conf=conf)
    r = results[0]

    annotated = r.plot()[..., ::-1]  # BGR -> RGB

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(image, use_container_width=True)
    col2.subheader("Detections")
    col2.image(annotated, use_container_width=True)

    if len(r.boxes) == 0:
        st.warning("No currency detected. Try a clearer photo or lower the confidence threshold.")
    else:
        st.subheader(f"Found {len(r.boxes)} note(s)")
        for box in r.boxes:
            name = pretty(r.names[int(box.cls)])
            st.write(f"**{name}** — confidence {float(box.conf):.0%}")
