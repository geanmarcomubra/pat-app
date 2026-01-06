import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Detección de Patologías en Hormigón",
    layout="wide"
)

st.title("🧱 Detección de Patologías en Hormigón")
st.write(
    "Sistema web para la detección automática de patologías en elementos "
    "de hormigón mediante visión artificial."
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "📤 Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Imagen original")
    st.image(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    with st.spinner("🔍 Analizando imagen..."):
        results = model.predict(
            source=img_path,
            conf=0.25,
            imgsz=640,
            device="cpu"
        )

    result = results[0]
    result_img = result.plot()

    st.subheader("Resultado de la detección")
    st.image(result_img)

    st.subheader("Diagnóstico")

    if result.boxes is None or len(result.boxes) == 0:
        st.success("✅ No se detectaron patologías en la imagen.")
    else:
        names = result.names
        detected = set(int(b.cls[0]) for b in result.boxes)

        st.warning("⚠️ Patologías detectadas:")
        for d in detected:
            st.write(f"- {names[d]}")

    os.remove(img_path)

else:
    st.info("⬆️ Sube una imagen para iniciar el análisis.")
