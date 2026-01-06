import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -------------------------------
# CONFIGURACIÓN DE PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Detección de Patologías en Hormigón",
    layout="wide"
)

st.title("🧱 Detección de Patologías en Hormigón")
st.write(
    "Aplicación web para detectar patologías en elementos de hormigón "
    "utilizando un modelo YOLOv8 entrenado."
)

# -------------------------------
# CARGA DEL MODELO
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt", task="detect")

model = load_model()

# -------------------------------
# SUBIDA DE IMAGEN
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Sube una imagen del elemento de hormigón",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Imagen original")
    st.image(image)

    # Guardar imagen temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # -------------------------------
    # INFERENCIA (FORMA SEGURA)
    # -------------------------------
    with st.spinner("🔍 Analizando imagen..."):
        results = model.predict(
            source=temp_image_path,
            conf=0.25,
            save=False
        )

    result = results[0]
    result_img = result.plot()

    st.subheader("Resultado de la detección")
    st.image(result_img)

    # -------------------------------
    # DIAGNÓSTICO
    # -------------------------------
    st.subheader("Diagnóstico")

    if result.boxes is None or len(result.boxes) == 0:
        st.success("✅ No se detectaron patologías en la imagen.")
    else:
        class_names = result.names
        detected = set()

        for box in result.boxes:
            cls_id = int(box.cls[0])
            detected.add(class_names[cls_id])

        st.warning("⚠️ Patologías detectadas:")
        for d in detected:
            st.write(f"- {d}")

    os.remove(temp_image_path)

else:
    st.info("⬆️ Por favor, sube una imagen para iniciar el análisis.")
