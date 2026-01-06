import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# -------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------
st.set_page_config(
    page_title="Detección de Patologías en Hormigón",
    layout="wide"
)

st.title("🧱 Detección de Patologías en Hormigón")
st.write(
    "Aplicación web para la detección automática de patologías en elementos "
    "de hormigón utilizando un modelo YOLOv8."
)

# -------------------------------
# CARGA DEL MODELO
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# SUBIDA DE IMAGEN
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Sube una imagen del elemento de hormigón",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Abrir imagen
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Imagen original")
    st.image(image)  # ⬅️ SIN width

    # Guardar imagen temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # -------------------------------
    # INFERENCIA
    # -------------------------------
    with st.spinner("🔍 Analizando imagen..."):
        results = model(temp_image_path)

    # Imagen con detecciones (numpy array)
    result_img = results[0].plot()

    st.subheader("Resultado de la detección")
    st.image(result_img)  # ⬅️ SIN width

    # -------------------------------
    # RESULTADOS TEXTUALES
    # -------------------------------
    st.subheader("Diagnóstico")

    if len(results[0].boxes) == 0:
        st.success("✅ No se detectaron patologías en la imagen.")
    else:
        clases = results[0].names
        detectadas = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            detectadas.add(clases[cls_id])

        st.warning("⚠️ Patologías detectadas:")
        for d in detectadas:
            st.write(f"- {d}")

    # Limpiar archivo temporal
    os.remove(temp_image_path)

else:
    st.info("⬆️ Por favor, sube una imagen para iniciar el análisis.")
