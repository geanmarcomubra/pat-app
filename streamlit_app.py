import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile


# Configuración de la página
st.set_page_config(
    page_title="Detección de Patologías en Hormigón",
    layout="centered"
)

st.title("🧱 Detección de Patologías en Hormigón")
st.write(
    "Aplicación web para la detección automática de patologías "
    "en estructuras de hormigón utilizando un modelo YOLOv8."
)

# Cargar modelo SOLO una vez
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

modelo = cargar_modelo()

# Subir imagen
imagen_subida = st.file_uploader(
    "📷 Suba una imagen de la estructura de hormigón",
    type=["jpg", "jpeg", "png"]
)

# Umbral de confianza
umbral = st.slider(
    "🔧 Umbral de confianza",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")

    st.image(
        imagen,
        caption="Imagen cargada",
        width="stretch"
    )

    # Inferencia
    resultados = modelo.predict(imagen, conf=umbral)
    resultado = resultados[0]

    st.subheader("🧪 Resultado del análisis")

    if len(resultado.boxes) == 0:
        st.success("✅ No se detectaron patologías visibles en la imagen.")
    else:
        st.warning("⚠ Patologías detectadas:")

        for box in resultado.boxes:
            clase_id = int(box.cls[0])
            clase_nombre = modelo.names[clase_id]
            confianza = float(box.conf[0])

            st.write(
                f"• **{clase_nombre}** — Confianza: **{confianza:.2%}**"
            )

        # Imagen con detecciones
        img_resultado = resultado.plot()

        st.image(
            img_resultado,
            caption="Resultado con detecciones",
            width="stretch"
        )
