import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Configuración de diseño de la página
st.set_page_config(page_title="Detector de Frutas Inteligente", page_icon="🥝", layout="centered")

st.title("🥝 Clasificador de Frutas con Inteligencia Artificial")
st.write("Sube la fotografía de una fruta para obtener la predicción del modelo.")

# Cargar el modelo guardado y las etiquetas usando caché para evitar recargas lentas
@st.cache_resource
def cargar_recursos():
    modelo = tf.keras.models.load_model('modelo_frutas.h5')
    with open('labels.json', 'r') as f:
        etiquetas = json.load(f)
    return modelo, etiquetas

try:
    model, labels = cargar_recursos()
    st.success("Modelo e índices de clases cargados con éxito.")
except Exception as e:
    st.error("Error al cargar los archivos del modelo. Asegúrate de haber ejecutado 'train.py' primero.")
    st.stop()

# Selector de archivos para el usuario
uploaded_file = st.file_uploader("Selecciona una imagen en formato JPG, JPEG o PNG...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen que subió el usuario
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen del usuario', use_container_width=True)
    
    st.write("Procesando predicción...")
    
    # Preprocesamiento idéntico al entrenamiento (Tamaño 100x100)
    img = image.convert('RGB').resize((100, 100))
    img_array = np.array(img) / 255.0  # Normalización
    img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de lote
    
    # Predicción
    predictions = model.predict(img_array)
    
    id_clase = str(np.argmax(predictions))
    fruta_detectada = labels[id_clase]
    porcentaje_confianza = np.max(predictions) * 100
    
    # Despliegue de resultados en pantalla
    st.subheader(f"Resultado: **{fruta_detectada}**")
    st.metric(label="Confianza de la predicción", value=f"{porcentaje_confianza:.2f}%")
    st.progress(int(porcentaje_confianza))
