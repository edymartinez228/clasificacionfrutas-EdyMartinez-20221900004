import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Clasificador de Frutas", layout="centered")

st.title("Clasificador de Frutas - Inteligencia_Artificial - Edy Martinez - 20221900004")
st.write("Sube una imagen para que el modelo MobileNetV2 identifique a qué categoría pertenece.")

# ----------------------------
# CARGAR EL MODELO ENTRENADO
# ----------------------------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_frutas.keras")

modelo = cargar_modelo()

# Lista en orden alfabético estricto
clases_lista = [
    "Almonds 1", "Apple Red Yellow 1", "Avocado Green 1", "Banana 1", 
    "Beetroot 1", "Blackberry 3", "Cabbage white 1", "Cactus fruit green 1", 
    "Cantaloupe 3", "Carambola 1", "Tomato 7"
]

# Diccionario de traducción al español
traduccion_frutas = {
    "Almonds 1": "Almendras",
    "Apple Red Yellow 1": "Manzana Roja Amarilla",
    "Avocado Green 1": "Aguacate Verde",
    "Banana 1": "Banano",
    "Beetroot 1": "Remolacha",
    "Blackberry 3": "Mora",
    "Cabbage white 1": "Repollo Blanco",
    "Cactus fruit green 1": "Tuna Verde (Fruto del Cactus)",
    "Cantaloupe 3": "Melón",
    "Carambola 1": "Carambola (Fruta Estrella)",
    "Tomato 7": "Tomate"
}

# ----------------------------
# INTERFAZ DE CARGA DE ARCHIVOS
# ----------------------------
archivo_subido = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido)
    st.image(imagen_pil, caption="Imagen subida", use_container_width=True)
    
    st.write("Clasificando...")
    
    # ----------------------------
    # PREPROCESAMIENTO DE IMAGEN
    # ----------------------------
    imagen_rgb = imagen_pil.convert("RGB")
    img_array = np.array(imagen_rgb)
    img_redimensionada = cv2.resize(img_array, (100, 100))
    img_normalizada = img_redimensionada.astype(float) / 255.0
    input_tensor = img_normalizada.reshape(-1, 100, 100, 3)
    
    # ----------------------------
    # PREDICCIÓN Y TRADUCCIÓN
    # ----------------------------
    prediccion = modelo.predict(input_tensor)
    indice_maximo = np.argmax(prediccion[0], axis=-1)
    porcentaje_confianza = prediccion[0][indice_maximo] * 100
    
    fruta_ingles = clases_lista[indice_maximo]
    fruta_espanol = traduccion_frutas.get(fruta_ingles, fruta_ingles)
    
    # Mostrar resultados estilizados en español
    st.success(f"**Predicción:** {fruta_espanol}")
    st.info(f"**Confianza del modelo:** {porcentaje_confianza:.2f}%")
