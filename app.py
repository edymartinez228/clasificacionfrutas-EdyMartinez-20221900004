import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =======================
# CARGAR MODELO
# =======================

model=tf.keras.models.load_model("modelo_frutas.keras")

class_names=np.load("class_names.npy",allow_pickle=True)

st.set_page_config(page_title="Clasificador de Frutas",layout="centered")

st.title("🍎 Clasificador de Frutas con IA")

st.write("Sube una imagen de una fruta.")

archivo=st.file_uploader(
    "Selecciona una imagen",
    type=["jpg","jpeg","png"]
)

if archivo is not None:

    imagen=Image.open(archivo).convert("RGB")

    st.image(imagen,width=250)

    imagen=imagen.resize((100,100))

    img=np.array(imagen)

    img=np.expand_dims(img,axis=0)

    img=tf.keras.applications.mobilenet_v2.preprocess_input(img)

    pred=model.predict(img)

    indice=np.argmax(pred)

    confianza=np.max(pred)*100

    st.success(f"Fruta: **{class_names[indice]}**")

    st.write(f"Confianza: **{confianza:.2f}%**")
