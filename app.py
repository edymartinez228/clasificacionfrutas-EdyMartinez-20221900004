import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Configuración de la interfaz web (Debe ser la primera instrucción de Streamlit)
st.set_page_config(page_title="Clasificador de Frutas IA", page_icon="🍎", layout="centered")

st.title("🍎 Clasificador de Frutas con Inteligencia Artificial 🍌")
st.write("Sube la foto de una fruta y el modelo compacto de Deep Learning te dirá qué fruta es.")

# 2. Función optimizada para cargar el modelo compacto y las etiquetas
@st.cache_resource
def cargar_modelo():
    # Carga el archivo del modelo liviano (.h5 de menos de 10MB)
    modelo_ia = tf.keras.models.load_model('modelo_frutas.h5')
    
    # Lee las etiquetas generadas por el train.py compacto
    lista_clases = []
    with open("clases.txt", "r") as f:
        for linea in f.readlines():
            linea_limpia = linea.strip()
            if linea_limpia:
                lista_clases.append(linea_limpia)
                
    # Aseguramos el orden alfabético estricto coincidente con el entrenamiento
    lista_clases = sorted(lista_clases)
    return modelo_ia, lista_clases

# Inicializamos variables globales para mitigar NameErrors en fallas de carga
modelo = None
clases = []

try:
    modelo, clases = cargar_modelo()
    st.success(f"¡Modelo de IA cargado con éxito! ({len(clases)} clases sincronizadas)")
except Exception as e:
    st.error(f"❌ Error crítico al inicializar la IA: {e}")
    st.info("Verifica que los archivos 'modelo_frutas.h5' y 'clases.txt' estén en la raíz de tu GitHub.")

# 3. Módulo de carga de archivos
archivo_subido = st.file_uploader("Elige una imagen de una fruta (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Desplegar la imagen en la app
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Imagen subida por el usuario', use_container_width=True)
    
    st.write("🔍 Extrayendo características y clasificando...")
    
    if modelo is None:
        st.error("El modelo de IA no está disponible en el servidor.")
    else:
        try:
            # 4. Preprocesamiento Estricto para el Modelo Compacto (64x64)
            imagen_rgb = imagen.convert("RGB")
            
            # Redimensionamos a 64x64 de forma exacta para coincidir con la nueva arquitectura
            imagen_redimensionada = imagen_rgb.resize((64, 64))
            imagen_array = np.array(imagen_redimensionada)
            
            # Normalización de píxeles al rango [0, 1]
            imagen_array = imagen_array / 255.0
            
            # Expandimos dimensiones para simular el lote -> Forma final: (1, 64, 64, 3)
            imagen_array = np.expand_dims(imagen_array, axis=0)
            
            # 5. Ejecución de la Inferencia
            predicciones = modelo.predict(imagen_array)
            indice_predicho = np.argmax(predicciones)
            
            # 6. Despliegue del Resultado Limpio
            if indice_predicho < len(clases):
                fruta_predicha = clases[indice_predicho]
                confianza = predicciones[0][indice_predicho] * 100
                
                st.subheader(f"Resultado de Predicción: **{fruta_predicha}**")
                st.progress(int(confianza))
                st.write(f"📊 Nivel de confianza algorítmica: **{confianza:.2f}%**")
            else:
                st.error(f"Desbalance: El modelo arrojó el índice {indice_predicho}, pero tu catálogo cuenta con {len(clases)} clases.")
                
        except Exception as error_proceso:
            st.error(f"Ocurrió un error matemático al procesar la imagen: {error_proceso}")
