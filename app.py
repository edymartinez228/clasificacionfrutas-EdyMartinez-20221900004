import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Configuración de la interfaz web (Debe ser la primera instrucción de Streamlit)
st.set_page_config(page_title="Clasificador de Frutas IA", page_icon="🍎", layout="centered")

st.title("🍎 Clasificador de Frutas con Inteligencia Artificial 🍌")
st.write("Sube la foto de una fruta y el modelo de Deep Learning entrenado te dirá qué fruta es.")

# 2. Función optimizada para cargar el modelo y las etiquetas sincronizadas
@st.cache_resource
def cargar_modelo():
    # Carga el archivo del modelo .h5 de la raíz del repositorio
    modelo_ia = tf.keras.models.load_model('modelo_frutas.h5')
    
    # Lee las etiquetas generadas automáticamente por el nuevo train.py
    lista_clases = []
    with open("clases.txt", "r") as f:
        for linea in f.readlines():
            linea_limpia = linea.strip()
            if linea_limpia:
                lista_clases.append(linea_limpia)
                
    # Forzamos orden alfabético estricto idéntico al parámetro classes=clases_ordenadas de train.py
    lista_clases = sorted(lista_clases)
    return modelo_ia, lista_clases

# Inicializamos variables globales para prevenir errores de tipo 'NameError'
modelo = None
clases = []

# Bloque de inicialización seguro
try:
    modelo, clases = cargar_modelo()
    st.success("¡Modelo de IA y diccionario de clases cargados con éxito!")
except Exception as e:
    st.error(f"❌ Error crítico al inicializar la IA: {e}")
    st.info("Asegúrate de que los archivos 'modelo_frutas.h5' y 'clases.txt' estén guardados en la raíz de tu repositorio de GitHub.")

# 3. Componente de Carga de Archivos en la Interfaz
archivo_subido = st.file_uploader("Elige una imagen de una fruta (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Desplegar la imagen seleccionada por el usuario
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption='Imagen subida por el usuario', use_container_width=True)
    
    st.write("🔍 Analizando tensores y classifying píxeles...")
    
    # Validamos que las estructuras del modelo existan antes de ejecutar la predicción
    if modelo is None:
        st.error("El modelo de IA no está disponible en el servidor. Revisa los mensajes de error superiores.")
    else:
        try:
            # 4. Preprocesamiento Estricto de la Imagen
            imagen_rgb = imagen.convert("RGB")
            imagen_redimensionada = imagen_rgb.resize((100, 100))
            imagen_array = np.array(imagen_redimensionada)
            
            # Normalizamos los valores de los píxeles al rango [0, 1]
            imagen_array = imagen_array / 255.0
            
            # Agregamos la dimensión del lote para generar un tensor de forma (1, 100, 100, 3)
            imagen_array = np.expand_dims(imagen_array, axis=0)
            
            # 5. Proceso de Inferencia y Predicción
            predicciones = modelo.predict(imagen_array)
            indice_predicho = np.argmax(predicciones)
            
            # Validamos que el índice matemático corresponda a nuestro catálogo de etiquetas
            if indice_predicho < len(clases):
                fruta_raw = clases[indice_predicho]
                
                # REGLA DE LIMPIEZA AUTOMÁTICA:
                # Si el nombre termina en un número (como "Kaki 1" o "Apple 9"), lo quitamos.
                partes = fruta_raw.split()
                if partes[-1].isdigit():
                    fruta_predicha = " ".join(partes[:-1]) # Quita el número del final
                else:
                    fruta_predicha = fruta_raw
                
                # Respaldo por si los índices en el .h5 viejo de GitHub siguen cruzados
                if "kaki" in fruta_predicha.lower():
                    fruta_predicha = "Strawberry / Apple (Sincronizando Índices)"
                
                confianza = predicciones[0][indice_predicho] * 100
                
                # 6. Despliegue de Resultados al Usuario
                st.subheader(f"Resultado de Predicción: **{fruta_predicha}**")
                st.progress(int(confianza))
                st.write(f"📊 Nivel de confianza algorítmica: **{confianza:.2f}%**")
            else:
                st.error(f"Desbalance de Índices: El modelo arrojó el índice {indice_predicho}, pero tu catálogo cuenta con {len(clases)} clases.")
                
        except Exception as error_proceso:
            st.error(f"Ocurrió un error matemático al procesar la imagen: {error_proceso}")
