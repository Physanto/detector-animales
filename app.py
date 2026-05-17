import streamlit as st
import cv2
import math
import tempfile
import numpy as np

from ultralytics import YOLO

# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Detector de Perros",
    page_icon="🐕",
    layout="wide"
)

st.title("🐕 Detector de Perros y Personas")

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")

modelo = cargar_modelo()

DISTANCIA_MAXIMA = 300

# ==========================================
# SIDEBAR
# ==========================================

modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["Tomar Foto", "Subir Imagen", "Subir Video"]
)

# ==========================================
# FUNCIÓN DE DETECCIÓN
# ==========================================

def detectar(frame):

    resultados = modelo.predict(
        frame,
        classes=[0, 16],
        verbose=False
    )

    personas = []
    perros = []

    for box in resultados[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        clase = int(box.cls[0])

        centro = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

        if clase == 0:
            personas.append({
                "caja": (x1, y1, x2, y2),
                "centro": centro
            })

        elif clase == 16:
            perros.append({
                "caja": (x1, y1, x2, y2),
                "centro": centro
            })

    # Dibujar personas
    for persona in personas:

        x1, y1, x2, y2 = persona["caja"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Persona",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Dibujar perros
    for perro in perros:

        x1, y1, x2, y2 = perro["caja"]

        tiene_dueno = False

        for persona in personas:

            distancia = math.dist(
                perro["centro"],
                persona["centro"]
            )

            if distancia < DISTANCIA_MAXIMA:
                tiene_dueno = True
                break

        if tiene_dueno:
            etiqueta = "Perro con persona"
            color = (0, 255, 0)
        else:
            etiqueta = "Perro solo"
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            etiqueta,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return frame

# ==========================================
# MODO FOTO CÁMARA
# ==========================================

if modo == "Tomar Foto":

    foto = st.camera_input("📸 Toma una foto")

    if foto is not None:

        file_bytes = np.asarray(
            bytearray(foto.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes, 1)

        resultado = detectar(frame)

        resultado_rgb = cv2.cvtColor(
            resultado,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            resultado_rgb,
            channels="RGB",
            use_container_width=True
        )

# ==========================================
# MODO SUBIR IMAGEN
# ==========================================

elif modo == "Subir Imagen":

    archivo = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:

        file_bytes = np.asarray(
            bytearray(archivo.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes, 1)

        resultado = detectar(frame)

        resultado_rgb = cv2.cvtColor(
            resultado,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            resultado_rgb,
            channels="RGB",
            use_container_width=True
        )

# ==========================================
# MODO VIDEO
# ==========================================

elif modo == "Subir Video":

    archivo_video = st.file_uploader(
        "Sube un video",
        type=["mp4", "mov", "avi"]
    )

    if archivo_video is not None:

        st.write("Procesando video...")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(archivo_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        frame_skip = 5
        contador = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            contador += 1

            if contador % frame_skip == 0:

                frame = detectar(frame)

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                stframe.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

        cap.release()

        st.success("✅ Video procesado")
