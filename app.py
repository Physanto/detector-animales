import streamlit as st
import cv2
import math
import tempfile
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Detector de Perros",
    page_icon="🐕",
    layout="wide"
)

st.title("🐕 Detector de Perros y Personas")

@st.cache_resource
def cargar_modelo():
    modelo = YOLO("yolov8n.pt")
    return modelo

modelo = cargar_modelo()

DISTANCIA_MAXIMA = 300
SALTAR_FRAMES = 5

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["Cámara en Vivo", "Subir Video"]
)

def dibujar_detecciones(frame, personas, perros):

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
            0.6,
            (255, 0, 0),
            2
        )

    for perro in perros:

        x1, y1, x2, y2 = perro["caja"]
        centro_perro = perro["centro"]

        tiene_dueno = False

        for persona in personas:

            distancia = math.dist(
                centro_perro,
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
            0.6,
            color,
            2
        )

    return frame

class ProcesadorVideoYOLO(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_personas = []
        self.last_perros = []

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        if self.frame_count % SALTAR_FRAMES == 0:

            resultados = modelo.predict(
                img,
                classes=[0, 16],
                verbose=False
            )

            personas_temp = []
            perros_temp = []

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
                    personas_temp.append({
                        "caja": (x1, y1, x2, y2),
                        "centro": centro
                    })

                elif clase == 16:
                    perros_temp.append({
                        "caja": (x1, y1, x2, y2),
                        "centro": centro
                    })

            self.last_personas = personas_temp
            self.last_perros = perros_temp

        img = dibujar_detecciones(
            img,
            self.last_personas,
            self.last_perros
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

if modo == "Cámara en Vivo":

    st.info("Presiona START y acepta permisos de cámara.")

    webrtc_streamer(
        key="detector-perros",
        video_processor_factory=ProcesadorVideoYOLO,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

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

        espacio_video = st.empty()

        frame_count = 0

        last_personas = []
        last_perros = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % SALTAR_FRAMES == 0:

                resultados = modelo.predict(
                    frame,
                    classes=[0, 16],
                    verbose=False
                )

                last_personas = []
                last_perros = []

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
                        last_personas.append({
                            "caja": (x1, y1, x2, y2),
                            "centro": centro
                        })

                    elif clase == 16:
                        last_perros.append({
                            "caja": (x1, y1, x2, y2),
                            "centro": centro
                        })

            frame = dibujar_detecciones(
                frame,
                last_personas,
                last_perros
            )

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            espacio_video.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

        st.success("✅ Video procesado correctamente")
