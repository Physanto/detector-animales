import streamlit as st
import cv2
import math
import tempfile
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Detector de Perros", page_icon="🐕")
st.title("Detector de Perros y Dueños")

# Configurar el modelo en caché
@st.cache_resource
def cargar_modelo():
    return YOLO('yolov8n.pt')

modelo = cargar_modelo()

# Parámetros globales
DISTANCIA_MAXIMA = 300
SALTAR_FRAMES = 5  # La IA solo procesará 1 de cada 5 fotogramas (reduce el lag)

# Configuración para que el video en vivo funcione en redes móviles
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Seleccionar modo
modo = st.sidebar.radio("Elige un modo:", ["Cámara en Vivo", "Subir Video"])

# ==========================================
# LÓGICA DE DIBUJO (Se usa en ambos modos)
# ==========================================
def dibujar_detecciones(frame, personas, perros):
    # Dibujar personas
    for p in personas:
        x1, y1, x2, y2 = p['caja']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    # Dibujar perros con lógica de distancia
    for d in perros:
        x1, y1, x2, y2 = d['caja']
        centro_perro = d['centro']
        tiene_dueno = False
        
        for p in personas:
            distancia = math.dist(centro_perro, p['centro'])
            if distancia < DISTANCIA_MAXIMA:
                tiene_dueno = True
                break
                
        if tiene_dueno:
            etiqueta = "Perro persona (Tiene dueno)"
            color = (0, 255, 0)
        else:
            etiqueta = "Calle"
            color = (0, 0, 255)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return frame

# ==========================================
# MODO 1: CÁMARA EN VIVO (WEBRTC)
# ==========================================
class ProcesadorVideoYOLO(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_personas = []
        self.last_perros = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Solo procesamos con YOLO cada X frames para evitar lag
        if self.frame_count % SALTAR_FRAMES == 0:
            resultados = modelo.predict(img, classes=[0, 16], verbose=False)
            personas_temp = []
            perros_temp = []
            
            for box in resultados[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                clase = int(box.cls[0])
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if clase == 0:
                    personas_temp.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                elif clase == 16:
                    perros_temp.append({'caja': (x1, y1, x2, y2), 'centro': centro})
            
            # Actualizamos las cajas guardadas
            self.last_personas = personas_temp
            self.last_perros = perros_temp

        # Dibujamos las últimas cajas conocidas en el fotograma actual
        img_dibujada = dibujar_detecciones(img, self.last_personas, self.last_perros)
        
        return av.VideoFrame.from_ndarray(img_dibujada, format="bgr24")

if modo == "Cámara en Vivo":
    st.write("Presiona 'START' y permite el acceso a la cámara de tu celular.")
    webrtc_streamer(
        key="detector-perros",
        video_processor_factory=ProcesadorVideoYOLO,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False} # No necesitamos audio
    )

# ==========================================
# MODO 2: SUBIR VIDEO
# ==========================================
elif modo == "Subir Video":
    archivo_video = st.file_uploader("Selecciona un video", type=["mp4", "mov", "avi"])

    if archivo_video is not None:
        st.write("Procesando video de forma optimizada...")
        
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
            
            # Procesar 1 de cada X frames
            if frame_count % SALTAR_FRAMES == 0:
                resultados = modelo.predict(frame, classes=[0, 16], verbose=False)
                last_personas = []
                last_perros = []
                
                for box in resultados[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    clase = int(box.cls[0])
                    centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if clase == 0:
                        last_personas.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                    elif clase == 16:
                        last_perros.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                        
            # Dibujar usando los datos guardados
            frame = dibujar_detecciones(frame, last_personas, last_perros)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            espacio_video.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("¡Análisis de video completado!")
