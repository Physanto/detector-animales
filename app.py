import streamlit as st
import cv2
import math
import tempfile
import av
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Detector de Perros", page_icon="🐕")
st.title("Detector de Perros y Dueños 🐕🚶‍♂️")

@st.cache_resource
def cargar_modelo():
    return YOLO('yolov8n.pt')

modelo = cargar_modelo()

DISTANCIA_MAXIMA = 300
SALTAR_FRAMES = 5 

# --- CORRECCIÓN 1: Más servidores STUN para evitar bloqueos de red móvil ---
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]}
)

modo = st.sidebar.radio("Elige un modo:", ["Cámara en Vivo", "Subir Video"])

# ==========================================
# LÓGICA DE DIBUJO
# ==========================================
def dibujar_detecciones(frame, personas, perros):
    for p in personas:
        x1, y1, x2, y2 = p['caja']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    for d in perros:
        x1, y1, x2, y2 = d['caja']
        centro_perro = d['centro']
        tiene_dueno = False
        
        for p in personas:
            if math.dist(centro_perro, p['centro']) < DISTANCIA_MAXIMA:
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
# MODO 1: CÁMARA EN VIVO
# ==========================================
class ProcesadorVideoYOLO(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_personas = []
        self.last_perros = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % SALTAR_FRAMES == 0:
            resultados = modelo.predict(img, classes=[0, 16], verbose=False)
            self.last_personas, self.last_perros = [] , []
            
            for box in resultados[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                clase = int(box.cls[0])
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if clase == 0:
                    self.last_personas.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                elif clase == 16:
                    self.last_perros.append({'caja': (x1, y1, x2, y2), 'centro': centro})

        img_dibujada = dibujar_detecciones(img, self.last_personas, self.last_perros)
        return av.VideoFrame.from_ndarray(img_dibujada, format="bgr24")

if modo == "Cámara en Vivo":
    st.write("Presiona 'START'. Si pide permisos, acéptalos.")
    
    webrtc_streamer(
        key="detector-perros",
        video_processor_factory=ProcesadorVideoYOLO,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "facingMode": "environment", # Cámara trasera
            },
            "audio": False
        },
        # --- CORRECCIÓN 2: Atributos HTML obligatorios para móviles (Especialmente iOS) ---
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #ccc"},
            "controls": False,
            "autoPlay": True,
            "playsinline": True, # ¡ESTO SOLUCIONA LA PANTALLA NEGRA EN MÓVILES!
        },
        async_processing=True,
    )

# ==========================================
# MODO 2: SUBIR VIDEO
# ==========================================
elif modo == "Subir Video":
    archivo_video = st.file_uploader("Selecciona un video", type=["mp4", "mov", "avi"])

    if archivo_video is not None:
        st.info("Procesando video... Esto puede tomar un momento.")
        barra_progreso = st.progress(0)
        
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(archivo_video.read())
        
        ruta_salida_webm = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
        
        cap = cv2.VideoCapture(tfile_in.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(ruta_salida_webm, fourcc, fps, (ancho, alto))
        
        frame_count = 0
        last_personas, last_perros = [], []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if total_frames > 0:
                barra_progreso.progress(min(frame_count / total_frames, 1.0))
            
            if frame_count % SALTAR_FRAMES == 0:
                resultados = modelo.predict(frame, classes=[0, 16], verbose=False)
                last_personas, last_perros = [], []
                
                for box in resultados[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    clase = int(box.cls[0])
                    centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if clase == 0:
                        last_personas.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                    elif clase == 16:
                        last_perros.append({'caja': (x1, y1, x2, y2), 'centro': centro})
                        
            frame_procesado = dibujar_detecciones(frame, last_personas, last_perros)
            out.write(frame_procesado)

        cap.release()
        out.release()
        
        st.success("¡Video procesado con éxito!")
        
        with open(ruta_salida_webm, 'rb') as v_file:
            st.video(v_file.read(), format="video/webm")
