import streamlit as st
import cv2
import math
import tempfile
import os
import av
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

# Configuración STUN para WebRTC
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

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
# MODO 1: CÁMARA EN VIVO (CORREGIDO PARA CELULAR)
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
            self.last_personas, self.last_perros = [], []
            
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
    st.write("Presiona 'START' para activar la cámara trasera de tu celular.")
    webrtc_streamer(
        key="detector-perros",
        video_processor_factory=ProcesadorVideoYOLO,
        rtc_configuration=RTC_CONFIG,
        # AQUI ESTA LA MAGIA PARA CELULARES: Pedimos explicitamente la cámara trasera (environment)
        media_stream_constraints={
            "video": {"facingMode": "environment"}, 
            "audio": False
        } 
    )

# ==========================================
# MODO 2: SUBIR VIDEO (CORREGIDO PARA FLUIDEZ)
# ==========================================
elif modo == "Subir Video":
    archivo_video = st.file_uploader("Selecciona un video", type=["mp4", "mov", "avi"])

    if archivo_video is not None:
        st.info("Procesando video... Esto puede tomar un momento. Al terminar se reproducirá fluidamente.")
        barra_progreso = st.progress(0)
        
        # Crear archivos temporales
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(archivo_video.read())
        ruta_salida_bruta = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        cap = cv2.VideoCapture(tfile_in.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Preparar escritura del video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(ruta_salida_bruta, fourcc, fps, (ancho, alto))
        
        frame_count = 0
        last_personas, last_perros = [], []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Actualizar barra de progreso visual
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
        
        st.write("Adaptando el formato del video para la web...")
        # TRUCO PARA WEB: Convertimos el video a un formato H264 compatible con todos los navegadores usando ffmpeg
        ruta_final_web = ruta_salida_bruta.replace(".mp4", "_web.mp4")
        os.system(f"ffmpeg -y -i {ruta_salida_bruta} -vcodec libx264 {ruta_final_web}")
        
        st.success("¡Video procesado con éxito!")
        
        # Mostrar el video resultante fluidamente
        if os.path.exists(ruta_final_web):
            with open(ruta_final_web, 'rb') as v_file:
                st.video(v_file.read())
        else:
            # Plan B por si falla la conversión
            with open(ruta_salida_bruta, 'rb') as v_file:
                st.video(v_file.read())
