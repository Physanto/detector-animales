import streamlit as st
import cv2
import math
import tempfile
from ultralytics import YOLO

# 1. Configurar la página de Streamlit
st.set_page_config(page_title="Detector de Perros", page_icon="🐕")
st.title("Detector de Perros y Dueños 🐕🚶‍♂️")
st.write("Sube un video o graba uno con tu celular para analizarlo.")

# 2. Cargar el modelo en caché (para que no se descargue cada vez que subes un video)
@st.cache_resource
def cargar_modelo():
    return YOLO('yolov8n.pt')

modelo = cargar_modelo()

# 3. Crear el botón para subir/grabar video
archivo_video = st.file_uploader("Selecciona o graba un video", type=["mp4", "mov", "avi"])

DISTANCIA_MAXIMA = 300 # Ajusta este valor si es necesario

if archivo_video is not None:
    st.write("Procesando video... por favor espera.")
    
    # Guardar el video subido en un archivo temporal para que OpenCV pueda leerlo
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(archivo_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Crear un espacio vacío en Streamlit donde mostraremos el video cuadro por cuadro
    espacio_video = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Fin del video
            
        # Hacer predicción
        resultados = modelo.predict(frame, classes=[0, 16], verbose=False)
        personas = []
        perros = []
        
        # Extraer coordenadas
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            clase = int(box.cls[0])
            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2
            
            if clase == 0: # Persona
                personas.append({'caja': (x1, y1, x2, y2), 'centro': (centro_x, centro_y)})
            elif clase == 16: # Perro
                perros.append({'caja': (x1, y1, x2, y2), 'centro': (centro_x, centro_y)})
                
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
            
        # IMPORTANTE: OpenCV usa colores BGR, pero Streamlit/Web usa RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Mostrar el fotograma en la página web
        espacio_video.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.success("¡Análisis completado!")