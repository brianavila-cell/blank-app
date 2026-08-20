import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import PIL.Image
import json
import imutils
from imutils import contours, perspective
from scipy.spatial import distance as dist
from io import BytesIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Global Baggage Check", page_icon="🧳", layout="wide")

# Estilo CSS Neutro y Moderno
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #2c3e50; color: white; border-radius: 8px; border: none; padding: 10px; font-weight: bold; }
    .stButton>button:hover { background-color: #34495e; border: 1px solid #3498db; }
    .header-container { padding: 20px; text-align: center; background-color: #ffffff; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; }
    .verdict-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .approved { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #a5d6a7; }
    .rejected { background-color: #ffebee; color: #c62828; border: 2px solid #ef9a9a; }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE VISIÓN Y CÁLCULO ---

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_vision(image_bytes, ref_width_cm):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts: return None, 0, 0
    
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    h_val, w_val = 0, 0

    for c in cnts:
        if cv2.contourArea(c) < 500: continue
        
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        dA = dist.euclidean(midpoint(box[0], box[1]), midpoint(box[3], box[2]))
        dB = dist.euclidean(midpoint(box[0], box[3]), midpoint(box[1], box[2]))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / ref_width_cm
        else:
            h_val, w_val = dA / pixelsPerMetric, dB / pixelsPerMetric
            # Dibujar resultados
            cv2.drawContours(orig, [box.astype("int")], -1, (46, 204, 113), 2)
            cv2.putText(orig, f"{h_val:.1f}cm", (int(box[0][0]), int(box[0][1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 152, 219), 2)
            cv2.putText(orig, f"{w_val:.1f}cm", (int(box[1][0]+10), int(box[1][1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 152, 219), 2)

    return cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), h_val, w_val

def analyze_safety(image_pil, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Eres un experto en seguridad aeroportuaria internacional. 
        Analiza la imagen para detectar objetos prohibidos en equipaje de mano según normas IATA generales.
        Responde exclusivamente en JSON:
        {
          "objetos_identificados": ["lista"],
          "riesgos_detectados": ["lista"],
          "cumple_norma_iata": true/false,
          "recomendacion": "texto breve"
        }
        """
        response = model.generate_content([prompt, image_pil])
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except:
        return {"error": "No se pudo procesar la IA"}

# --- INTERFAZ ---

st.markdown("""
    <div class='header-container'>
        <h1 style='margin:0; color: #2c3e50;'>🧳 Global Baggage Check</h1>
        <p style='color: #7f8c8d;'>Validador Universal de Equipaje con Inteligencia Artificial</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar: Configuración Global
with st.sidebar:
    st.header("⚙️ Parámetros de Vuelo")
    api_key = st.text_input("Gemini API Key", type="password", help="Obtenla en Google AI Studio")
    
    st.divider()
    st.subheader("Límites Permitidos")
    limit_h = st.number_input("Alto Máximo (cm)", value=55)
    limit_w = st.number_input("Ancho Máximo (cm)", value=40)
    
    st.divider()
    st.subheader("Referencia de Medición")
    ref_obj = st.selectbox("Objeto de referencia", ["Tarjeta (ID/Crédito)", "Pasaporte", "Otro"])
    ref_cm = st.number_input("Ancho real del objeto (cm)", value=8.56 if "Tarjeta" in ref_obj else 12.5)

# Cuerpo Principal
col_ui, col_res = st.columns([1, 1])

with col_ui:
    st.info("📸 Sube una foto de tu maleta. Asegúrate de que el objeto de referencia esté a la izquierda de la maleta en el mismo plano.")
    
    input_method = st.tabs(["📤 Subir Imagen", "📸 Usar Cámara"])
    
    with input_method[0]:
        uploaded_file = st.file_uploader("Selecciona una foto", type=['jpg', 'jpeg', 'png'])
    with input_method[1]:
        camera_file = st.camera_input("Captura tu equipaje")

    active_file = uploaded_file if uploaded_file else camera_file

if active_file:
    with col_res:
        if st.button("🚀 ANALIZAR EQUIPAJE"):
            if not api_key:
                st.warning("⚠️ Ingresa tu API Key en el panel lateral.")
            else:
                with st.spinner("Procesando visión y seguridad..."):
                    # Datos
                    bytes_data = active_file.getvalue()
                    pil_img = PIL.Image.open(BytesIO(bytes_data))
                    
                    # Ejecución Módulos
                    viz_img, h, w = process_vision(bytes_data, ref_cm)
                    safety = analyze_safety(pil_img, api_key)
                    
                    # Lógica de Validación
                    size_ok = h <= limit_h and w <= limit_w
                    safety_ok = safety.get("cumple_norma_iata", True)
                    
                    # Mostrar Imagen
                    if viz_img is not None:
                        st.image(viz_img, caption="Análisis Dimensional", use_column_width=True)
                    
                    # Resultados IA
                    st.write("### 🔍 Análisis de Seguridad")
                    st.write(f"**Objetos:** {', '.join(safety.get('objetos_identificados', ['No detectados']))}")
                    
                    if not safety_ok:
                        st.error(f"⚠️ Riesgos: {', '.join(safety.get('riesgos_detectados', []))}")
                    
                    # VERDICTO FINAL
                    if size_ok and safety_ok:
                        st.markdown(f"""
                            <div class='verdict-card approved'>
                                <h2 style='margin:0;'>✅ EQUIPAJE VALIDADO</h2>
                                <p>Cumple con las dimensiones ({h:.1f}x{w:.1f} cm) y normas de seguridad.</p>
                                <small>{safety.get('recomendacion', '')}</small>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='verdict-card rejected'>
                                <h2 style='margin:0;'>❌ NO APTO PARA CABINA</h2>
                                <p>Medidas: {h:.1f}cm x {w:.1f}cm (Máx: {limit_h}x{limit_w}cm)</p>
                                <p>{'⚠️ Posee objetos restringidos' if not safety_ok else '📏 Excede dimensiones permitidas'}</p>
                            </div>
                        """, unsafe_allow_html=True)
else:
    with col_res:
        st.write("### Instrucciones")
        st.markdown("""
        1. Configura los límites de tu aerolínea en el panel izquierdo.
        2. Pon una tarjeta de crédito o pasaporte en el suelo junto a tu maleta.
        3. Toma la foto de frente asegurando que ambos objetos sean visibles.
        4. Presiona analizar para obtener tu veredicto.
        """)
        st.image("https://img.icons8.com/illustrations/external-outline-design-circle-course-horizontal/512/external-Baggage-Check-delivery-and-logistics-outline-design-circle-course-horizontal.png", width=250)
