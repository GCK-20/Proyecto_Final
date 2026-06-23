!pip install sentence-transformers faiss-cpu gradio
!pip install pdfplumber

# ingest.py
# Rol: Preprocesamiento de documentos para búsqueda semántica
# Descripción: Este módulo se encarga de cargar documentos (PDF, TXT, CSV),
# limpiarlos y dividirlos en fragmentos listos para vectorización.

import re
import os
import pdfplumber
import pandas as pd

def limpiar_texto(texto: str) -> str:
    """
    Normaliza el texto eliminando saltos de línea, caracteres raros y espacios repetidos.
    """
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto)  # espacios múltiples
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', '', texto)  # caracteres raros
    return texto.strip()

def dividir_en_fragmentos(texto: str, max_len: int = 300) -> list:
    """
    Divide el texto en fragmentos de tamaño máximo (ej. 300 palabras).
    Esto permite que los embeddings sean más precisos.
    """
    palabras = texto.split()
    fragmentos = []
    for i in range(0, len(palabras), max_len):
        fragmento = " ".join(palabras[i:i+max_len])
        fragmentos.append(fragmento)
    return fragmentos

def cargar_pdf(ruta_pdf: str) -> str:
    """
    Extrae texto de un archivo PDF usando pdfplumber.
    """
    texto = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto += pagina.extract_text() + " "
    return texto

def cargar_csv(ruta_csv: str, columna: str = "texto") -> str:
    """
    Carga un archivo CSV y concatena los textos de la columna indicada.
    """
    df = pd.read_csv(ruta_csv)
    return " ".join(df[columna].astype(str).tolist())

def procesar_documento(ruta: str) -> list:
    """
    Procesa un documento (PDF, TXT o CSV) y devuelve fragmentos limpios.
    """
    extension = os.path.splitext(ruta)[1].lower()
    
    if extension == ".pdf":
        texto = cargar_pdf(ruta)
    elif extension == ".csv":
        texto = cargar_csv(ruta)
    elif extension == ".txt":
        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read()
    else:
        raise ValueError(f"Formato no soportado: {extension}")
    
    texto_limpio = limpiar_texto(texto)
    fragmentos = dividir_en_fragmentos(texto_limpio)
    return fragmentos




#Prueba para después de la organización de carpetas
#from src.ingest import procesar_documento

#fragmentos = procesar_documento("data/documentos/contrato.pdf")
#print(fragmentos[:3])  # muestra los primeros 3 fragmentos

