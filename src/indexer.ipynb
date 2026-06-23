#indexer.py
# indexer.py
# Rol: Indexación semántica de fragmentos
# Descripción: Este módulo convierte fragmentos de texto en embeddings
# y construye un índice FAISS para búsquedas rápidas.

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Cargar modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

def crear_indice(fragmentos: list):
    """
    Convierte fragmentos en embeddings y construye un índice FAISS.
    
    Parámetros:
    - fragmentos: lista de textos ya preprocesados
    
    Retorna:
    - index: índice FAISS con los embeddings
    - embeddings: matriz numpy con los vectores
    """
    # Generar embeddings
    embeddings = modelo.encode(fragmentos)
    
    # Crear índice FAISS con distancia euclidiana (L2)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Agregar embeddings al índice
    index.add(np.array(embeddings))
    
    return index, embeddings
