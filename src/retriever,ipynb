# retriever.ipynb
# Rol: Recuperación semántica de fragmentos
# Descripción: Este notebook recibe una pregunta del usuario,
# la convierte en embedding y busca los fragmentos más relevantes en el índice FAISS.

import numpy as np

def buscar_fragmentos(pregunta: str, index, fragmentos: list, modelo, k: int = 3) -> list:
    """
    Busca los fragmentos más relevantes para una pregunta usando FAISS.
    
    Parámetros:
    - pregunta: texto de la consulta del usuario
    - index: índice FAISS creado en indexer.ipynb
    - fragmentos: lista de textos preprocesados
    - modelo: modelo de embeddings cargado en indexer.ipynb
    - k: número de fragmentos relevantes a devolver (por defecto 3)
    
    Retorna:
    - lista de fragmentos relevantes
    """
    # Convertir la pregunta en embedding
    embedding_q = modelo.encode([pregunta])
    
    # Buscar en el índice FAISS
    distancias, indices = index.search(np.array(embedding_q), k)
    
    # Recuperar los fragmentos más cercanos
    resultados = [fragmentos[i] for i in indices[0]]
    return resultados
