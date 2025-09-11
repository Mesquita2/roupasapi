from fastapi import FastAPI, HTTPException, Request, UploadFile, File # type: ignore
from fastapi.responses import JSONResponse # type: ignore 
from pydantic import BaseModel # type: ignore
import uvicorn # type: ignore
import requests # type: ignore
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore 
from dotenv import load_dotenv  # type: ignore
from datetime import date
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import shutil
import tensorflow as tf
from PIL import Image


# Carregar .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("CX")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")


# Contador
request_count = 0
current_day = date.today()
MAX_REQUESTS_PER_DAY = 50
app = FastAPI()
# Modelo para entrada JSON
class Filtro(BaseModel):
    categoria: str
    genero: str | None = None
    cor: str | None = None
    estilo: str | None = None
cache = {} # armazenamento para caso o query ja tenha sido feito 

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

# Envio para a ia do colab 
@app.post("/upload-image")
async def predict(file: UploadFile = File(...)):
    # Abrir a imagem recebida
    image = Image.open(file.file).convert("L")  # "L" = grayscale
    image = image.resize((28, 28))  # Fashion MNIST é 28x28
    
    # Converter para numpy array normalizado
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)  # Batch de 1 imagem

    # Fazer a predição
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"classe_predita": predicted_class, "confianca": confidence}
    

# Busca de produtos 
@app.post("/buscar-produtos")
@limiter.limit("50/day") 
def buscar_produtos(request: Request, filtro: Filtro):
    
    global request_count, current_day
    
    # reseta se mudou o dia
    if date.today() != current_day:
        current_day = date.today()
        request_count = 0

    # bloqueia se passar do limite
    if request_count >= MAX_REQUESTS_PER_DAY:
        raise HTTPException(status_code=429, detail="Limite diário atingido (50 requisições). Tente amanhã.")

    request_count += 1
    
    
    query = " ".join([v for v in filtro.dict().values() if v])  # gera string da query
    resultados = {}

    # -------- Mercado Livre --------
    if query in cache:
        return cache[query]
    
    url_ml = f"https://api.mercadolibre.com/sites/MLB/search?q={query}"
    resp_ml = requests.get(url_ml).json()

    resultados["mercado_livre"] = [
        {
            "titulo": item.get("title"),
            "preco": item.get("price"),
            "url": item.get("permalink"),
            "imagem": item.get("thumbnail")
        }
        for item in resp_ml.get("results", [])[:5]
    ]

    # -------- Shein (via Apify) --------
    actor_url = f"https://api.apify.com/v2/acts/factual_biscotti~shein-visual-search-actor/run-sync-get-dataset?token={APIFY_TOKEN}"

    try:
        payload = {"searchTerm": query}
        resp_shein = requests.post(actor_url, json=payload, timeout=30).json()
        resultados["shein"] = [
            {
                "titulo": item.get("name"),
                "preco": item.get("price", {}).get("amount"),
                "moeda": item.get("price", {}).get("currency"),
                "url": item.get("url"),
                "imagem": item.get("imageUrl")
            }
            for item in resp_shein.get("items", [])[:5]
        ]
    except Exception:
        resultados["shein"] = []

    # -------- Google Shopping (Custom Search API) --------
    url_google = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CX}"
    resp_google = requests.get(url_google).json()

    resultados["google_shopping"] = [
        {
            "titulo": item.get("title"),
            "url": item.get("link"),
            "imagem": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src"),
            "snippet": item.get("snippet")
        }
        for item in resp_google.get("items", [])[:5]
    ]

    cache[query] = resultados  # salva no cache
    return resultados


