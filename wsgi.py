from io import BytesIO
import os
import urllib.parse
from datetime import date
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from PIL import Image
import uvicorn
import logging

# CONFIGURAÇÃO E VARIÁVEIS
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("CX")                 # Pesquisa geral
CX_SHEIN = os.getenv("CX_SHEIN")     # Pesquisa apenas Shein
CX_SHOPEE = os.getenv("CX_SHOPEE")   # Pesquisa apenas Shopee

# Carregar modelo treinado (Fashion MNIST)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model = tf.keras.models.load_model("fashion_model.keras")

# Limite de requests
MAX_REQUESTS_PER_DAY = 1000
request_count = 0
current_day = date.today()

# FastAPI + rate limiting
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("produtos_debug")

# MODEL PREDICTION

@app.post("/upload-image")
async def predict(file: UploadFile = File(...)):
    """Classifica a imagem com modelo Fashion MNIST"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("L")
        image = image.resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {"classe_predita": predicted_class, "confianca": confidence}

    except Exception as e:
        return {"erro": f"Não foi possível processar a imagem: {str(e)}"}

# BUSCA DE PRODUTOS

class Filtro(BaseModel):
    categoria: str
    genero: str | None = None
    cor: str | None = None
    estilo: str | None = None

@app.post("/buscar-produtos")
@limiter.limit("50/day")
def buscar_produtos(request: Request, filtro: Filtro):
    """Busca produtos no Google Shopping, Shein e Shopee"""
    global request_count, current_day

    # Reset diário do contador
    if date.today() != current_day:
        current_day = date.today()
        request_count = 0

    if request_count >= MAX_REQUESTS_PER_DAY:
        raise HTTPException(status_code=429, detail="Limite diário atingido.")
    request_count += 1
    logger.debug(f"Request count: {request_count}")

    query = " ".join([v for v in filtro.model_dump().values() if v])
    query_encoded = urllib.parse.quote_plus(query)
    logger.debug(f"Query: {query}")

    resultados = {"google_shopping": [], "shein": [], "shopee": []}

    url_google = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={query_encoded}&key={GOOGLE_API_KEY}&cx={CX}&gl=br&hl=pt-BR"
    )

    url_shein = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={query_encoded}&key={GOOGLE_API_KEY}&cx={CX_SHEIN}&gl=br&hl=pt-BR"
    )

    url_shopee = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={query_encoded}&key={GOOGLE_API_KEY}&cx={CX_SHOPEE}&gl=br&hl=pt-BR"
    )

    def extrair_preco(pagemap: dict):
        price = None
        currency = None

        if "offer" in pagemap:
            offer = pagemap.get("offer", [{}])[0]
            price = offer.get("price")
            currency = offer.get("pricecurrency") or offer.get("priceCurrency")
        if not price and "product" in pagemap:
            product = pagemap.get("product", [{}])[0]
            price = product.get("price")
            currency = product.get("pricecurrency") or product.get("priceCurrency")
        if not price and "metatags" in pagemap:
            meta = pagemap.get("metatags", [{}])[0]
            price = meta.get("product:price:amount") or meta.get("og:price:amount")
            currency = meta.get("product:price:currency") or meta.get("og:price:currency")

        return price or "N/A", currency or ""

    def processar_resultados(json_data):
        items = []
        for i in json_data.get("items", [])[:5]:
            pagemap = i.get("pagemap", {})
            preco, moeda = extrair_preco(pagemap)
            items.append({
                "titulo": i.get("title"),
                "url": i.get("link"),
                "imagem": (pagemap.get("cse_image", [{}])[0].get("src")
                           if "cse_image" in pagemap else None),
                "preco": preco,
                "moeda": moeda,
                "snippet": i.get("snippet"),
            })
        return items

    try:
        # Google Shopping
        resp_google = requests.get(url_google, timeout=10).json()
        resultados["google_shopping"] = processar_resultados(resp_google)

        # Shein
        resp_shein = requests.get(url_shein, timeout=10).json()
        resultados["shein"] = processar_resultados(resp_shein)

        # Shopee
        resp_shopee = requests.get(url_shopee, timeout=10).json()
        resultados["shopee"] = processar_resultados(resp_shopee)

    except Exception as e:
        logger.error(f"Erro Google API: {e}")
        raise HTTPException(status_code=500, detail=f"Erro Google API: {e}")

    return resultados


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
