from io import BytesIO
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import date
from slowapi import Limiter
from slowapi.util import get_remote_address
import tensorflow as tf
from PIL import Image

# Classes do Fashion MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Carregar modelo treinado
model = tf.keras.models.load_model("fashion_model.keras")

# Carregar .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CX = os.getenv("CX")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

request_count = 0
current_day = date.today()
MAX_REQUESTS_PER_DAY = 1000
cache = {}

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

class Filtro(BaseModel):
    categoria: str
    genero: str | None = None
    cor: str | None = None
    estilo: str | None = None

@app.post("/upload-image")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler o conteúdo do upload
        contents = await file.read()
        import io
        image = Image.open(BytesIO(contents)).convert("L")  # grayscale
        image = image.resize((28, 28))

        # Converter para numpy array normalizado
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # batch de 1, com canal

        # Fazer predição
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"classe_predita": predicted_class, "confianca": confidence}

    except Exception as e:
        return {"erro": f"Não foi possível processar a imagem: {str(e)}"}

@app.post("/buscar-produtos")
@limiter.limit("50/day")
def buscar_produtos(request: Request, filtro: Filtro):
    global request_count, current_day

    if date.today() != current_day:
        current_day = date.today()
        request_count = 0

    if request_count >= MAX_REQUESTS_PER_DAY:
        raise HTTPException(status_code=429, detail="Limite diário atingido (50 requisições). Tente amanhã.")
    request_count += 1

    query = " ".join([v for v in filtro.dict().values() if v])
    if query in cache:
        return cache[query]

    resultados = {}

    # Mercado Livre
    url_ml = f"https://api.mercadolibre.com/sites/MLB/search?q={query}"
    resp_ml = requests.get(url_ml, timeout=10).json()
    resultados["mercado_livre"] = [
        {"titulo": i.get("title"), "preco": i.get("price"), "url": i.get("permalink"), "imagem": i.get("thumbnail")}
        for i in resp_ml.get("results", [])[:5]
    ]

    # Shein
    actor_url = f"https://api.apify.com/v2/acts/factual_biscotti~shein-visual-search-actor/run-sync-get-dataset?token={APIFY_TOKEN}"
    try:
        payload = {"searchTerm": query}
        resp_shein = requests.post(actor_url, json=payload, timeout=10).json()
        resultados["shein"] = [
            {"titulo": i.get("name"), "preco": i.get("price", {}).get("amount"), "moeda": i.get("price", {}).get("currency"),
             "url": i.get("url"), "imagem": i.get("imageUrl")}
            for i in resp_shein.get("items", [])[:5]
        ]
    except Exception:
        resultados["shein"] = []

    # Google Shopping
    url_google = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CX}"
    resp_google = requests.get(url_google, timeout=10).json()
    resultados["google_shopping"] = [
        {"titulo": i.get("title"), "url": i.get("link"),
         "imagem": i.get("pagemap", {}).get("cse_image", [{}])[0].get("src"),
         "snippet": i.get("snippet")}
        for i in resp_google.get("items", [])[:5]
    ]

    cache[query] = resultados
    return resultados
