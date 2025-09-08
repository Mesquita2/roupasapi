from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
import requests # type: ignore
import os
from dotenv import load_dotenv  # type: ignore
from datetime import date

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


# Controllers
@app.post("/buscar-produtos")
def buscar_produtos(filtro: Filtro):
    
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
