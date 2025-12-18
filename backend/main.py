from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS pour que le frontend puisse appeler
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change en prod par ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY manquante dans .env !")

genai.configure(api_key=GEMINI_API_KEY)

# Modèle rapide et gratuit (change pour gemini-1.5-pro si tu veux plus puissant)
MODEL_NAME = "gemini-1.5-flash"  # Très rapide, gratuit, excellent en malagasy

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction="""
Tu es un conseiller juridique expert à Madagascar (basé sur le Code Civil Malgache et les lois rurales/Dina).
Ta mission est de rendre le droit accessible aux jeunes et aux citoyens.
Utilise un ton bienveillant, pédagogique et rassurant.
Explique avec des exemples simples de la vie quotidienne.
Si la situation est grave ou complexe, conseille toujours : 
"Tsara kokoa ny manatona mpisolovava na manam-pahefana eo an-toerana mba hahazoana torohevitra manokana."
Réponds en Malagasy clair et standard, sauf si l'utilisateur demande autre chose.
    """
)

class ExplainRequest(BaseModel):
    text: str

@app.post("/explain")
async def explain(req: ExplainRequest):
    try:
        # Prompt clair pour Gemini
        full_prompt = f"Explique cet article du Code Civil Malgache de façon simple et pédagogique en Malagasy :\n\n{req.text}"

        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )

        explanation = response.text.strip()

        return {"explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Gemini : {str(e)}")