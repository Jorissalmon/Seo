import os
from dotenv import load_dotenv
import requests
import json
import base64
from PIL import Image
from openai import OpenAI

# Chargement des variables d'environnement
load_dotenv()
LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')

def message_llama(prompt, model="llama3.2:1b", max_tokens=100, temperature=0.7, top_p=1, n=1):
    client = OpenAI(
        api_key=LLAMA_API_KEY,
        base_url="https://api.llama-api.com"
    )
    
    try:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = client.chat.completions.create(
            model="llama3.2-90b",  # Modèle pour la génération de texte
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n
        )
        
        if isinstance(response.choices, list) and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "Aucun texte généré."
    
    except Exception as e:
        return f"Erreur de connexion : {str(e)}"

def encode_image_to_base64(image_path):
    """Encode une image en base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé : {image_path}")
    except Exception as e:
        raise Exception(f"Erreur lors de l'encodage de l'image : {str(e)}")

def analyse_image_llm(image_path):
    """Analyse une image en utilisant l'API vision"""
    try:
        client = OpenAI(
            api_key=LLAMA_API_KEY,
            base_url="https://api.llama-api.com"
        )
        
        base64_image = encode_image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image? Please provide a detailed description."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="llama3.2-90b-vision",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=1
        )
        
        if isinstance(response.choices, list) and len(response.choices) > 0:
            return response.choices[0].message.content
        return "Aucune description générée."
            
    except Exception as e:
        print(f"Erreur lors de l'analyse d'image : {str(e)}")
        return None

if __name__ == "__main__":
    # Test des deux fonctionnalités
    print("\n=== Test de l'analyse d'image avec recommandations SEO ===")
    result = analyse_image_llm('./bear.jpg')
    if result:
        print("\nDescription de l'image:", result)
    
    print("\n=== Test de génération de texte ===")
    prompt = "Rédige un article optimisé SEO sur les ours bruns."
    text_result = message_llama(prompt)
    print("\nRéponse:", text_result)