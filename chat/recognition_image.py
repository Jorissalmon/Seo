import json
import requests
import base64
from openai import OpenAI

def encode_image_to_base64(image_path):
    """Encode une image en base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier non trouvé : {image_path}")
    except Exception as e:
        raise Exception(f"Erreur lors de l'encodage de l'image : {str(e)}")

def analyse_image_llm(image_path):
    """Analyse une image en utilisant un modèle LLM via API."""
    try:
        # Configuration du client OpenAI
        client = OpenAI(
            api_key="LA-36fd3e88d5a14d0490758fd0ad98730a47da942fb90e403f96cd088e1b539257",
            base_url="https://api.llama-api.com"
        )

        # Encoder l'image en base64
        base64_image = encode_image_to_base64(image_path)

        # Construction du message avec l'image en base64
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

        # Appel à l'API
        response = client.chat.completions.create(
            model="llama3.2-90b-vision",
            messages=messages,
            max_tokens=100,  # Augmenté pour obtenir des descriptions plus détaillées
            temperature=0.7,  # Légèrement réduit pour plus de cohérence
            top_p=0.9,
            n=1
        )

        # Extraction de la réponse
        if isinstance(response.choices, list) and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "Aucune description générée par le modèle."

    except Exception as e:
        return f"Erreur lors de l'analyse de l'image : {str(e)}"

if __name__ == "__main__":
    try:
        resultat = analyse_image_llm('bear.jpg')
        print("\nDescription de l'image :", resultat)
    except Exception as e:
        print(f"Erreur : {str(e)}")