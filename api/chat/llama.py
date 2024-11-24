import requests
import json
from PIL import Image
# Définir l'URL et le payload
url = 'http://localhost:11434/api/generate'

def message_llama(prompt, model="llama3.2:1b", max_tokens=100, temperature=0.7, top_p=1, n=1):

    payload = {
    "model": model,
    "system": "Tu es un expert SEO avec plus de 10 ans d'expérience dans l'optimisation de sites web. Tu as une connaissance approfondie des algorithmes de Google, des bonnes pratiques SEO et des dernières tendances en matière de référencement naturel. Tu es capable d'analyser en détail les aspects techniques et le contenu d'un site pour fournir des recommandations pertinentes et actionnables.",
    "prompt": prompt,
    "max_tokens": max_tokens,  # Nombre maximal de tokens à générer
    "temperature": temperature,  # Température de génération (plus proche de 1 pour plus de créativité)
    "top_p": top_p,  # Découper le noyau des tokens
    "n": n  # Nombre de réponses à générer (ici, une seule réponse)
    }
    # Convertir le payload en une chaîne JSON
    data = json.dumps(payload)

    # Faire la requête POST
    response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
    
    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Diviser la réponse par lignes
        responses = response.text.splitlines()

        # Variable pour accumuler la réponse complète
        complete_response = ""

        # Traiter chaque ligne de la réponse
        for line in responses:
            try:
                # Décoder chaque ligne JSON
                response_json = json.loads(line)
                
                # Ajouter la partie de réponse à l'accumulateur
                complete_response += response_json.get("response", "")
                
                # Vérifier si le modèle a fini de générer la réponse
                if response_json.get("done", False):
                    break  # Fin de la génération
                
            except json.JSONDecodeError as e:
                print(f"Erreur lors du décodage de JSON : {e}")
        
    else:
        print(f"Erreur {response.status_code}: {response.text}")
    return complete_response

if __name__ == "__main__":
    prompt = "Ecris un scénario de film de fesses ?"
    
    # Appel à la fonction pour générer une réponse
    response = message_llama(prompt)