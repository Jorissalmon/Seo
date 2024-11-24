import json
import requests
import base64

def analyse_image_llm(image):
    # Configuration de l'URL de l'API locale Ollama
    url = 'http://localhost:11434/api/generate'
    
    # Définir les paramètres du modèle
    model = 'llama3.2-vision'  # Modèle utilisé
    prompt = 'Décris ce qui est sur limage en restant conçit et clair'  # Question pour le modèle
    image_path = image  # Chemin de l'image locale à analyser
    
    # Charger et encoder l'image en base64
    try:
        with open(image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')  # Encodage en base64
    except FileNotFoundError:
        return f"Erreur : Fichier non trouvé à l'emplacement {image_path}"
    except Exception as e:
        return f"Erreur lors du chargement de l'image : {e}"
    
    # Construire le payload de la requête
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],  # Transmettre l'image encodée en base64
        "max_tokens": 30,  # Limite du nombre de tokens générés
        "temperature": 0.8,  # Température pour ajuster la créativité
        "top_p": 0.9,  # Découpe pour choisir les meilleurs tokens
        "n": 1  # Nombre de réponses générées
    }
    
    # Envoyer la requête POST à l'API Ollama
    try:
        response = requests.post(
            url,
            data=json.dumps(payload),  # Convertir le payload en JSON
            headers={'Content-Type': 'application/json'}
        )
    except requests.RequestException as e:
        return f"Erreur lors de l'appel à l'API : {e}"
    
    # Afficher la réponse brute pour débogage
    print("Réponse brute de l'API :")
    print(response.text)
    
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
                print(f"Ligne problématique : {line}")
        
        # Afficher et retourner la réponse complète
        print(f"Réponse complète : {complete_response}")
        return complete_response
    else:
        return f"Erreur {response.status_code}: {response.text}"

if __name__ == "__main__":
    # Appel à la fonction pour générer une réponse
    response = analyse_image('./bear.jpg')
    print(response)
