from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse, parse_qs
import spacy
from collections import Counter
import concurrent.futures
from datetime import datetime
from chat.llama import message_llama, analyse_image_llm
# from ..chat.recognition_image import analyse_image_llm
from pytrends.request import TrendReq
import time
from statistics import mean
import os

app = Flask(__name__)
CORS(app)

##########################################################################
#                           Initialisation                               #
##########################################################################

# Charger le modèle spaCy pour le traitement NLP
nlp=spacy.load("fr_core_news_sm")

# Fonction pour nettoyer et filtrer les mots-clés
def filter_keywords(text):
    """Filtre les mots-clés en supprimant les stop words et mots peu pertinents."""
    doc = nlp(text)
    filtered_keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    return filtered_keywords

# Fonction pour filtrer les images (formats valides et non w3.org)
def filter_images(images, base_url):
    """Filtre les images en excluant celles de w3.org et ne gardant que les formats valides."""
    valid_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    filtered_images = []

    for image in images:
        try:
            image_url = image["src"]
            # Vérifier le format et l'origine des images
            if 'w3.org' not in image_url and any(image_url.lower().endswith(ext) for ext in valid_formats):
                full_url = urljoin(base_url, image_url)
                filtered_images.append({"src": full_url, "alt": image.get("alt", "").strip()})
        except KeyError as e:
            print(f"Erreur : clé manquante dans une image - {e}")
        except Exception as e:
            print(f"Erreur inattendue lors du traitement d'une image : {e}")
    
    return filtered_images
#Analyse 

# Classe Page pour stocker et analyser les informations SEO de chaque page
class Page:
    def __init__(self, url, base_url):
        self.url = url
        self.base_url = self._get_base_url(url)  # Déterminer automatiquement le base_url
        self.title = ""
        self.meta_description = ""
        self.h1 = ""
        self.text_content = ""
        self.images = []
        self.internal_links = set()
        self.seo_score = 0
        self.score_details = {}
        self.description_image = []
        self.ensemble_description = []
        self.load_time = {}

        # Compteur pour les statistiques
        self.stats = {
            "total_links": 0,
            "valid_links": 0,
            "invalid_links": 0,
            "verified_pages": 0,
            "total_verified_links": 0,  # Total des liens vérifiés
            "valid_internal_links": 0,   # Liens internes valides
            "valid_external_links": 0,   # Liens externes valides
            "invalid_internal_links": 0, # Liens internes invalides
            "invalid_external_links": 0, # Liens externes invalides
            "total_images": 0,           # Total d'images
            "images_with_alt_text": 0,   # Images avec texte alt
            "images_without_alt_text": 0  # Images sans texte alt
        }

        self._fetch_and_parse()
        self.analyze_images()

    def _get_base_url(self, url):
        """Extrait automatiquement le base_url à partir de l'URL fournie."""
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    def _fetch_and_parse(self):
        """Récupère et analyse le contenu HTML de la page pour les éléments SEO, tout en ignorant les erreurs critiques."""
        try:
            start_time = time.time()  # Début du chronométrage
            
            # Récupération de la page
            response = requests.get(self.url, timeout=5)
            response.encoding = response.apparent_encoding  # Set the encoding based on the response
            response.raise_for_status()  # Vérifie pour les erreurs HTTP
            
            # Si la requête réussit, seulement alors analyser avec BeautifulSoup
            if response.status_code == 200:
                end_time = time.time()  # Fin du chronométrage
                self.load_time = round(end_time - start_time, 3)  # Temps de réponse en secondes
                
                self.stats["verified_pages"] += 1  # Incrémenter le compteur de pages vérifiées
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Récupérer le titre
                self.title = soup.title.string if soup.title else ""
                
                # Récupérer la meta description
                meta_desc = soup.find("meta", attrs={"name": "description"})
                self.meta_description = meta_desc["content"] if meta_desc else ""
                
                # Récupérer le H1
                h1_tag = soup.find("h1")
                self.h1 = h1_tag.get_text(strip=True) if h1_tag else ""
                
                # Récupérer les autres titres H2, H3, H4, etc.
                self.headings = {
                    "h2": [h2.get_text(strip=True) for h2 in soup.find_all("h2")],
                    "h3": [h3.get_text(strip=True) for h3 in soup.find_all("h3")],
                    "h4": [h4.get_text(strip=True) for h4 in soup.find_all("h4")],
                }
                
                # Récupérer le texte brut
                self.text_content = soup.get_text(separator=" ", strip=True)
                
                # Collecter et valider les liens internes
                for a_tag in soup.find_all("a", href=True):
                    link = self._is_valid_link(a_tag["href"])
                    self.stats["total_links"] += 1
                    if link:
                        self.internal_links.add(link)  # Un ensemble évite les doublons
                        self.stats["valid_links"] += 1

                # Sauvegarder les liens valides dans un fichier
                existing_links = set()
                # Lire les liens existants pour éviter les doublons
                try:
                    with open("valid_links.txt", "r") as file:
                        existing_links = set(file.read().splitlines())
                except FileNotFoundError:
                    pass  # Si le fichier n'existe pas, on continue

                # Ajouter uniquement les nouveaux liens uniques
                new_links = self.internal_links - existing_links
                with open("valid_links.txt", "a") as file:
                    file.writelines(f"{link}\n" for link in sorted(new_links))
                
                # Collecter les images et analyser leurs descriptions
                images = [
                    {"src": urljoin(self.url, img["src"]), "alt": img.get("alt", "").strip()}
                    for img in soup.find_all("img", src=True)
                ]
                self.images = filter_images(images, self.url)
                self.description_image = {img["src"]: img["alt"] for img in self.images}

                # Mettre à jour les statistiques des images
                self.stats["total_images"] += len(self.images)
                self.stats["images_with_alt_text"] += sum(1 for img in self.images if img["alt"])
                self.stats["images_without_alt_text"] += sum(1 for img in self.images if not img["alt"])
            else:
                print(f"Code HTTP inattendu ({response.status_code}) pour {self.url}")
                self.stats["invalid_links"] += 1

        except requests.HTTPError as http_err:
            if response.status_code == 404:
                print(f"Erreur 404 : La page {self.url} n'existe pas.")
                self.stats["invalid_links"] += 1  # Incrémenter le compteur des liens invalides
            else:
                print(f"Erreur HTTP pour la page {self.url}: {http_err}")
                self.stats["invalid_links"] += 1  # Incrémenter le compteur des liens invalides
        except requests.exceptions.RequestException as req_err:
            # Gérer d'autres erreurs de requêtes (timeout, DNS, etc.)
            print(f"Erreur réseau pour la page {self.url}: {req_err}")
            self.stats["invalid_links"] += 1
        except Exception as e:
            # Gérer toutes les autres erreurs inattendues
            print(f"Erreur lors de l'analyse de la page {self.url}: {e}")
            self.stats["invalid_links"] += 1
        except UnicodeDecodeError as decode_err:
            print(f"Erreur de décodage pour la page {self.url}: {decode_err}")
            self.stats["invalid_links"] += 1  # Increment invalid links count
            return  # Exit the method on decoding error

    def _is_valid_link(self, href):
        """Valide les liens et les transforme en liens absolus si nécessaire."""
        try:
            # Ignorer les ancres et les liens JavaScript
            if href.startswith("#") or href.lower().startswith("javascript:"):
                self.stats["invalid_links"] += 1
                return None

            # Vérifier si le lien est absolu
            if not href.startswith(("http://", "https://")):
                # Construire une URL absolue
                full_url = urljoin(self.base_url, href)
            else:
                full_url = href  # Le lien est déjà absolu

            # Normaliser l'URL
            normalized_url = full_url.rstrip("/")  # Supprime les barres obliques finales

            # Vérifier que le lien est interne
            if urlparse(normalized_url).netloc == urlparse(self.base_url).netloc:
                self.stats["valid_internal_links"] += 1  # Incrémenter pour les liens internes valides
                # Récupérer le chemin après le dernier '/' dans l'URL
                path_after_last_slash = normalized_url.split('/')[-1]  # Obtenir la partie après le dernier '/'

                # Vérifier que le chemin après le dernier '/' n'est pas vide, une ancre ou un lien non valide
                if path_after_last_slash and not path_after_last_slash.startswith(("#", "javascript:", "?")):
                    # Vérifier que l'URL ne se termine pas par des extensions indésirables
                    if not path_after_last_slash.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.webp', '.gif')):
                        # Vérifier le code de statut HTTP
                        response = requests.head(normalized_url, allow_redirects=True)  # Effectuer une requête HEAD
                        if response.status_code == 200:
                            self.stats["verified_pages"] += 1  # Incrémenter le compteur des pages vérifiées
                            return normalized_url  # Retourner le lien HTTPS validé
                        else:
                            self.stats["invalid_links"] += 1  # Incrémenter le compteur des liens invalides
                            return None
                    else:
                        self.stats["invalid_links"] += 1  # Incrémenter le compteur des liens invalides pour les fichiers indésirables
                        return None
                else:
                    self.stats["invalid_links"] += 1
                    return None
            else:
                self.stats["valid_external_links"] += 1  # Incrémenter pour les liens externes valides
                self.stats["invalid_links"] += 1
                with open("external_links.txt", "a") as file:
                    file.write(f"{normalized_url}\n")
                return None
        except Exception as e:
            print(f"Erreur lors de la validation du lien {href} : {e}")
            self.stats["invalid_links"] += 1
            return None

    def analyze_images(self) -> None:
        """Analyse les images en utilisant une fonction externe (OCR)."""
        for image in self.images:
            image_url = image.get('src', '')
            if not image_url:
                continue

            try:
                # Normaliser l'URL de l'image si elle est relative
                if not image_url.startswith(('http://', 'https://')):
                    image_url = urljoin(self.base_url, image_url)

                image_path = download_image(image_url)
                if not image_path:
                    raise ValueError(f"Échec du téléchargement: {image_url}")

                # Analyse de l'image (fonction supposée définie ailleurs)
                description = analyse_image_llm(image_path)
                self.description_image[image_path] = description
                self.ensemble_description.append(
                    f"Image : {image_path}, Description : {description}"
                )

            except Exception as e:
                error_message = f"⚠️ Erreur d'analyse de l'image {image_url}: {str(e)}"
                self.ensemble_description.append(error_message)
                print(error_message)

    def calculate_seo_score(self):
        """Calcule un score SEO amélioré avec détails sur chaque critère."""
        score = 0

        # Critères de base
        self.score_details["Title Presence"] = 10 if self.title else 0
        self.score_details["Title Length"] = 10 if 10 <= len(self.title) <= 60 else 5 if self.title else 0
        self.score_details["Meta Description Presence"] = 10 if self.meta_description else 0
        self.score_details["Meta Description Length"] = (
            10 if 70 <= len(self.meta_description) <= 160 else 5 if self.meta_description else 0
        )
        self.score_details["H1 Presence"] = 10 if self.h1 else 0
        self.score_details["Content Length"] = (
            10 if len(self.text_content.split()) >= 300 else 5 if len(self.text_content.split()) >= 100 else 0
        )
        self.score_details["Images Present"] = 10 if self.images else 0
        self.score_details["Images with Alt Text"] = (
            10 if all(img["alt"] for img in self.images) else 5 if any(img["alt"] for img in self.images) else 0
        )
        self.score_details["Internal Links Present"] = 10 if self.internal_links else 0

        # Critères avancés
        self.score_details["H2 Presence"] = 5 if hasattr(self, "h2") and self.h2 else 0
        self.score_details["H3 Presence"] = 5 if hasattr(self, "h3") and self.h3 else 0
        self.score_details["External Links Present"] = 5 if hasattr(self, "external_links") and self.external_links else 0
        self.score_details["Page Load Time"] = (
            10 if hasattr(self, "load_time") and self.load_time <= 3 else 5 if self.load_time <= 5 else 0
        )

        # Répartition des scores
        self.seo_score = sum(self.score_details.values())/115*100

##########################################################################
#                           SCRAPING DU SITE                               #
##########################################################################

# Fonction pour explorer le site et stocker les pages (nombre de niveau)
MAX_PAGES_TO_CRAWL =1000
def safe_fetch_and_parse(url, base_url):
    """Récupère et analyse le contenu HTML de la page, gérant les erreurs."""
    try:
        response = requests.get(url, timeout=5)
        response.encoding = response.apparent_encoding  # Set the encoding based on the response
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        
        # Si la réponse est valide, procédez à l'analyse
        page = Page(url, base_url)
        page.calculate_seo_score()
        page.analyze_images()
        page._fetch_and_parse()
        return page  # Retourne l'objet Page si tout se passe bien

    except Exception as e:
        print(f"Erreur lors de la récupération ou de l'analyse de {url}: {e}")
        return None  # Retourne None pour indiquer que la page est invalide

def crawl_site(start_url, base_url):
    visited_links = set()  # Ensemble pour garder une trace des liens visités
    all_pages = []  # Liste pour stocker toutes les pages analysées
    links_to_explore = {start_url}  # Ensemble de liens à explorer, initialisé avec l'URL de départ
    pages_crawled = 0  # Compteur de pages analysées
    
    while links_to_explore and pages_crawled < MAX_PAGES_TO_CRAWL:
        next_level_links = set()
        for link in links_to_explore:
            if link not in visited_links:
                visited_links.add(link)
                page = Page(link, base_url)
                page.calculate_seo_score()
                page.analyze_images()
                all_pages.append(page)
                next_level_links.update(page.internal_links - visited_links)
                pages_crawled += 1
        if next_level_links:
            links_to_explore = next_level_links
            time.sleep(1)
        else:
            break
    return all_pages  # Retourner toutes les pages analysées

def download_image(url: str, download_folder: str = "images"):
    """Télécharge une image avec gestion des erreurs et validation."""
    try:
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        # Nettoyer et valider le nom du fichier
        image_name = os.path.basename(url)
        if not image_name:
            image_name = f"image_{hash(url)}"
        
        image_path = os.path.join(download_folder, image_name)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Vérifier le type de contenu
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError(f"Le contenu n'est pas une image: {content_type}")

        with open(image_path, 'wb') as file:
            file.write(response.content)

        return image_path

    except Exception as e:
        print(f"Erreur lors du téléchargement de {url}: {str(e)}")
        return None
##########################################################################
#                           ANALYSE & RECO                               #
##########################################################################
# Fonction pour obtenir des recommandations ChatGPT pour chaque page
def get_gpt_recommendations(page, main_keywords):
    # Construire l'analyse des images
    image_analysis = "\n".join(
        f"- **Image**: {img['src']}\n  **Description (alt)**: {img['alt'] or 'Aucune description'}"
        for img in page.images
    )
    
    # Récupérer les mots-clés tendances pour chaque mot-clé principal
    trending_keywords = []
    for keyword in main_keywords[0:15]:
        trending_keywords.extend(fetch_trending_keywords(keyword, region="FR"))
    
    trending_keywords_text = ", ".join(trending_keywords[:10]) if trending_keywords else "Aucun mot-clé tendance trouvé."
    
    # Génrer le prompt complet pour le modèle
    prompt = f"""
    Analyse cette page et fournis une réponse JSON structurée avec les sections suivantes.
    Ne pas utiliser de markdown, uniquement du texte simple. Passes des lignes entre chaque axe. Ne pas utiliser de "*" ou de "#" pour la mise en forme

    Voici les données à analyser :
    - URL: {page.url}
    - Score SEO: {page.seo_score}/100
    - Titre ({len(page.title)} caractères): {page.title or "Manquant"}
    - Meta description ({len(page.meta_description)} caractères): {page.meta_description or "Manquante"}
    - Contenu textuel: {len(page.text_content.split())} mots
    - Images: {len(page.images)} et la description des images {image_analysis}
    - Liens internes: {len(page.internal_links)}
    - Mots-clés tendances (résultat de Google): {trending_keywords_text}

    Structure ta réponse avec :
    1. Une section "Résumer" avec un bref aperçu en 2-3 phrases
    2. Une section "Points_forts" avec 2-3 éléments positifs
    3. Une section "Points_faibles" avec 2-3 éléments à améliorer
    4. Une section "Recommandations" avec 3-4 actions concrètes à entreprendre
    5. Une section "Mots_cles" avec les suggestions de mots-clés pertinents

    Garde les réponses concises et directes pour faciliter l'affichage.
    """

    try:
        response = message_llama(prompt=prompt)
        return response
    except Exception as e:
        return {"error": f"Erreur d'analyse : {e}"}

#Trouver les mots tendances sur google
def fetch_trending_keywords(keyword, region="FR", num_results=10):
    """Récupère les mots-clés tendances liés à un mot-clé principal."""
    try:
        time.sleep(1)
        
        pytrends = TrendReq(
            hl="fr-FR",
            tz=360,
            timeout=(10, 25),
            retries=2,
            requests_args={
                'verify': True,
            }
        )
        
        trending_keywords = []
        for kw in keyword:
            suggestions = pytrends.suggestions(keyword=kw)
            trending_keywords.extend(suggestion['title'] for suggestion in suggestions)
        
        return trending_keywords 
        
    except Exception as e:
        print(f"Erreur lors de la récupération des tendances pour {keyword}: {e}")
        return []

# Fais un résumé sur l'ensemble des indicateurs SEO
def generate_seo_summary(all_pages, keyword_counts):
    """Génère une synthèse SEO globale basée sur les données des pages et les mots-clés."""
    
    # Calcul des indicateurs clés (KPIs)
    total_pages = len(all_pages)
    avg_word_count = mean(len(page.text_content.split()) for page in all_pages)  # Nombre moyen de mots par page
    total_images = sum(len(page.images) for page in all_pages)
    images_without_alt = sum(1 for page in all_pages for img in page.images if not img["alt"])  # Images sans description
    total_internal_links = sum(len(page.internal_links) for page in all_pages)
    avg_title_length = mean(len(page.title) for page in all_pages if page.title)
    avg_meta_desc_length = mean(len(page.meta_description) for page in all_pages if page.meta_description)

    # Top mots-clés
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(10)]

    # Résumé concis pour la synthèse SEO
    summary_prompt = f"""
    Analyse cette page et fournis une réponse JSON structurée avec les sections suivantes.
    Ne pas utiliser de markdown, uniquement du texte simple. Passes des lignes entre chaque axe.

    Données à analyser :
    - Pages analysées : {total_pages}
    - Mots moyens par page : {avg_word_count:.1f}
    - Total d'images : {total_images} (dont {images_without_alt} sans description alt)
    - Liens internes : {total_internal_links}
    - Longueur moyenne des titres : {avg_title_length:.1f}
    - Longueur moyenne des meta descriptions : {avg_meta_desc_length:.1f}
    - Mots-clés principaux : {', '.join(top_keywords)}

    Structure ta réponse avec :
    1. Une section "kpis_resume" avec un bref aperçu des métriques clés.
    2. Une section "points_attention" listant 2-3 métriques critiques.
    3. Une section "recommandations" avec 3-4 actions concrètes à entreprendre.
    4. Une section "mots_cles_suggestions" avec des suggestions d'optimisation.

    Garde les réponses concises pour faciliter l'affichage dans l'interface.
    """

    # Appel au modèle pour enrichir les recommandations
    try:
        overall_gpt_recommendations = message_llama(summary_prompt)
        return overall_gpt_recommendations
    except Exception as e:
        return f"Erreur lors de la génération de la synthèse SEO : {e}"


@app.route('/api/app', methods=['POST'])
def analyze_site():
    try:

        open("valid_links.txt", "w").close()
        open("external_links.txt", "w").close()

        data = request.get_json()
        url = data.get('url')

        if url.endswith('/'):
            url = url[:-1]

        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))

        print("Analyse des pages")
        all_pages = crawl_site(url, base_url)
        
        # Sort pages based on SEO score
        sorted_pages = sorted(all_pages, key=lambda p: p.seo_score)

        # Calculate average SEO score
        average_score = sum(page.seo_score for page in all_pages) / len(all_pages) if all_pages else 0

        print("Analyse des images")
        # Analyze OCR for each page's images
        for page in all_pages:
            page.analyze_images()  # Apply OCR to the images

        # Keyword analysis for all pages
        all_text_content = " ".join(page.text_content for page in all_pages)
        all_keywords = filter_keywords(all_text_content)
        keyword_counts = Counter(all_keywords)
        # display_keyword_treemap(keyword_counts)
        # List of pages' detailed information
        page_details = []
        keywords = []
        print("Calcul des insights et reco IA par page")
        for page in sorted_pages:
            # Extraction des mots-clés à partir du contenu texte (limité à 500 caractères)
            keywords = filter_keywords(page.text_content[0:1000] if page.text_content else "")
            print(f"Analyse de {page.url}")
            # Construction des données pour chaque page
            page_data = {
                "url": page.url,
                "seo_score": page.seo_score,
                "color": f"rgba({max(0, 255 - int(255 * page.seo_score / 100))}, {max(0, int(255 * page.seo_score / 70))}, 0, 0.5)",
                # Génération des recommandations GPT
                "gpt_recommendation": get_gpt_recommendations(page, keywords),
                # Analyse d'image OCR
                "image_analysis": page.image_analysis_results if hasattr(page, 'image_analysis_results') else None,
                # Statistiques sur les liens
                "links": {
                    "total_links": page.stats.get("total_links", 0),
                    "valid_links": page.stats.get("valid_links", 0),
                    "invalid_links": page.stats.get("invalid_links", 0),
                    "verified_links": page.stats.get("verified_pages", 0),
                    "valid_internal_links": page.stats.get("valid_internal_links", 0),
                    "valid_external_links": page.stats.get("valid_external_links", 0),
                    "invalid_internal_links": page.stats.get("invalid_internal_links", 0),
                    "invalid_external_links": page.stats.get("invalid_external_links", 0),
                },
                # Statistiques sur les images
                "image_stats": {
                    "total_images": page.stats.get("total_images", 0),
                    "images_with_alt_text": page.stats.get("images_with_alt_text", 0),
                    "images_without_alt_text": page.stats.get("images_without_alt_text", 0),
                }
            }
            page_details.append(page_data)

        print("IA reco générale")
        # Generate SEO summary for all pages
        overall_gpt_recommendations = generate_seo_summary(all_pages, keyword_counts)

        # Global statistics
        verified_pages = sum(page.stats["verified_pages"] for page in all_pages)
        total_links = sum(page.stats["total_links"] for page in all_pages)
        valid_links = sum(page.stats["valid_links"] for page in all_pages)
        invalid_links = sum(page.stats["invalid_links"] for page in all_pages)
        
        # Nouvelles statistiques
        total_verified_links = sum(page.stats["verified_pages"] for page in all_pages)  # Total des liens vérifiés
        valid_internal_links = sum(page.stats["valid_internal_links"] for page in all_pages)  # Liens internes valides
        valid_external_links = sum(page.stats["valid_external_links"] for page in all_pages)  # Liens externes valides
        invalid_internal_links = sum(page.stats["invalid_internal_links"] for page in all_pages)  # Liens internes invalides
        invalid_external_links = sum(page.stats["invalid_external_links"] for page in all_pages)  # Liens externes invalides
        
        # Statistiques sur les images
        total_images = sum(page.stats["total_images"] for page in all_pages)  # Total d'images
        images_with_alt_text = sum(page.stats["images_with_alt_text"] for page in all_pages)  # Images avec texte alt
        images_without_alt_text = sum(page.stats["images_without_alt_text"] for page in all_pages)  # Images sans texte alt

        # Prepare the response data
        response = {
            "summary": {
                "total_pages": len(all_pages),
                "average_score": round(average_score, 1),
                "good_pages": len([page for page in all_pages if page.seo_score >= 80]),
                "fair_pages": len([page for page in all_pages if 60 <= page.seo_score < 80]),
                "poor_pages": len([page for page in all_pages if page.seo_score < 60]),
                "total_links": total_links,
                "valid_links": valid_links,
                "invalid_links": invalid_links,
                "verified_links": verified_pages,
                "total_verified_links": total_verified_links,  # Ajout
                "valid_internal_links": valid_internal_links,  # Ajout
                "valid_external_links": valid_external_links,  # Ajout
                "invalid_internal_links": invalid_internal_links,  # Ajout
                "invalid_external_links": invalid_external_links,  # Ajout
                "total_images": total_images,  # Ajout
                "images_with_alt_text": images_with_alt_text,  # Ajout
                "images_without_alt_text": images_without_alt_text  # Ajout
            },
            "pages": page_details,  # List of detailed information for each page
            "keywords": keyword_counts.most_common(10),  # Top 10 most common keywords across all pages
            "overall_gpt_recommendations": overall_gpt_recommendations,
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
