import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import re
import os
import glob
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import tempfile

# --- Configuration par défaut ---
DB_PATH = "pokemon_cards.db"
IMAGES_DIR = "card_detector/pokemon_cards_images"
EMBEDDINGS_PATH = "card_detector/pokemon_card_embeddings.npz"
MODEL_INPUT_SIZE = (224, 224)

# Sets Pokémon
DEFAULT_SETS = ['A1', 'A1a', 'A2', 'A2a', 'A2b', 'A3']
SET_NAMES = {
    'A1': 'Puissance Génétique',
    'A1a': 'Île Fabuleuse',
    'A2': 'Choc Spatio-Temporelle',
    'A2a': 'Lumière Triomphale',
    'A2b': 'Réjouissances Rayonnantes',
    'A3': 'Gardien Astraux'
}

# --- Fonctions de base de données ---
def initialize_db(db_path):
    """Initialise la base de données avec toutes les tables et colonnes nécessaires."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Création de la table des cartes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        set_id TEXT,
        card_number INTEGER,
        card_name TEXT,
        french_name TEXT,
        image_url TEXT,
        rarity TEXT,
        image_path TEXT,
        UNIQUE(set_id, card_number)
    )
    ''')
    
    # Création de la table des sets
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sets (
        set_id TEXT PRIMARY KEY,
        name TEXT,
        release_date TEXT
    )
    ''')
    
    conn.commit()
    print("Base de données initialisée avec succès.")
    return conn, cursor

# --- Fonctions de scraping ---
def get_pokemon_french_name(english_name):
    """Récupère le nom français d'un Pokémon à partir de son nom anglais via PokéAPI"""
    # Nettoyer le nom pour qu'il corresponde au format de l'API
    clean_name = re.sub(r'\s+ex$', '', english_name.lower())
    clean_name = re.sub(r'[^a-z0-9]', '-', clean_name)
    
    try:
        # Requête à l'API
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{clean_name}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Rechercher le nom français dans les noms
            for name_data in data.get("names", []):
                if name_data.get("language", {}).get("name") == "fr":
                    french_name = name_data.get("name")
                    
                    # Si le Pokémon avait "ex" dans son nom, l'ajouter à la traduction
                    if "ex" in english_name.lower():
                        french_name += " ex"
                    
                    return french_name
        
        # Si le Pokémon n'est pas trouvé ou pas de nom français
        return english_name
    
    except Exception as e:
        print(f"Erreur lors de la traduction de {english_name}: {e}")
        return english_name

def extract_rarity(soup):
    """Extrait la rareté d'une carte à partir de la soupe BeautifulSoup."""
    # Chercher d'abord dans la section des versions actuelles
    rarity_span = soup.select_one("div.prints-current-details span:nth-child(2)")
    if rarity_span:
        rarity_text = rarity_span.text.strip()
        
        # Vérifier les différents formats de rareté
        if "Crown Rare" in rarity_text:
            return "Crown Rare"
        elif "·" in rarity_text:
            # Extraire le symbole après le point médian
            parts = rarity_text.split("·")
            if len(parts) > 1:
                rarity_symbol = parts[1].strip().split()[0]
                
                # Compter les symboles pour déterminer le niveau de rareté
                if "◊" in rarity_symbol:
                    diamond_count = rarity_symbol.count("◊")
                    return f"Diamond {diamond_count}"
                elif "☆" in rarity_symbol:
                    star_count = rarity_symbol.count("☆")
                    return f"Star {star_count}"
    
    # Si rien n'est trouvé, vérifier dans la table des versions
    rarity_cell = soup.select_one("table.card-prints-versions tr.current td:nth-child(2)")
    if rarity_cell:
        rarity_symbol = rarity_cell.text.strip()
        
        if "◊" in rarity_symbol:
            diamond_count = rarity_symbol.count("◊")
            return f"Diamond {diamond_count}"
        elif "☆" in rarity_symbol:
            star_count = rarity_symbol.count("☆")
            return f"Star {star_count}"
    
    # Par défaut
    return "Common"

def scrape_sets(conn, cursor, sets_to_scrape):
    """Scrape les informations de base pour les sets spécifiés."""
    print(f"Scraping des séries : {', '.join(sets_to_scrape)}")
    
    # Mise à jour de la table des sets
    print("  - Mise à jour/Ajout des informations des séries dans la table 'sets'...")
    for set_id in sets_to_scrape:
        # Utiliser .get() pour fournir un nom par défaut si non trouvé dans SET_NAMES
        # Si l'ID est nouveau et pas dans SET_NAMES, un nom générique sera utilisé.
        set_name = SET_NAMES.get(set_id, f"Série {set_id} (Nom Inconnu)") 
        cursor.execute(
            "INSERT OR REPLACE INTO sets (set_id, name) VALUES (?, ?)",
            (set_id, set_name)
        )
        print(f"    - Assuré l'existence/mise à jour de {set_id} avec le nom '{set_name}'")
    conn.commit()
    
    # Scraping des cartes
    total_new_cards = 0
    
    for set_id in sets_to_scrape:
        print(f"\nScraping de la série {set_id} - {SET_NAMES.get(set_id, '')}")
        card_number = 1
        consecutive_failures = 0
        set_new_cards = 0
        
        with tqdm(desc=f"Cartes {set_id}", unit="carte") as pbar:
            while consecutive_failures < 5:  # On arrête après 5 échecs consécutifs
                url = f"https://pocket.limitlesstcg.com/cards/{set_id}/{card_number}"
                try:
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extraction du nom de la carte
                        card_name_elem = soup.select_one('.card-text-name a')
                        if card_name_elem:
                            card_name = card_name_elem.text.strip()
                        else:
                            card_name = f"Card {set_id}-{card_number}"
                        
                        # Extraction de l'URL de l'image
                        image_elem = soup.select_one('.card-image img.card.shadow.resp-w')
                        image_url = image_elem['src'] if image_elem else ""
                        
                        # Extraction de la rareté
                        rarity = extract_rarity(soup)
                        
                        # Obtenir le nom français
                        french_name = get_pokemon_french_name(card_name)
                        
                        # Vérifier si la carte existe déjà
                        cursor.execute(
                            "SELECT id FROM cards WHERE set_id = ? AND card_number = ?",
                            (set_id, card_number)
                        )
                        existing_card = cursor.fetchone()
                        
                        if existing_card:
                            # Mise à jour des informations
                            cursor.execute(
                                "UPDATE cards SET card_name = ?, french_name = ?, image_url = ?, rarity = ? WHERE set_id = ? AND card_number = ?",
                                (card_name, french_name, image_url, rarity, set_id, card_number)
                            )
                            pbar.set_postfix(status="Mise à jour")
                        else:
                            # Insertion nouvelle carte
                            cursor.execute(
                                "INSERT INTO cards (set_id, card_number, card_name, french_name, image_url, rarity) VALUES (?, ?, ?, ?, ?, ?)",
                                (set_id, card_number, card_name, french_name, image_url, rarity)
                            )
                            set_new_cards += 1
                            total_new_cards += 1
                            pbar.set_postfix(status="Nouvelle")
                        
                        conn.commit()
                        consecutive_failures = 0
                        pbar.update(1)
                    else:
                        consecutive_failures += 1
                        pbar.set_postfix(status=f"Échec: HTTP {response.status_code}")
                    
                    card_number += 1
                    
                    
                except Exception as e:
                    print(f"Erreur lors du scraping de {set_id}-{card_number}: {e}")
                    consecutive_failures += 1
                    card_number += 1
                    
                    pbar.update(1)
        
        print(f"Terminé pour la série {set_id}: {set_new_cards} nouvelles cartes ajoutées.")
    
    print(f"\nScraping terminé! Total: {total_new_cards} nouvelles cartes ajoutées.")
    return total_new_cards

def update_missing_data(conn, cursor):
    """Met à jour les données manquantes dans la base de données."""
    # Mise à jour des raretés manquantes
    cursor.execute("SELECT id, set_id, card_number FROM cards WHERE rarity IS NULL OR rarity = ''")
    cards_to_update = cursor.fetchall()
    
    if cards_to_update:
        print(f"\nMise à jour de la rareté pour {len(cards_to_update)} cartes...")
        
        with tqdm(total=len(cards_to_update), desc="Mise à jour des raretés") as pbar:
            for (card_id, set_id, card_number) in cards_to_update:
                url = f"https://pocket.limitlesstcg.com/cards/{set_id}/{card_number}"
                
                try:
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        rarity = extract_rarity(soup)
                        
                        # Mise à jour de la base de données
                        cursor.execute(
                            "UPDATE cards SET rarity = ? WHERE id = ?",
                            (rarity, card_id)
                        )
                        conn.commit()
                        pbar.set_postfix(status=f"Rareté: {rarity}")
                    else:
                        pbar.set_postfix(status=f"Échec: HTTP {response.status_code}")
                        
                    # Pause pour ne pas surcharger le serveur
                    
                    
                except Exception as e:
                    pbar.set_postfix(status=f"Erreur: {str(e)[:20]}")
                    
                
                pbar.update(1)
    
    # Mise à jour des noms français manquants
    cursor.execute("SELECT id, card_name FROM cards WHERE french_name IS NULL OR french_name = ''")
    cards_to_translate = cursor.fetchall()
    
    if cards_to_translate:
        print(f"\nTraduction de {len(cards_to_translate)} noms de cartes...")
        
        with tqdm(total=len(cards_to_translate), desc="Traduction des noms") as pbar:
            for (card_id, card_name) in cards_to_translate:
                # Extraire le nom du Pokémon (sans suffixes comme "V", "VMAX", etc.)
                base_name = re.sub(r'\s+(V|VMAX|GX|EX|ex|V-UNION|BREAK)$', '', card_name)
                
                # Obtenir la traduction
                french_name = get_pokemon_french_name(base_name)
                
                # Ajouter les suffixes à la traduction si présents dans le nom original
                for suffix in ["V", "VMAX", "GX", "EX", "ex", "V-UNION", "BREAK"]:
                    if card_name.endswith(f" {suffix}"):
                        french_name += f" {suffix}"
                
                # Mettre à jour la base de données
                cursor.execute("UPDATE cards SET french_name = ? WHERE id = ?", (french_name, card_id))
                
                if (pbar.n + 1) % 10 == 0:
                    conn.commit()
                
                pbar.update(1)
                pbar.set_postfix(status=f"{french_name[:15]}")
                
                # Pause pour éviter de surcharger l'API
                
        
        conn.commit()
        print("Traduction des noms terminée!")

# --- Fonctions de gestion des images (Supprimée / Non utilisée pour embeddings) ---
# def download_card_images(conn, cursor, images_dir):
#     """Télécharge les images des cartes qui ont une URL mais pas de chemin local."""
#     # ... (code supprimé ou commenté) ...
#     pass # Fonction laissée vide ou supprimée

# --- Fonctions de création d'embeddings ---
def compute_embedding(image, model):
    """Calcule l'embedding ResNet50 pour une image."""
    if image is None:
        return None
    try:
        # Redimensionner et prétraiter
        img_resized = cv2.resize(image, MODEL_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = preprocess_input(img_array)

        # Calculer l'embedding
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()  # Aplatir le vecteur
    except Exception as e:
        print(f"Erreur lors du calcul de l'embedding: {e}")
        return None

def create_embedding_database(conn, cursor, embeddings_path):
    """Crée la base de données d'embeddings en téléchargeant les images à la volée."""
    print("\nCréation de la base de données d'embeddings (à la volée)...")

    # Charger le modèle ResNet50
    print("Chargement du modèle ResNet50...")
    # Assurer la compatibilité GPU si disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limiter la mémoire GPU si nécessaire
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Exemple: 4GB
            for gpu in gpus:
                 tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) détecté(s) et configuré(s).")
        except RuntimeError as e:
            print(f"Erreur configuration GPU: {e}")
            
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    print("Modèle chargé.")

    # Préparer les tableaux pour stocker les embeddings et les labels
    all_embeddings = []
    all_labels = []

    # Récupérer toutes les cartes avec une URL d'image depuis la base de données
    cursor.execute("SELECT set_id, card_number, card_name, french_name, image_url FROM cards WHERE image_url IS NOT NULL AND image_url != ''")
    cards_to_process = cursor.fetchall()

    if not cards_to_process:
        print("Erreur: Aucune carte avec une URL d'image trouvée dans la base de données.")
        return False

    print(f"Traitement de {len(cards_to_process)} cartes pour les embeddings...")

    with tqdm(total=len(cards_to_process), desc="Calcul des embeddings") as pbar:
        for (set_id, card_number, card_name, french_name, image_url) in cards_to_process:
            label = "Unknown"
            embedding = None
            try:
                # Utiliser le nom français s'il est disponible, sinon le nom anglais
                display_name = french_name if french_name else card_name
                if not display_name: # Fallback si les deux sont vides
                    display_name = f"Unknown_{set_id}_{card_number}"
                
                # Créer une étiquette propre pour le fichier et la DB d'embeddings
                clean_display_name = re.sub(r'[\/:*?"<>|\s]+', '_', display_name)
                label = f"{set_id}_{card_number:03d}_{clean_display_name}"

                # Créer un fichier temporaire pour l'image
                # delete=False permet de débugger si nécessaire en gardant le fichier
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_image_file:
                    tmp_path = tmp_image_file.name
                    # Télécharger l'image dans le fichier temporaire
                    response = requests.get(image_url, stream=True, timeout=10) # Ajout timeout
                    if response.status_code == 200:
                        for chunk in response.iter_content(1024):
                            tmp_image_file.write(chunk)
                        tmp_image_file.flush() # S'assurer que tout est écrit

                        # Lire l'image depuis le fichier temporaire
                        image = cv2.imread(tmp_path)
                        if image is not None:
                            # Calculer l'embedding
                            embedding = compute_embedding(image, model)
                            if embedding is not None:
                                all_embeddings.append(embedding)
                                all_labels.append(label)
                                pbar.set_postfix(status=f"OK {label[:20]}")
                            else:
                                pbar.set_postfix(status=f"Échec emb {label[:20]}")
                        else:
                             pbar.set_postfix(status=f"Erreur lecture tmp {label[:20]}")
                    else:
                        pbar.set_postfix(status=f"Échec DL {label[:20]} HTTP {response.status_code}")
            except requests.exceptions.RequestException as e_req:
                 pbar.set_postfix(status=f"Erreur DL {label[:20]}: {str(e_req)[:15]}")
            except Exception as e_gen:
                pbar.set_postfix(status=f"Erreur Gnl {label[:20]}: {str(e_gen)[:15]}")
            finally:
                # Le fichier temporaire est automatiquement supprimé ici si delete=True
                pbar.update(1)
                time.sleep(0.05) # Très courte pause

    if not all_embeddings:
        print("Erreur: Aucun embedding n'a pu être calculé.")
        return False
    
    # Convertir en tableaux NumPy
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels)
    
    print(f"\nCalcul terminé. {embeddings_array.shape[0]} embeddings générés.")
    print(f"Sauvegarde de la base de données d'embeddings dans {embeddings_path}...")
    
    # Sauvegarder les embeddings
    np.savez_compressed(embeddings_path, embeddings=embeddings_array, labels=labels_array)
    print("Base de données d'embeddings sauvegardée avec succès!")
    
    return True

# --- Fonction principale ---
def main():
    # Analyse des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Étend la base de données de cartes Pokémon et crée des embeddings.")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Chemin de la base de données SQLite.")
    parser.add_argument("--images", type=str, default=IMAGES_DIR, help="Répertoire pour stocker les images des cartes.")
    parser.add_argument("--embeddings", type=str, default=EMBEDDINGS_PATH, help="Chemin pour la base de données d'embeddings.")
    parser.add_argument("--sets", type=str, nargs='+', default=DEFAULT_SETS, help="Liste des séries à scraper.")
    parser.add_argument("--skip-scrape", action="store_true", help="Sauter l'étape de scraping.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Sauter l'étape de création des embeddings.")
    
    args = parser.parse_args()
    
    # Initialiser la base de données
    conn, cursor = initialize_db(args.db)
    
    try:
        # Étape 1: Scraping des cartes
        if not args.skip_scrape:
            print("\n=== PHASE 1: SCRAPING DES CARTES ===")
            scrape_sets(conn, cursor, args.sets)
            update_missing_data(conn, cursor)
        
        # Étape 2: Téléchargement des images (Supprimée / Gérée à la volée)
        # print("\n=== PHASE 2: TÉLÉCHARGEMENT DES IMAGES ===")
        # download_card_images(conn, cursor, args.images)
        
        # Étape 2 (anciennement 3): Création des embeddings
        if not args.skip_embeddings:
            print("\n=== PHASE 2: CRÉATION DES EMBEDDINGS (À LA VOLÉE) ===")
            create_embedding_database(conn, cursor, args.embeddings)
        
        print("\nExtension de la base de données terminée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
    
    finally:
        # Fermer la connexion à la base de données
        conn.close()

if __name__ == "__main__":
    main() 