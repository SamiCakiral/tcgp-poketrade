import os
import glob
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import argparse

# --- Configuration ---
CARDS_DIR = "IA/pokemon_cards_images"  # Répertoire racine des cartes
DB_PATH = "pokemon_card_embeddings.npz"  # Fichier de sortie pour la base de données
MODEL_INPUT_SIZE = (224, 224)
# --- Fin Configuration ---

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
        return features.flatten() # Aplatir le vecteur
    except Exception as e:
        print(f"Erreur lors du calcul de l'embedding: {e}")
        return None

def parse_label_from_filename(filename, series_name):
    """Extrait le numéro et le nom de la carte à partir du nom de fichier."""
    parts = os.path.splitext(filename)[0].split('_')
    try:
        # Format attendu: Set_Num_Nom... ou Num_Nom...
        if len(parts) >= 2:
            card_number_str = parts[1] if parts[0] == series_name else parts[0] # Ajuster si le nom de série est au début
            card_number = int(card_number_str)
            card_name = '_'.join(parts[2:]) if parts[0] == series_name else '_'.join(parts[1:])
            label = f"{series_name}_{card_number:03d}_{card_name}" # Format standardisé: SERIE_NUM_NOM
            return label, card_number
        else:
            print(f"Format de nom de fichier non reconnu: {filename}")
            return None, None
    except ValueError:
         # Cas où le numéro n'est pas au début ou après le nom de série (ex: juste le nom)
        try:
            # Essayer d'extraire un numéro quelque part
            numbers = [int(p) for p in parts if p.isdigit()]
            if numbers:
                card_number = numbers[0]
                card_name = os.path.splitext(filename)[0] # Utiliser le nom complet si parsing difficile
                label = f"{series_name}_{card_number:03d}_{card_name}"
                return label, card_number
            else:
                 print(f"Impossible d'extraire le numéro de: {filename}")
                 return None, None
        except Exception as e_inner:
            print(f"Erreur de parsing complexe pour {filename}: {e_inner}")
            return None, None
    except Exception as e:
        print(f"Erreur de parsing pour {filename}: {e}")
        return None, None


def build_database(cards_root_dir, output_db_path):
    """Construit la base de données d'embeddings à partir des images."""
    print("Chargement du modèle ResNet50...")
    # include_top=False pour enlever la couche de classification
    # pooling='avg' pour obtenir un vecteur de caractéristiques fixe
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    print("Modèle chargé.")

    all_embeddings = []
    all_labels = []

    print(f"Parcours du répertoire des cartes : {cards_root_dir}")
    series_dirs = [d for d in os.listdir(cards_root_dir) if os.path.isdir(os.path.join(cards_root_dir, d))]

    if not series_dirs:
        print(f"ERREUR: Aucun sous-répertoire (série) trouvé dans {cards_root_dir}")
        return

    for series_name in tqdm(series_dirs, desc="Traitement des séries"):
        series_path = os.path.join(cards_root_dir, series_name)
        print(f"\nTraitement de la série : {series_name}")

        image_files = glob.glob(os.path.join(series_path, "*.jpg")) + \
                      glob.glob(os.path.join(series_path, "*.png")) + \
                      glob.glob(os.path.join(series_path, "*.webp"))

        if not image_files:
            print(f"Aucune image trouvée pour la série {series_name}")
            continue

        for img_path in tqdm(image_files, desc=f"Cartes {series_name}", leave=False):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Avertissement: Impossible de lire l'image {img_path}")
                continue

            embedding = compute_embedding(image, model)
            if embedding is not None:
                filename = os.path.basename(img_path)
                label, _ = parse_label_from_filename(filename, series_name)

                if label:
                    all_embeddings.append(embedding)
                    all_labels.append(label)
                else:
                     print(f"Avertissement: Ignorer l'image car l'étiquette n'a pas pu être extraite: {filename}")

    if not all_embeddings:
        print("ERREUR: Aucun embedding n'a pu être calculé. Vérifiez les images et les chemins.")
        return

    # Convertir en tableaux NumPy
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels)

    print(f"\nCalcul terminé. {embeddings_array.shape[0]} embeddings générés.")
    print("Sauvegarde de la base de données...")

    # Sauvegarder en utilisant np.savez_compressed pour l'efficacité
    np.savez_compressed(output_db_path, embeddings=embeddings_array, labels=labels_array)
    print(f"Base de données sauvegardée dans : {output_db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construire une base de données d'embeddings pour les cartes Pokémon.")
    parser.add_argument("--cards_dir", type=str, default=CARDS_DIR, help="Répertoire racine contenant les images de cartes organisées par série.")
    parser.add_argument("--db_path", type=str, default=DB_PATH, help="Chemin du fichier .npz où sauvegarder la base de données.")
    args = parser.parse_args()

    build_database(args.cards_dir, args.db_path) 