import os
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
from collections import defaultdict, Counter
import re
import math
import traceback # Pour le débogage

# --- Configuration Globale (utilisée si non surchargée) ---
DEFAULT_DB_PATH = "pokemon_card_embeddings.npz"
DEFAULT_YOLO_MODEL_PATH = "IA/yolov8x_finetune_mps/weights/yolo-carte-empty.pt"
DEFAULT_MODEL_INPUT_SIZE = (224, 224)
DEFAULT_SIMILARITY_THRESHOLD = 0.65 # Seuil initial pour l'identification individuelle (un peu plus bas pour avoir plus de candidats)
# GRID_CONFIDENCE_THRESHOLD = 0.60 # On n'utilise plus ce seuil spécifique pour la référence
MIN_CONSISTENT_CARDS_FOR_ANCHOR = 2 # Nombre minimum de cartes cohérentes pour valider une hypothèse de départ
ANCHOR_SCORE_WEIGHTING = True # Utiliser la similarité comme poids dans le score de cohérence
# --- Fin Configuration Globale ---

# --- Fonctions Utilitaires (restent en dehors de la classe pour clarté) ---

def compute_embedding(image, model, input_size=DEFAULT_MODEL_INPUT_SIZE):
    """Calcule l'embedding ResNet50 pour une image."""
    if image is None or image.size == 0:
        print("Avertissement: Image vide fournie pour l'embedding.")
        return None
    try:
        img_resized = cv2.resize(image, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = preprocess_input(img_array)
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Erreur lors du calcul de l'embedding: {e}")
        return None

def load_embedding_database(db_path):
    """Charge la base de données d'embeddings."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Fichier base de données '{db_path}' non trouvé. Exécutez create_embedding_database.py d'abord.")
    try:
        data = np.load(db_path)
        embeddings = data['embeddings']
        labels = data['labels']
        print(f"Base de données chargée depuis {db_path}: {len(labels)} cartes.")
        if embeddings.ndim == 1: # Corriger si aplati par erreur
             embeddings = embeddings.reshape(len(labels), -1)
        return embeddings, labels
    except Exception as e:
        raise IOError(f"ERREUR lors du chargement de la base de données '{db_path}': {e}")

def build_knn_index(embeddings):
    """Construit l'index NearestNeighbors pour une recherche rapide."""
    print("Construction de l'index KNN...")
    if embeddings is None or embeddings.shape[0] == 0:
        raise ValueError("Impossible de construire l'index KNN : pas d'embeddings.")
    if embeddings.ndim != 2:
        raise ValueError(f"Les embeddings doivent être un tableau 2D, mais ont la forme {embeddings.shape}")

    knn_index = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine') # On n'a besoin que du plus proche
    knn_index.fit(embeddings)
    print("Index KNN construit.")
    return knn_index

def derive_series_info(all_labels):
    """Extrait la liste complète des numéros de carte pour chaque série."""
    series_cards = defaultdict(set)
    # Regex améliorée pour gérer divers formats de labels (SERIE_NUM_... ou SERIE_NUM)
    pattern = re.compile(r"^([^_]+)_(\d+)(?:_.*)?$")

    for label in all_labels:
        match = pattern.match(label)
        if match:
            series_name = match.group(1)
            card_number = int(match.group(2))
            series_cards[series_name].add(card_number)
        # else:
        #     print(f"Avertissement: Format de label non reconnu pour dériver série/numéro: {label}")

    series_info = {name: sorted(list(numbers)) for name, numbers in series_cards.items()}
    print(f"Informations dérivées pour {len(series_info)} séries.")
    return series_info

def determine_grid_structure(detections, y_gap_threshold_ratio=0.65):
    """Détermine les lignes, colonnes et la position de chaque détection dans la grille."""
    if not detections:
        return [], 0, 0, {} # grid, num_rows, num_cols, grid_map

    centers_with_id = []
    heights = []
    widths = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers_with_id.append({'id': d['id'], 'cx': cx, 'cy': cy})
        heights.append(y2 - y1)
        widths.append(x2 - x1)

    if not heights:
        return [], 0, 0, {}

    avg_height = np.mean(heights)
    centers_with_id.sort(key=lambda c: (c['cy'], c['cx']))
    y_coords = [c['cy'] for c in centers_with_id]
    y_gaps = np.diff(y_coords)
    row_separation_threshold = avg_height * y_gap_threshold_ratio
    split_indices = np.where(y_gaps > row_separation_threshold)[0]
    grid_rows_data = []
    start_idx = 0
    for end_idx in split_indices:
        grid_rows_data.append(centers_with_id[start_idx: end_idx + 1])
        start_idx = end_idx + 1
    grid_rows_data.append(centers_with_id[start_idx:])

    grid = []
    grid_map = {} # map detection_id -> (row, col)
    num_cols = 0
    for row_index, row_data in enumerate(grid_rows_data):
        row_data.sort(key=lambda c: c['cx'])
        current_row_ids = []
        for col_index, detection_info in enumerate(row_data):
            det_id = detection_info['id']
            grid_map[det_id] = (row_index, col_index)
            current_row_ids.append(det_id)
        if current_row_ids:
            grid.append(current_row_ids)
            num_cols = max(num_cols, len(current_row_ids))

    num_rows = len(grid)
    print(f"Structure de grille (Gap Detection) : {num_rows} lignes, {num_cols} colonnes.")
    return grid, num_rows, num_cols, grid_map

# --- Classe Principale pour l'Analyse ---

class ScreenshotAnalyzer:
    """
    Classe pour analyser des screenshots de cartes Pokémon, identifier les cartes
    et déterminer les cartes manquantes en utilisant YOLO et des embeddings visuels.
    """
    def __init__(self, db_path=DEFAULT_DB_PATH, yolo_model_path=DEFAULT_YOLO_MODEL_PATH):
        """
        Initialise l'analyseur en chargeant les modèles et la base de données.

        Args:
            db_path (str): Chemin vers le fichier .npz de la base de données d'embeddings.
            yolo_model_path (str): Chemin vers le modèle YOLOv8 entraîné (.pt).
        """
        self.db_path = db_path
        self.yolo_model_path = yolo_model_path
        self.model_input_size = DEFAULT_MODEL_INPUT_SIZE

        self.db_embeddings, self.db_labels = load_embedding_database(self.db_path)
        print("Chargement du modèle ResNet50 pour l'inférence...")
        self.embedding_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("Modèle ResNet50 chargé.")
        if not os.path.exists(self.yolo_model_path):
             raise FileNotFoundError(f"Modèle YOLO non trouvé à '{self.yolo_model_path}'")
        print(f"Chargement du modèle YOLO depuis {self.yolo_model_path}...")
        self.yolo_model = YOLO(self.yolo_model_path)
        # Utiliser le device MPS si disponible sur Mac
        if 'mps' in self.yolo_model.device.type:
             print("Utilisation du device MPS pour YOLO.")
        else:
             print(f"Utilisation du device {self.yolo_model.device.type} pour YOLO.")
        print("Modèle YOLO chargé.")
        self.knn_index = build_knn_index(self.db_embeddings)
        self.series_info = derive_series_info(self.db_labels)
        # Précalculer les embeddings normalisés pour la similarité cosinus
        self.db_embeddings_norm = self.db_embeddings / np.linalg.norm(self.db_embeddings, axis=1, keepdims=True)

    def analyze(self, screenshot_path, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
        """
        Analyse un screenshot donné.

        Args:
            screenshot_path (str): Chemin vers l'image du screenshot.
            similarity_threshold (float): Seuil de similarité (0-1) pour l'identification.

        Returns:
            dict: Un dictionnaire contenant:
                  'identified_cards' (dict): Cartes identifiées groupées par série.
                  'missing_cards' (dict): Cartes manquantes groupées par série.
                  'annotated_image' (np.ndarray): L'image du screenshot avec annotations.
                  'error' (str | None): Message d'erreur si l'analyse échoue.
        """
        print(f"\nAnalyse du screenshot: {screenshot_path}")
        img_screenshot = cv2.imread(screenshot_path)
        if img_screenshot is None:
            print(f"ERREUR: Impossible de charger l'image du screenshot: {screenshot_path}")
            return {"error": f"Impossible de charger l'image: {screenshot_path}"}

        # 1. Détection YOLO (cartes ET vides)
        print("Détection des cartes et emplacements vides avec YOLO...")
        try:
            # Utiliser stream=True pour potentiellement économiser de la mémoire sur de grosses images
            results = self.yolo_model(img_screenshot, verbose=False, stream=False) # stream=False pour avoir tous les résultats d'un coup
        except Exception as e:
             print(f"Erreur lors de l'inférence YOLO: {e}")
             return {"error": f"Erreur YOLO: {e}"}
        print("Détection terminée.")

        all_detections = [] # Stocker {'id': unique_id, 'bbox': ..., 'is_card': True/False, 'yolo_conf': ...}
        initial_identifications = {} # Stocker les identifications initiales par embedding

        det_id_counter = 0
        for r in results: # YOLO v8 renvoie une liste (souvent d'un seul élément)
            boxes = r.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                is_card = (cls_id == 1) # 1 = card, 0 = empty (à vérifier selon ton modèle)
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_screenshot.shape[1], x2), min(img_screenshot.shape[0], y2)
                if x1 >= x2 or y1 >= y2: continue # Ignorer bbox invalide

                detection_data = {
                    "id": det_id_counter,
                    "bbox": [x1, y1, x2, y2],
                    "is_card": is_card,
                    "yolo_conf": float(box.conf[0]),
                    "initial_series": None,
                    "initial_number": None,
                    "initial_similarity": 0.0
                }
                all_detections.append(detection_data)

                # Si c'est une carte, tenter l'identification par embedding
                if is_card:
                    card_img = img_screenshot[y1:y2, x1:x2]
                    query_embedding = compute_embedding(card_img, self.embedding_model, self.model_input_size)
                    if query_embedding is not None:
                         # Normaliser l'embedding requête
                         query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
                         # Calculer la similarité cosinus (produit scalaire sur vecteurs normalisés)
                         similarities = np.dot(self.db_embeddings_norm, query_embedding_norm)
                         best_match_index = np.argmax(similarities)
                         similarity = similarities[best_match_index]

                         if similarity >= similarity_threshold:
                            identified_label = self.db_labels[best_match_index]
                            match = re.match(r"^([^_]+)_(\d+)(?:_.*)?$", identified_label)
                            if match:
                                series_name = match.group(1)
                                card_number = int(match.group(2))
                                detection_data["initial_series"] = series_name
                                detection_data["initial_number"] = card_number
                                detection_data["initial_similarity"] = similarity
                                initial_identifications[det_id_counter] = detection_data

                det_id_counter += 1

        print(f"{len(all_detections)} emplacements détectés au total.")
        if not initial_identifications:
            print("Aucune carte n'a pu être identifiée initialement avec le seuil requis.")
            # On pourrait quand même essayer de déduire la grille si des emplacements sont détectés
            # mais on ne pourra pas calculer les numéros manquants sans référence.
            # Pour l'instant, on s'arrête là si pas d'identification initiale.
            annotated_img = img_screenshot.copy() # Juste pour retourner quelque chose
            for det in all_detections:
                 x1,y1,x2,y2 = det['bbox']
                 color = (255,0,0) # Bleu si pas identifié
                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                 cv2.putText(annotated_img, "?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return {
                 "identified_cards": {}, "missing_cards": {}, "annotated_image": annotated_img,
                 "error": "Identification initiale échouée pour toutes les cartes."
            }

        # 2. Détermination Grille (relativement fiable grâce aux vides aussi)
        print("Détermination structure grille...")
        grid_structure, num_rows, num_cols, grid_map = determine_grid_structure(all_detections)
        if num_cols != 5 and num_cols !=0: # Vérification cruciale de la contrainte des 5 colonnes
            print(f"AVERTISSEMENT: La grille détectée n'a pas 5 colonnes (détecté: {num_cols}). La numérotation sera probablement fausse.")
            # On pourrait tenter de forcer 5 colonnes si num_cols est proche (e.g., 4 ou 6),
            # mais pour l'instant on continue en signalant le problème potentiel.
            num_cols = 5 # Forcer 5 colonnes pour le calcul, même si la détection est bizarre
            print("Forçage à 5 colonnes pour les calculs.")

        # 3. Recherche de l'Ancre la Plus Cohérente
        best_start_num_hypothesis = None
        highest_consistency_score = -1
        best_hypothesis_details = {}

        print("Recherche de l'ancre de numérotation la plus cohérente...")
        if not initial_identifications:
             print("Aucune identification initiale pour baser la recherche d'ancre.")
        else:
            # Tester chaque carte initialement identifiée comme point de départ potentiel
            for ref_det_id, ref_id_info in initial_identifications.items():
                if ref_det_id not in grid_map: continue # Ignorer si pas placée dans la grille

                ref_row, ref_col = grid_map[ref_det_id]
                ref_num = ref_id_info['initial_number']
                ref_sim = ref_id_info['initial_similarity']

                # Hypothèse: le numéro de départ est...
                current_start_num = ref_num - (ref_row * num_cols + ref_col)
                current_consistency_score = 0
                consistent_cards_count = 0
                consistent_series_votes = Counter()

                # Vérifier la cohérence des autres cartes identifiées avec cette hypothèse
                for check_det_id, check_id_info in initial_identifications.items():
                    if check_det_id not in grid_map: continue

                    check_row, check_col = grid_map[check_det_id]
                    check_num = check_id_info['initial_number']
                    check_sim = check_id_info['initial_similarity']
                    check_series = check_id_info['initial_series']

                    # Numéro attendu basé sur l'hypothèse de départ
                    expected_num = current_start_num + (check_row * num_cols + check_col)

                    if check_num == expected_num:
                        # Cohérent ! Ajouter au score
                        consistent_cards_count += 1
                        if ANCHOR_SCORE_WEIGHTING:
                            current_consistency_score += check_sim # Pondérer par la similarité
                        else:
                            current_consistency_score += 1 # Juste compter les cartes cohérentes
                        consistent_series_votes[check_series] += 1

                # Si cette hypothèse est meilleure et suffisamment supportée
                if consistent_cards_count >= MIN_CONSISTENT_CARDS_FOR_ANCHOR and current_consistency_score > highest_consistency_score:
                    highest_consistency_score = current_consistency_score
                    best_start_num_hypothesis = current_start_num
                    # Déterminer la série majoritaire PARMI les cartes cohérentes avec la MEILLEURE hypothèse
                    major_series_for_best_hypothesis = consistent_series_votes.most_common(1)[0][0] if consistent_series_votes else None
                    best_hypothesis_details = {
                        'start_num': best_start_num_hypothesis,
                        'score': highest_consistency_score,
                        'supporting_cards': consistent_cards_count,
                        'ref_card_id': ref_det_id, # Garder une trace de quelle carte a généré cette hypothèse
                        'major_series': major_series_for_best_hypothesis
                    }

        if best_start_num_hypothesis is not None:
            print(f"Meilleure hypothèse trouvée: Numéro de départ = {best_hypothesis_details['start_num']}, "
                  f"Score = {best_hypothesis_details['score']:.2f} ({best_hypothesis_details['supporting_cards']} cartes cohérentes), "
                  f"Série Principale = {best_hypothesis_details['major_series']}")
            definitive_start_num = best_start_num_hypothesis
            major_series = best_hypothesis_details['major_series']
            reference_found = True
        else:
            print("Aucune hypothèse de numérotation de départ suffisamment cohérente trouvée.")
            definitive_start_num = None
            major_series = None # On ne peut pas déterminer la série non plus
            reference_found = False

        # 4. Assignation Finale & Calcul Manquants
        final_identified_cards = defaultdict(set)
        final_missing_cards = defaultdict(set)
        annotated_img = img_screenshot.copy()
        slot_details = {}

        print("Assignation finale des numéros et statut...")
        for det in all_detections:
            det_id = det["id"]
            x1, y1, x2, y2 = det["bbox"]
            color = (255, 0, 0); label_text = "?" # Défauts

            if det_id in grid_map:
                row, col = grid_map[det_id]
                predicted_number = None
                status = "Inconnu"
                is_plausible = False

                if reference_found and definitive_start_num is not None:
                    predicted_number = definitive_start_num + (row * num_cols + col)
                    label_text = f"S{predicted_number}"
                    status = "Manquant (Slot Vide)" if not det["is_card"] else "Présent"

                    if major_series and major_series in self.series_info:
                         if predicted_number in self.series_info[major_series]:
                             is_plausible = True
                         else:
                             status += " (Numéro Implausible)"
                             label_text += " !Plaus"

                    # Logique pour remplir les sets finaux
                    if is_plausible and major_series:
                        if det["is_card"]:
                            final_identified_cards[major_series].add(predicted_number)
                            color = (0, 255, 0) # Vert
                            label_text = f"{major_series} {predicted_number}"
                            # Comparer avec l'identification initiale si elle existe
                            if det_id in initial_identifications:
                                init_id = initial_identifications[det_id]
                                if init_id['initial_number'] != predicted_number:
                                    label_text += f" !{init_id['initial_number']} ({init_id['initial_similarity']:.2f})"
                                    color = (0, 255, 255) # Jaune si mismatch
                                elif init_id['initial_series'] != major_series:
                                     label_text += f" !{init_id['initial_series']} ({init_id['initial_similarity']:.2f})"
                                     color = (0, 165, 255) # Orange si série mismatch
                                else:
                                     label_text += f" ({init_id['initial_similarity']:.2f})" # OK
                        else: # C'est un slot vide
                            final_missing_cards[major_series].add(predicted_number)
                            color = (0, 0, 255) # Rouge
                            label_text = f"Manquant {predicted_number}"
                    elif det["is_card"]: # Présent mais numéro implausible ou pas de série
                         color = (0, 165, 255) # Orange
                         label_text = f"Présent? {predicted_number}" if predicted_number is not None else "Présent?"
                    else: # Vide et numéro implausible
                         color = (100, 100, 100) # Gris foncé
                         label_text = f"Vide? {predicted_number}" if predicted_number is not None else "Vide?"

                    slot_details[(row, col)] = {"predicted_number": predicted_number, "status": status, "is_plausible": is_plausible}

                else: # Pas de référence trouvée
                    label_text = "NoRef"
                    color = (128, 128, 128) # Gris
                    # Essayer d'afficher l'id initiale si elle existe
                    if det['is_card'] and det_id in initial_identifications:
                         init_id = initial_identifications[det_id]
                         label_text = f"{init_id['initial_series']}? {init_id['initial_number']}? ({init_id['initial_similarity']:.2f})"
                         color = (255, 0, 255) # Magenta si carte identifiée mais pas de référence grille

            else: # Hors grille
                label_text = "Hors Grille?"
                color = (128, 0, 128)

            # Annoter
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Renvoyer les résultats finaux
        final_results = {
            "identified_cards": {k: sorted(list(v)) for k, v in final_identified_cards.items()},
            "missing_cards": {k: sorted(list(v)) for k, v in final_missing_cards.items()},
            "annotated_image": annotated_img,
            "major_series": major_series,
            "reference_found": reference_found,
            "grid_rows": num_rows,
            "grid_cols": num_cols, # Le nombre de colonnes utilisé pour le calcul
            "slot_details": slot_details,
            "error": None
        }
        return final_results

# --- Bloc d'Exécution Principal (pour utilisation en ligne de commande SANS argparse) ---

if __name__ == "__main__":
    # --- Configuration pour l'exécution directe ---
    SCREENSHOT_TO_ANALYZE = "IA/background/image1.jpg" # À MODIFIER
    DISPLAY_IMAGE = True
    yolo_path = DEFAULT_YOLO_MODEL_PATH
    db_path = DEFAULT_DB_PATH
    threshold = DEFAULT_SIMILARITY_THRESHOLD # Utilise le seuil initial ici
    # --- Fin de la configuration ---

    if not os.path.exists(SCREENSHOT_TO_ANALYZE):
        print(f"ERREUR: Le fichier screenshot spécifié '{SCREENSHOT_TO_ANALYZE}' n'existe pas.")
        print("Veuillez modifier la variable SCREENSHOT_TO_ANALYZE dans le script.")
        exit(1)

    try:
        analyzer = ScreenshotAnalyzer(db_path=db_path, yolo_model_path=yolo_path)
        results = analyzer.analyze(SCREENSHOT_TO_ANALYZE, similarity_threshold=threshold)

        if results and results.get("error") is None:
            print(f"\n--- Analyse Terminée (Série Principale: {results.get('major_series', 'Non déterminée')}) ---")

            print("\n--- Cartes Présentes (basé sur la grille) ---")
            if results["identified_cards"]:
                for series, numbers in results["identified_cards"].items():
                    print(f"Série {series}: {numbers}")
            else:
                 print("Aucune carte présente déterminée via la grille.")

            print("\n--- Cartes Manquantes (basé sur la grille et les slots vides) ---")
            if results["missing_cards"]:
                for series, missing in results["missing_cards"].items():
                    print(f"Série {series}: Manquants = {missing}")
            else:
                print("Aucune carte manquante déterminée via la grille.")

            if DISPLAY_IMAGE:
                max_display_height = 900; h, w = results["annotated_image"].shape[:2]
                if h > max_display_height: scale = max_display_height / h; display_img = cv2.resize(results["annotated_image"], (int(w * scale), max_display_height))
                else: display_img = results["annotated_image"]
                cv2.imshow("Screenshot Annoté Final", display_img)
                print("\nAppuyez sur une touche pour fermer."); cv2.waitKey(0); cv2.destroyAllWindows()
        elif results:
            print(f"\nErreur lors de l'analyse : {results.get('error')}")

    except (FileNotFoundError, IOError, ValueError, Exception) as e:
        print(f"\nUne erreur critique est survenue : {e}")
        print(traceback.format_exc())
        print("Vérifiez les chemins des fichiers, l'intégrité de la base de données et les dépendances.") 