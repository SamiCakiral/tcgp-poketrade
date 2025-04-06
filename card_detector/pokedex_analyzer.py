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
import traceback
import base64

# --- Constantes Configurables ---
# Peut être mis dans un fichier de config ou en variables d'environnement plus tard
DEFAULT_DB_PATH = "card_detector/pokemon_card_embeddings.npz"
DEFAULT_YOLO_MODEL_PATH = "card_detector/yolo-carte-empty.pt"
DEFAULT_MODEL_INPUT_SIZE = (224, 224)
DEFAULT_SIMILARITY_THRESHOLD = 0.65 # Seuil initial pour considérer une ID
HIGH_CONFIDENCE_SIMILARITY = 0.85 # Seuil pour faire confiance à l'ID initiale même si l'ancre est trouvée
MIN_CONSISTENT_CARDS_FOR_ANCHOR = 2
ANCHOR_SCORE_WEIGHTING = True
GRID_COLS_EXPECTED = 5 # Contrainte forte
YOLO_CARD_CLASS_ID = 1 # ID de la classe 'card' dans ton modèle YOLO
YOLO_EMPTY_CLASS_ID = 0 # ID de la classe 'empty'
ANNOTATION_JPEG_QUALITY = 85 # Qualité pour l'image annotée
# --- Fin Constantes ---

# --- Fonctions Utilitaires (privées ou publiques selon besoin) ---
def _compute_embedding(image, model, input_size):
    if image is None or image.size == 0: return None
    try:
        img_resized = cv2.resize(image, input_size); img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0); img_preprocessed = preprocess_input(img_array)
        features = model.predict(img_preprocessed, verbose=0); return features.flatten()
    except Exception as e: print(f"Erreur embedding: {e}"); return None

def _load_embedding_database(db_path):
    if not os.path.exists(db_path): raise FileNotFoundError(f"DB non trouvée: {db_path}")
    try:
        data = np.load(db_path); embeddings = data['embeddings']; labels = data['labels']
        print(f"DB chargée: {len(labels)} cartes."); return embeddings, labels
    except Exception as e: raise IOError(f"Erreur chargement DB: {e}")

def _build_knn_index(embeddings):
    print("Construction index KNN..."); n_neighbors = 1
    if embeddings is None or embeddings.shape[0] < n_neighbors: raise ValueError("Pas assez d'embeddings.")
    if embeddings.ndim != 2: raise ValueError(f"Embeddings non 2D: {embeddings.shape}")
    knn_index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
    knn_index.fit(embeddings); print("Index KNN construit."); return knn_index

def _derive_series_info(all_labels):
    series_cards = defaultdict(set); pattern = re.compile(r"^([^_]+)_(\d+)(?:_.*)?$")
    for label in all_labels:
        match = pattern.match(label)
        if match: series_cards[match.group(1)].add(int(match.group(2)))
    # Créer un dictionnaire de set pour une recherche rapide O(1)
    series_sets = {name: numbers for name, numbers in series_cards.items()}
    print(f"Infos séries dérivées: {len(series_sets)} séries.");
    return series_sets # Retourner le dict de sets

def _determine_grid_structure(detections, y_gap_threshold_ratio=0.65):
    # ... (version précédente basée sur les gaps, semble ok pour l'instant) ...
    if not detections: return [], 0, 0, {}
    centers_with_id = []; heights = []; widths = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']; cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        centers_with_id.append({'id': d['id'], 'cx': cx, 'cy': cy})
        heights.append(y2 - y1); widths.append(x2 - x1)
    if not heights: return [], 0, 0, {}
    avg_height = np.mean(heights); centers_with_id.sort(key=lambda c: (c['cy'], c['cx']))
    y_coords = [c['cy'] for c in centers_with_id]; y_gaps = np.diff(y_coords)
    row_separation_threshold = avg_height * y_gap_threshold_ratio
    split_indices = np.where(y_gaps > row_separation_threshold)[0]
    grid_rows_data = []; start_idx = 0
    for end_idx in split_indices: grid_rows_data.append(centers_with_id[start_idx : end_idx + 1]); start_idx = end_idx + 1
    grid_rows_data.append(centers_with_id[start_idx:])
    grid = []; grid_map = {}; num_cols = 0
    for row_index, row_data in enumerate(grid_rows_data):
        row_data.sort(key=lambda c: c['cx']); current_row_ids = []
        for col_index, detection_info in enumerate(row_data):
            det_id = detection_info['id']; grid_map[det_id] = (row_index, col_index); current_row_ids.append(det_id)
        if current_row_ids: grid.append(current_row_ids); num_cols = max(num_cols, len(current_row_ids))
    num_rows = len(grid)
    print(f"Structure grille détectée: {num_rows} lignes, {num_cols} colonnes.")
    return grid, num_rows, num_cols, grid_map

# --- Classe Principale ---

class PokedexScreenshotAnalyzer:
    """
    Analyse une image de screenshot de Pokédex pour identifier les cartes
    présentes, déterminer une numérotation cohérente et générer une image annotée.
    """
    def __init__(self,
                 db_path=DEFAULT_DB_PATH,
                 yolo_model_path=DEFAULT_YOLO_MODEL_PATH,
                 model_input_size=DEFAULT_MODEL_INPUT_SIZE,
                 preload_models=True):
        """
        Initialise l'analyseur.

        Args:
            db_path (str): Chemin vers le fichier .npz des embeddings.
            yolo_model_path (str): Chemin vers le modèle YOLOv8 .pt.
            model_input_size (tuple): Taille (H, W) pour le modèle d'embedding.
            preload_models (bool): Si True, charge les modèles immédiatement.
                                   Si False, les charge à la première analyse.
        """
        self.db_path = db_path
        self.yolo_model_path = yolo_model_path
        self.model_input_size = model_input_size

        self.db_embeddings = None
        self.db_labels = None
        self.series_info = None # Sera un dict {series_id: set(card_numbers)}
        self.knn_index = None
        self.embedding_model = None
        self.yolo_model = None

        if preload_models:
            self._load_dependencies()

    def _load_dependencies(self):
        """Charge les modèles et la base de données si ce n'est pas déjà fait."""
        if self.db_embeddings is None or self.db_labels is None:
            self.db_embeddings, self.db_labels = _load_embedding_database(self.db_path)
            self.series_info = _derive_series_info(self.db_labels)
            # Construire l'index KNN seulement après avoir chargé les embeddings
            if self.knn_index is None and self.db_embeddings is not None:
                 self.knn_index = _build_knn_index(self.db_embeddings)

        if self.embedding_model is None:
            print("Chargement ResNet50...")
            self.embedding_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            print("ResNet50 chargé.")

        if self.yolo_model is None:
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"Modèle YOLO non trouvé: {self.yolo_model_path}")
            print(f"Chargement YOLO: {self.yolo_model_path}...")
            self.yolo_model = YOLO(self.yolo_model_path)
            print("YOLO chargé.")

    def analyze_image(self, image_np, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, jpeg_quality=ANNOTATION_JPEG_QUALITY):
        """
        Analyse une image (tableau NumPy BGR) de screenshot.

        Args:
            image_np (np.ndarray): L'image à analyser (format OpenCV BGR).
            similarity_threshold (float): Seuil pour l'identification initiale.
            jpeg_quality (int): Qualité pour l'image annotée.

        Returns:
            dict: Résultats structurés de l'analyse pour cette image, incluant :
                  'detections' (list): Liste détaillée de chaque slot détecté.
                  'grid_info' (dict): Infos sur la grille (lignes, colonnes).
                  'anchor_info' (dict): Infos sur l'ancre trouvée (numéro départ, série...).
                  'annotated_image_base64' (str | None): Image annotée encodée en base64.
                  'error' (str | None): Message si une erreur est survenue.
                  'filename' (str): Nom de l'image analysée.
        """
        # Assurer que les dépendances sont chargées
        self._load_dependencies()

        if image_np is None or image_np.size == 0:
            return {"error": "Image fournie est vide ou invalide.", "filename": "unknown_image"}

        filename = "unknown_image" # Valeur par défaut si non fournie autrement

        # 1. Détection YOLO & Identification Initiale
        print(f"[{filename}] Analyse YOLO & Identification initiale...")
        try:
            results = self.yolo_model(image_np, verbose=False)
        except Exception as e:
            return {"error": f"Erreur YOLO: {e}", "filename": filename}

        all_detections = [] # Stocke les infos de chaque slot détecté
        # initial_identifications = {} # Plus nécessaire, infos stockées direct dans all_detections
        det_id_counter = 0

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                is_card = (cls_id == YOLO_CARD_CLASS_ID) # Utiliser la constante
                is_empty = (cls_id == YOLO_EMPTY_CLASS_ID) # Utiliser la constante

                if not is_card and not is_empty: continue # Ignorer autres classes si elles existent

                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                if x1 >= x2 or y1 >= y2: continue

                detection_data = {
                    "id": det_id_counter, "bbox": [x1, y1, x2, y2],
                    "is_card": is_card, "yolo_conf": float(box.conf[0]),
                    # Initialisé à None/0, sera rempli si identifié
                    "initial_series": None, "initial_number": None, "initial_similarity": 0.0,
                    # Infos finales calculées plus tard
                    "row": -1, "col": -1, "predicted_number": None, "status": "Inconnu",
                    "filename": filename # Ajouter filename ici aussi
                }
                all_detections.append(detection_data)

                if is_card:
                    card_img = image_np[y1:y2, x1:x2]
                    query_embedding = _compute_embedding(card_img, self.embedding_model, self.model_input_size)
                    if query_embedding is not None:
                         distances, indices = self.knn_index.kneighbors([query_embedding])
                         distance = distances[0][0]; similarity = max(0.0, 1.0 - distance)
                         # Stocker même si sous le seuil, pour debug éventuel
                         detection_data["initial_similarity"] = similarity
                         if similarity >= similarity_threshold:
                            identified_label = self.db_labels[indices[0][0]]
                            match = re.match(r"^([^_]+)_(\d+)(?:_.*)?$", identified_label)
                            if match:
                                series, num = match.group(1), int(match.group(2))
                                detection_data["initial_series"] = series
                                detection_data["initial_number"] = num
                det_id_counter += 1
        print(f"[{filename}] {len(all_detections)} slots détectés.")

        # Filtrer les cartes initialement identifiées pour l'étape d'ancrage
        initially_identified_cards = [d for d in all_detections if d['is_card'] and d['initial_number'] is not None]
        print(f"[{filename}] {len(initially_identified_cards)} cartes initialement identifiées (seuil {similarity_threshold}).")

        # 2. Détermination Grille
        grid_structure, num_rows, num_cols, grid_map = _determine_grid_structure(all_detections)

        # Appliquer la contrainte de 5 colonnes si nécessaire
        grid_cols_used = GRID_COLS_EXPECTED # Utiliser la contrainte pour les calculs
        if num_cols != GRID_COLS_EXPECTED and num_cols != 0:
             print(f"[{filename}] AVERTISSEMENT: Grille détectée non conforme ({num_cols} colonnes). Utilisation de {GRID_COLS_EXPECTED} colonnes pour le calcul.")
        elif num_cols == 0 and len(all_detections) > 0:
             print(f"[{filename}] AVERTISSEMENT: Grille non détectée mais slots présents. Tentative avec {GRID_COLS_EXPECTED} colonnes.")
        elif num_cols == 0 and len(all_detections) == 0:
             print(f"[{filename}] Aucun slot détecté, impossible de déterminer la grille.")
             # Compter les slots vides pour le front
             empty_slots_count = sum(1 for d in all_detections if not d['is_card'])
             return {"error": "Aucun slot détecté.", "detections": [], "grid_info": {}, "anchor_info": {}, "filename": filename, "empty_slots_count": empty_slots_count}

        # Mettre à jour les positions (row, col) dans all_detections
        for det in all_detections:
            if det['id'] in grid_map:
                det['row'], det['col'] = grid_map[det['id']]

        # 3. Recherche de l'Ancre
        best_start_num_hypothesis = None
        highest_consistency_score = -1
        best_hypothesis_details = {}
        anchor_info = {"status": "Non recherchée", "start_num": None, "major_series": None, "score": 0, "supporting_cards": 0}

        print(f"[{filename}] Recherche de l'ancre...")
        if not initially_identified_cards:
             anchor_info["status"] = "Échec (aucune id initiale)"
             print(f"[{filename}] {anchor_info['status']}")
        else:
            for ref_det in initially_identified_cards:
                if ref_det['row'] == -1: continue # Ignorer si pas dans grille valide

                ref_num = ref_det['initial_number']
                # Ignorer comme référence si le numéro initial est potentiellement une secrète/alternative
                # Heuristique simple : si le numéro > 100 et la série est connue
                ref_series = ref_det['initial_series']
                if ref_series and ref_series in self.series_info and ref_num > 100:
                     # On pourrait affiner avec la taille réelle du set si on l'avait
                     # print(f"[{filename}] Ignorer ref {ref_det['id']} (num {ref_num} > 100)")
                     continue

                current_start_num = ref_num - (ref_det['row'] * grid_cols_used + ref_det['col'])
                current_consistency_score = 0
                consistent_cards_count = 0
                consistent_series_votes = Counter()

                for check_det in initially_identified_cards:
                     if check_det['row'] == -1: continue
                     check_num = check_det['initial_number']
                     expected_num = current_start_num + (check_det['row'] * grid_cols_used + check_det['col'])

                     if check_num == expected_num:
                        consistent_cards_count += 1
                        score_increment = check_det['initial_similarity'] if ANCHOR_SCORE_WEIGHTING else 1
                        current_consistency_score += score_increment
                        consistent_series_votes[check_det['initial_series']] += 1

                if consistent_cards_count >= MIN_CONSISTENT_CARDS_FOR_ANCHOR and current_consistency_score > highest_consistency_score:
                    highest_consistency_score = current_consistency_score
                    best_start_num_hypothesis = current_start_num
                    major_series_for_best = consistent_series_votes.most_common(1)[0][0] if consistent_series_votes else None
                    best_hypothesis_details = {
                        'start_num': best_start_num_hypothesis, 'score': highest_consistency_score,
                        'supporting_cards': consistent_cards_count, 'major_series': major_series_for_best,
                        'ref_card_id': ref_det['id'] # Garder trace de la carte ayant généré la meilleure hypothèse
                    }

            if best_start_num_hypothesis is not None:
                anchor_info = best_hypothesis_details
                anchor_info["status"] = "Trouvée"
                print(f"[{filename}] Ancre trouvée: Num départ={anchor_info['start_num']}, Série={anchor_info['major_series']}, Score={anchor_info['score']:.2f}")
            else:
                anchor_info["status"] = f"Échec (pas assez de cohérence, min {MIN_CONSISTENT_CARDS_FOR_ANCHOR} requis)"
                print(f"[{filename}] {anchor_info['status']}")

        # 4. Assignation Finale des Numéros et Statuts (Logique Modifiée)
        print(f"[{filename}] Assignation finale (logique modifiée)...")
        reference_found = (anchor_info["status"] == "Trouvée")
        definitive_start_num = anchor_info.get("start_num")
        major_series = anchor_info.get("major_series")
        major_series_numbers = self.series_info.get(major_series) if major_series else None

        empty_slots_count = 0 # Compteur pour le front

        for det in all_detections:
            if det['row'] == -1: # Si pas dans la grille
                 det['status'] = "Hors Grille"
                 continue

            initial_num = det.get('initial_number')
            initial_series = det.get('initial_series')
            initial_similarity = det.get('initial_similarity', 0.0)
            has_high_confidence_initial_id = initial_num is not None and initial_similarity >= HIGH_CONFIDENCE_SIMILARITY

            # Calculer le numéro basé sur la grille si possible
            predicted_number_grid = None
            if reference_found and definitive_start_num is not None:
                 predicted_number_grid = definitive_start_num + (det['row'] * grid_cols_used + det['col'])

            # --- Logique d'assignation ---
            if det['is_card']:
                if has_high_confidence_initial_id:
                    # Priorité à l'ID initiale forte
                    det['predicted_number'] = initial_num
                    det['status'] = "Présent (ID Forte)"
                    # Vérifier la plausibilité par rapport à sa propre série initiale si possible
                    initial_series_numbers = self.series_info.get(initial_series) if initial_series else None
                    if initial_series_numbers and initial_num not in initial_series_numbers:
                        # Ce cas est étrange mais on le signale (ID forte mais numéro hors set connu?)
                        det['status'] += " (Numéro Hors Série Initiale?)"
                    elif predicted_number_grid is not None and initial_num != predicted_number_grid:
                        det['status'] += " [vs Grille Mismatch]" # Signaler l'écart avec la grille

                elif reference_found and predicted_number_grid is not None:
                    # Ancre trouvée, mais ID initiale faible ou absente -> on utilise la grille
                    det['predicted_number'] = predicted_number_grid
                    det['status'] = "Présent (Ancré)"
                    # Vérifier la plausibilité par rapport à la série de l'ancre
                    is_plausible_in_anchor_series = major_series_numbers and predicted_number_grid in major_series_numbers
                    if not is_plausible_in_anchor_series:
                         det['status'] += " (Numéro Implausible Série Ancre)"
                    # Signaler si l'ID initiale (même faible) ne correspond pas
                    if initial_num is not None and initial_num != predicted_number_grid:
                        det['status'] += f" [vs ID Initiale {initial_num}?]"

                elif initial_num is not None and initial_similarity >= similarity_threshold:
                     # Pas d'ancre, mais une ID initiale passable
                     det['predicted_number'] = initial_num
                     det['status'] = "Présent (ID Initiale Seule)"
                     initial_series_numbers = self.series_info.get(initial_series) if initial_series else None
                     if initial_series_numbers and initial_num not in initial_series_numbers:
                         det['status'] += " (Numéro Hors Série Initiale?)"

                else:
                    # Pas d'ancre, pas d'ID initiale fiable
                    det['predicted_number'] = None # Important: Ne pas deviner !
                    det['status'] = "Présent? (ID Incertaine)"

            else: # Ce n'est PAS une carte (détecté comme 'empty' ou autre)
                empty_slots_count += 1 # Incrémenter le compteur
                if reference_found and predicted_number_grid is not None:
                    det['predicted_number'] = predicted_number_grid # Numéro attendu à cet emplacement
                    det['status'] = "Manquant (Slot Vide Attendu)"
                    is_plausible_in_anchor_series = major_series_numbers and predicted_number_grid in major_series_numbers
                    if not is_plausible_in_anchor_series:
                        det['status'] += " (Numéro Implausible Série Ancre)"
                else:
                     # Pas d'ancre, on ne sait pas quel numéro devrait être là
                     det['predicted_number'] = None
                     det['status'] = "Manquant (Slot Vide)"

        # === 5. DESSIN DE L'IMAGE ANNOTÉE ===
        print(f"[{filename}] Génération de l'image annotée...")
        annotated_img = image_np.copy()
        for det in all_detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (128, 128, 128) # Gris par défaut
            label_text = "?"; status_for_label = det['status']
            pred_num = det.get('predicted_number', '?')
            # Simplification des labels pour l'image annotée
            if "Hors Grille" in status_for_label: color = (128, 0, 128); label_text = "Hors"
            elif "Présent (ID Forte)" in status_for_label:
                color = (0, 128, 0); # Vert foncé
                series_label = det.get('initial_series', '?')
                label_text = f"{series_label} {pred_num}"
                if "[vs Grille Mismatch]" in status_for_label: label_text += " G!"
            elif "Présent (Ancré)" in status_for_label:
                 color = (0, 255, 0); # Vert clair
                 series_label = major_series or '?'
                 label_text = f"{series_label} {pred_num}"
                 if "Implausible" in status_for_label: color=(255,165,0); label_text+=" P!" # Orange
                 if "[vs ID Initiale" in status_for_label: color=(0, 255, 255); label_text+=" ID!" # Cyan
            elif "Présent (ID Initiale Seule)" in status_for_label:
                 color = (255, 0, 255); # Magenta
                 series_label = det.get('initial_series', '?')
                 label_text = f"{series_label} {pred_num}?"
            elif "Présent? (ID Incertaine)" in status_for_label:
                 color = (200, 200, 0); # Jaune sombre
                 label_text = f"Card?"
            elif "Manquant" in status_for_label:
                 color = (0, 0, 255); # Rouge
                 label_text = f"Vide {pred_num}"
                 if "Implausible" in status_for_label: color=(100,100,100); label_text+=" P!" # Gris foncé
            # Dessiner
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            font_scale = 0.4
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(annotated_img, (x1, y1 - text_height - baseline - 2), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

        # === 6. ENCODAGE BASE64 ===
        print(f"[{filename}] Encodage de l'image annotée...")
        annotated_image_base64 = None; encoding_error = None
        try:
            retval, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if retval: annotated_image_base64 = base64.b64encode(buffer).decode('utf-8'); print(f"[{filename}] Encodage JPEG OK.")
            else: encoding_error = "imencode JPEG a échoué"; print(f"[{filename}] {encoding_error}")
        except Exception as e: encoding_error = f"Exception encodage: {e}"; print(f"[{filename}] {encoding_error}"); traceback.print_exc()

        # === 7. RÉSULTAT FINAL ===
        grid_info = {"rows": num_rows, "cols": num_cols, "cols_used": grid_cols_used}
        final_result = {
            "detections": all_detections, # Liste complète avec toutes les infos
            "grid_info": grid_info,
            "anchor_info": anchor_info,
            "annotated_image_base64": annotated_image_base64,
            "error": encoding_error, # Retourner l'erreur d'encodage s'il y en a eu une
            "filename": filename, # Retourner le nom de fichier pour contexte
            "empty_slots_count": empty_slots_count # Retourner le compte des slots vides
        }
        return final_result


# --- Bloc d'Exécution (pour test rapide, SANS affichage cv2) ---
if __name__ == "__main__":
    print("--- Test du PokedexScreenshotAnalyzer ---")
    SCREENSHOT_TO_ANALYZE = "IA/background/image5.jpg" # À MODIFIER
    if not os.path.exists(SCREENSHOT_TO_ANALYZE):
        print(f"ERREUR: Fichier test '{SCREENSHOT_TO_ANALYZE}' non trouvé.")
        exit(1)

    try:
        # Initialiser avec preload=False pour tester le chargement différé si besoin
        analyzer = PokedexScreenshotAnalyzer(preload_models=True)
        image = cv2.imread(SCREENSHOT_TO_ANALYZE)
        if image is None:
             print(f"ERREUR: Impossible de lire l'image test: {SCREENSHOT_TO_ANALYZE}")
             exit(1)

        # Test avec un seuil de confiance élevé personnalisé
        results = analyzer.analyze_image(image, similarity_threshold=0.70) # Test avec seuil initial plus haut

        if results and results.get("error") is None:
            print("\n--- Résultat de l'Analyse ---")
            print(f"Ancre: {results['anchor_info']}")
            print(f"Grille: {results['grid_info']}")
            print(f"Slots vides comptés: {results.get('empty_slots_count', 'N/A')}")

            # Compter les statuts pour résumé
            status_counts = Counter(d.get('status', 'Erreur') for d in results['detections'])
            print("\nRépartition des statuts:")
            for status, count in status_counts.items():
                print(f"- {status}: {count}")

            # Afficher les détails de quelques détections pour vérification
            print("\nDétails de quelques détections:")
            for i, det in enumerate(results['detections'][:10]): # Afficher les 10 premières
                 print(f"  Slot {i}: ID={det['id']}, Card={det['is_card']}, Pos=({det['row']},{det['col']}), "
                       f"PredNum={det['predicted_number']}, Status='{det['status']}', "
                       f"InitID=({det.get('initial_series', 'N/A')},{det.get('initial_number', 'N/A')}), Sim={det.get('initial_similarity', 0):.2f}")
            if len(results['detections']) > 10: print("  ...")

            # Sauvegarder l'image annotée si elle existe
            if results.get("annotated_image_base64"):
                img_data = base64.b64decode(results["annotated_image_base64"])
                with open("annotated_test_output.jpg", "wb") as f:
                    f.write(img_data)
                print("\nImage annotée sauvegardée dans 'annotated_test_output.jpg'")

        else:
            print(f"\nErreur lors de l'analyse : {results.get('error')}")

    except (FileNotFoundError, IOError, ValueError, Exception) as e:
        print(f"\nErreur critique lors du test : {e}")
        print(traceback.format_exc()) 