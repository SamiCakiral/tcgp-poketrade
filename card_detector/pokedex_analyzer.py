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
from database.databaseManager import get_db_connection

# --- Constantes Configurables ---
# Peut être mis dans un fichier de config ou en variables d'environnement plus tard
DEFAULT_DB_PATH = "card_detector/pokemon_card_embeddings.npz"
DEFAULT_YOLO_MODEL_PATH = "card_detector/yolo-carte-empty.onnx"
DEFAULT_MODEL_INPUT_SIZE = (224, 224)
DEFAULT_SIMILARITY_THRESHOLD = 0.65 # Seuil initial pour considérer une ID
HIGH_CONFIDENCE_SIMILARITY = 0.85 # Seuil pour faire confiance à l'ID initiale même si l'ancre est trouvée
MIN_CONSISTENT_CARDS_FOR_ANCHOR = 2
ANCHOR_SCORE_WEIGHTING = True
GRID_COLS_EXPECTED = 5 # Contrainte forte
YOLO_CARD_CLASS_ID = 1 # ID de la classe 'card' dans ton modèle YOLO
YOLO_EMPTY_CLASS_ID = 0 # ID de la classe 'empty'
ANNOTATION_JPEG_QUALITY = 85 # Qualité pour l'image annotée
# Raretés considérées comme NON communes (donc rares) - SUPPRIMÉ
# RARE_RARITIES = ['Star 1', 'Star 2', 'Star 3', 'Crown Rare'] 
# Ces raretés ne suivent pas strictement la séquence X1-X5/X6-X0
# --- Fin Constantes ---

# --- Fonctions Utilitaires (privées ou publiques selon besoin) ---
def _compute_iou(boxA, boxB):
    """Calcule l'Intersection over Union (IoU) entre deux boîtes.
    Format des boîtes: [x1, y1, x2, y2]
    """
    # déterminer les coordonnées (x, y) de la boîte d'intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # calculer l'aire de l'intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # calculer l'aire des deux boîtes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # calculer l'union
    unionArea = float(boxAArea + boxBArea - interArea)

    # éviter la division par zéro
    if unionArea == 0:
        return 0

    # calculer l'IoU
    iou = interArea / unionArea

    # retourner la valeur IoU
    return iou

def _compute_embedding(image, model, input_size):
    """Calcule l'embedding d'une image en utilisant le modèle ResNet50.
    
    Args:
        image (np.ndarray): Image au format BGR (OpenCV)
        model: Modèle ResNet50 préchargé
        input_size (tuple): Taille cible pour le redimensionnement (H, W)
        
    Returns:
        np.ndarray: Vecteur d'embedding ou None en cas d'erreur
    """
    if image is None or image.size == 0: 
        print("Erreur: Image vide ou invalide fournie à _compute_embedding")
        return None
    
    try:
        # Prétraitement: redimensionnement et conversion BGR->RGB
        img_resized = cv2.resize(image, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Préparation pour ResNet50
        img_array = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = preprocess_input(img_array)
        
        # Calcul de l'embedding
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except cv2.error as e:
        print(f"Erreur OpenCV lors du prétraitement: {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue dans _compute_embedding: {e}")
        traceback.print_exc(limit=2)
        return None

def _load_embedding_database(db_path):
    """Charge la base de données d'embeddings à partir d'un fichier NPZ.
    
    Args:
        db_path (str): Chemin vers le fichier .npz contenant les embeddings et labels
        
    Returns:
        tuple: (embeddings, labels) ou lève une exception en cas d'erreur
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        IOError: Si le chargement échoue
        ValueError: Si le contenu est invalide
    """
    if not os.path.exists(db_path): 
        raise FileNotFoundError(f"DB non trouvée: {db_path}")
    
    try:
        # Charger le fichier NPZ
        data = np.load(db_path)
        
        # Extraire embeddings et labels
        embeddings = data['embeddings']
        labels = data['labels']
        
        # Vérifier la validité des données
        if len(labels) == 0 or embeddings.shape[0] == 0:
            raise ValueError(f"Base de données vide: {db_path}")
            
        if embeddings.shape[0] != len(labels):
            raise ValueError(f"Incohérence dans la DB: {embeddings.shape[0]} embeddings vs {len(labels)} labels")
        
        # Obtenir quelques statistiques pour le débogage
        unique_series = set()
        for label in labels:
            match = re.match(r"^([^_]+)_\d+(?:_.*)?$", label)
            if match:
                unique_series.add(match.group(1))
                
        print(f"DB chargée: {len(labels)} cartes, {len(unique_series)} séries, dimension embedding: {embeddings.shape[1]}.")
        return embeddings, labels
        
    except Exception as e:
        # Convertir toutes les autres erreurs en IOError
        raise IOError(f"Erreur chargement DB: {e}")

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

# --- Classe Principale ---

class PokedexScreenshotAnalyzer:
    """
    Analyse une image de screenshot de Pokédex pour identifier les cartes
    présentes, déterminer une numérotation cohérente et générer une image annotée.
    Utilise ResNet50 (Keras/TF) pour l'embedding et YOLOv8 (via ONNX) pour la détection.
    (Logique d'ancre et de règles de position supprimée)
    """
    def __init__(self,
                 db_path=DEFAULT_DB_PATH,
                 yolo_model_path=DEFAULT_YOLO_MODEL_PATH,
                 model_input_size=DEFAULT_MODEL_INPUT_SIZE,
                 preload_models=True,
                 use_embedding_cache=True):
        """
        Initialise l'analyseur.

        Args:
            db_path (str): Chemin vers le fichier .npz des embeddings.
            yolo_model_path (str): Chemin vers le modèle YOLOv8 .onnx.
            model_input_size (tuple): Taille (H, W) pour le modèle d'embedding.
            preload_models (bool): Si True, charge les modèles immédiatement.
                                   Si False, les charge à la première analyse.
            use_embedding_cache (bool): Si True, active le cache d'embeddings.
        """
        self.db_path = db_path
        self.yolo_model_path = yolo_model_path
        self.model_input_size = model_input_size
        self.use_embedding_cache = use_embedding_cache

        self.db_embeddings = None
        self.db_labels = None
        self.series_info = None # Sera un dict {series_id: set(card_numbers)}
        self.knn_index = None
        self.embedding_model = None
        self.yolo_model = None
        
        # Cache pour les embeddings calculés
        self.embedding_cache = {}
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

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
                raise FileNotFoundError(f"Modèle YOLO ONNX non trouvé: {self.yolo_model_path}")
            print(f"Chargement YOLO depuis ONNX: {self.yolo_model_path} (nécessite onnxruntime)...")
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                print("Modèle YOLO (backend ONNX) chargé.")
            except ImportError:
                 print("\nERREUR: La bibliothèque 'onnxruntime' est nécessaire mais n'est pas installée.")
                 print("Veuillez l'ajouter à requirements.txt et l'installer: pip install onnxruntime\n")
                 raise
            except Exception as e:
                 print(f"\nERREUR: Échec du chargement du modèle YOLO ONNX : {e}")
                 print("Assurez-vous que 'onnxruntime' est installé et que le fichier .onnx est valide.")
                 print(traceback.format_exc())
                 raise
        
    def _compute_embedding_with_cache(self, card_img):
        """
        Calcule l'embedding d'une image avec mise en cache pour éviter les calculs redondants.
        
        Args:
            card_img (np.ndarray): Image de la carte à traiter
            
        Returns:
            np.ndarray: Vecteur d'embedding ou None en cas d'erreur
        """
        if not self.use_embedding_cache or card_img is None:
            # Si le cache est désactivé ou l'image est invalide, calculer directement
            return _compute_embedding(card_img, self.embedding_model, self.model_input_size)
            
        # Créer une clé de cache basée sur un hachage de l'image
        try:
            # Réduire l'image pour le hachage (plus rapide, toujours distinctif)
            small_img = cv2.resize(card_img, (32, 32))
            cache_key = hash(small_img.tobytes())
            
            if cache_key in self.embedding_cache:
                self.embedding_cache_hits += 1
                if self.embedding_cache_hits % 10 == 0:  # Limiter les logs
                    print(f"Cache d'embeddings: {self.embedding_cache_hits} hits, {self.embedding_cache_misses} misses")
                return self.embedding_cache[cache_key]
            
            # Cache miss - calculer l'embedding
            self.embedding_cache_misses += 1
            embedding = _compute_embedding(card_img, self.embedding_model, self.model_input_size)
            
            # Stocker dans le cache si valide
            if embedding is not None:
                self.embedding_cache[cache_key] = embedding
                
            return embedding
            
        except Exception as e:
            print(f"Erreur dans le cache d'embeddings: {e}")
            # En cas d'erreur, revenir au calcul direct
            return _compute_embedding(card_img, self.embedding_model, self.model_input_size)

    def analyze_image(self, image_np, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, jpeg_quality=ANNOTATION_JPEG_QUALITY):
        """
        Analyse une image (tableau NumPy BGR) de screenshot - Version Simplifiée.
        Détecte les cartes (YOLO), les identifie (ResNet), nettoie les détections
        et détermine une grille spatiale simple (5 colonnes).
        AUCUNE logique d'ancre ou de règles de position n'est appliquée.

        Args:
            image_np (np.ndarray): L'image à analyser (format OpenCV BGR).
            similarity_threshold (float): Seuil pour l'identification initiale.
            jpeg_quality (int): Qualité pour l'image annotée.

        Returns:
            dict: Résultats structurés de l'analyse pour cette image.
        """
        self._load_dependencies()
        if image_np is None or image_np.size == 0:
            return {"error": "Image fournie est vide ou invalide.", "filename": "unknown_image"}
        filename = "unknown_image"

        # 1. Détection YOLO & Identification Initiale (inchangé)
        print(f"[{filename}] Analyse YOLO (via ONNX) & Identification initiale (ResNet50)...")
        try:
            results = self.yolo_model(image_np, verbose=False)
            
            # Logging détaillé des résultats bruts de YOLO
            print(f"[{filename}] --- DEBUG YOLO RAW OUTPUT ---")
            for r_idx, r in enumerate(results):
                boxes = r.boxes.cpu().numpy()
                print(f"[{filename}] Result {r_idx}: Found {len(boxes)} raw boxes")
                for b_idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].astype(int)
                    print(f"[{filename}]   Box {b_idx}: xyxy={xyxy}, conf={conf:.4f}, cls={cls_id}")
            print(f"[{filename}] --- END DEBUG YOLO RAW OUTPUT ---")
        except Exception as e:
            return {"error": f"Erreur YOLO (ONNX): {e}", "filename": filename}

        # 2. Traitement des détections YOLO et identification par ResNet (inchangé)
        all_detections = []
        det_id_counter = 0
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                is_card = (cls_id == YOLO_CARD_CLASS_ID)
                is_empty = (cls_id == YOLO_EMPTY_CLASS_ID)
                if not is_card and not is_empty: continue

                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                if x1 >= x2 or y1 >= y2: continue

                detection_data = {
                    "id": det_id_counter, "bbox": [x1, y1, x2, y2],
                    "is_card": is_card, "yolo_conf": float(box.conf[0]),
                    "initial_series": None, "initial_number": None, "initial_similarity": 0.0,
                    "row": -1, "col": -1,
                    "filename": filename
                }
                all_detections.append(detection_data)

                if is_card:
                    card_img = image_np[y1:y2, x1:x2]
                    query_embedding = self._compute_embedding_with_cache(card_img)
                    if query_embedding is not None:
                         distances, indices = self.knn_index.kneighbors([query_embedding])
                         distance = distances[0][0]; similarity = max(0.0, 1.0 - distance)
                         detection_data["initial_similarity"] = similarity
                         if similarity >= similarity_threshold:
                            identified_label = self.db_labels[indices[0][0]]
                            match = re.match(r"^([^_]+)_(\d+)(?:_.*)?$", identified_label)
                            if match:
                                series, num = match.group(1), int(match.group(2))
                                detection_data["initial_series"] = series
                                detection_data["initial_number"] = num
                det_id_counter += 1
        print(f"[{filename}] {len(all_detections)} slots détectés (bruts).")

        # Nettoyage des détections YOLO (chevauchement, outliers) (inchangé)
        print(f"[{filename}] Nettoyage des détections (chevauchement et éloignement)...")
        detections_to_remove_overlap = set()
        iou_threshold = 0.4 # Seuil pour considérer un chevauchement significatif
        
        # Comparer toutes les paires de détections
        indices = list(range(len(all_detections)))
        for i in range(len(indices)):
            if indices[i] in detections_to_remove_overlap: continue # Déjà marquée pour suppression
            for j in range(i + 1, len(indices)):
                if indices[j] in detections_to_remove_overlap: continue

                idx1 = indices[i]
                idx2 = indices[j]
                det1 = all_detections[idx1]
                det2 = all_detections[idx2]

                iou = _compute_iou(det1['bbox'], det2['bbox'])

                if iou > iou_threshold:
                    # Si chevauchement, garder celle avec la meilleure confiance YOLO
                    if det1['yolo_conf'] >= det2['yolo_conf']:
                        detections_to_remove_overlap.add(idx2)
                    else:
                        detections_to_remove_overlap.add(idx1)

        # Créer une nouvelle liste sans les détections à supprimer
        cleaned_detections_overlap = [det for idx, det in enumerate(all_detections) if idx not in detections_to_remove_overlap]
        print(f"[{filename}] Après filtre chevauchement: {len(cleaned_detections_overlap)} détections.")

        # 2b. Suppression des outliers (trop éloignés)
        if len(cleaned_detections_overlap) > 5: # Ne pas faire si trop peu de détections
            centers = []
            for det in cleaned_detections_overlap:
                x1, y1, x2, y2 = det['bbox']
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            
            centers_np = np.array(centers)
            centroid = np.mean(centers_np, axis=0)
            distances = np.linalg.norm(centers_np - centroid, axis=1)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Seuil pour considérer un outlier (ex: distance > moyenne + 2 * écart-type)
            outlier_threshold_dist = mean_dist + 2.5 * std_dist 
            
            detections_to_remove_outlier = set()
            for idx, dist in enumerate(distances):
                if dist > outlier_threshold_dist:
                    detections_to_remove_outlier.add(idx)
            
            # Créer la liste finale nettoyée
            cleaned_detections_final = [det for idx, det in enumerate(cleaned_detections_overlap) if idx not in detections_to_remove_outlier]
            print(f"[{filename}] Après filtre outliers: {len(cleaned_detections_final)} détections.")
        else:
            # Pas assez de détections pour une analyse d'outliers fiable
            cleaned_detections_final = cleaned_detections_overlap
            print(f"[{filename}] Pas de filtre outliers appliqué (trop peu de détections).")

        # Utiliser la liste nettoyée pour la suite
        # Remplace 'all_detections' par 'cleaned_detections_final' dans les appels suivants
        active_detections = cleaned_detections_final 

        # 3. Détermination Grille (inchangé, utilise le clustering spatial)
        print(f"[{filename}] Détermination de la structure de grille (5 colonnes forcées) sur {len(active_detections)} détections...")
        self.image_name = filename
        num_rows, num_cols, active_detections_with_grid = self._determine_grid_structure(
            active_detections, force_columns=True
        )
        del self.image_name
        grid_cols_used = GRID_COLS_EXPECTED
        if num_rows == 0 or num_cols == 0 and len(active_detections_with_grid) > 0:
            print(f"[{filename}] AVERTISSEMENT: La structure de la grille n'a pas pu être déterminée ...")
        elif len(active_detections_with_grid) == 0:
             print(f"[{filename}] Aucun slot détecté après nettoyage...")
             return {"error": "Aucun slot détecté après nettoyage.", "detections": [], "grid_info": {}, "anchor_info": {}, "filename": filename, "empty_slots_count": 0}

        # Créer yolo_detections_map (inchangé)
        yolo_detections_map = {}
        for det in active_detections_with_grid:
            if det['row'] != -1 and det['col'] != -1:
                yolo_detections_map[(det['row'], det['col'])] = det

        # 4. Recherche de l'Ancre et de la Série Principale (NOUVEAU)
        print(f"[{filename}] Recherche de l'ancre et de la série principale...")
        anchor_info = {"status": "Non trouvée", "row": -1, "col": -1, "series": None, "number": None, "similarity": 0.0}
        major_series = None
        
        # Filtrer les détections candidates pour l'ancre (carte identifiée avec haute confiance)
        anchor_candidates = []
        for det in active_detections_with_grid:
            if (det.get('is_card') 
                and det.get('initial_series') is not None 
                and det.get('initial_number') is not None
                and det.get('row', -1) != -1
                and det.get('col', -1) != -1
                and det.get('initial_similarity', 0.0) >= HIGH_CONFIDENCE_SIMILARITY): 
                anchor_candidates.append(det)
                
        print(f"[{filename}] {len(anchor_candidates)} détections candidates pour l'ancre (similarité >= {HIGH_CONFIDENCE_SIMILARITY}).")

        if anchor_candidates:
            # Compter les séries parmi les candidats
            series_counts = Counter(det['initial_series'] for det in anchor_candidates)
            print(f"[{filename}] Séries trouvées parmi les candidats: {dict(series_counts)}")
            
            # Trouver la série la plus fréquente
            if series_counts:
                most_common = series_counts.most_common(1)[0]
                potential_major_series = most_common[0]
                count = most_common[1]
                
                # Vérifier si elle est suffisamment représentée
                if count >= MIN_CONSISTENT_CARDS_FOR_ANCHOR:
                    major_series = potential_major_series
                    print(f"[{filename}] Série principale déterminée: '{major_series}' (présente {count} fois parmi les candidats fiables).")
                    
                    # Filtrer les candidats appartenant à la série principale
                    major_series_candidates = [c for c in anchor_candidates if c['initial_series'] == major_series]
                    
                    # Choisir la meilleure ancre (celle avec le plus petit numéro initial parmi les candidats fiables de la série majeure)
                    best_anchor = min(major_series_candidates, key=lambda c: c['initial_number'])
                    
                    anchor_info = {
                        "status": "Trouvée",
                        "row": best_anchor['row'],
                        "col": best_anchor['col'],
                        "series": best_anchor['initial_series'],
                        "number": best_anchor['initial_number'],
                        "similarity": best_anchor['initial_similarity']
                    }
                    print(f"[{filename}] Ancre initialement sélectionnée (min numéro): Pos=({anchor_info['row']},{anchor_info['col']}), ID={anchor_info['series']} {anchor_info['number']}, Sim={anchor_info['similarity']:.3f}")

                    # --- NOUVELLE HEURISTIQUE v2: Vérifier si l'ancre implique un début de ligne correct --- 
                    offset_heuristic = 0 # Offset par défaut
                    anchor_r, anchor_c = anchor_info['row'], anchor_info['col']
                    anchor_num_initial = anchor_info['number']
                    current_grid_cols = grid_cols_used

                    # Calculer le numéro que l'ancre impliquerait pour le début de SA PROPRE ligne (colonne 0)
                    implied_start_of_anchor_row = anchor_num_initial + (anchor_r * current_grid_cols + 0) - (anchor_r * current_grid_cols + anchor_c)
                    # Simplifié: implied_start_of_anchor_row = anchor_num_initial - anchor_c
                    print(f"[{filename}][HeuristiqueV2] L'ancre ({anchor_r},{anchor_c}) N°{anchor_num_initial} implique que sa ligne devrait commencer par N°{implied_start_of_anchor_row}.")

                    # Vérifier si ce début de ligne implicite est correct (finit par 1 ou 6 => % 5 == 1)
                    if implied_start_of_anchor_row % 5 != 1:
                        print(f"[{filename}][HeuristiqueV2] Le début de ligne implicite {implied_start_of_anchor_row} est incorrect (num % 5 != 1).")
                        # Calculer quel devrait être le vrai début de cette ligne (en arrondissant vers le bas au 1 ou 6 précédent)
                        correct_start_for_anchor_row = implied_start_of_anchor_row - ((implied_start_of_anchor_row - 1) % 5)
                        print(f"[{filename}][HeuristiqueV2] Le début correct pour cette ligne devrait être {correct_start_for_anchor_row}.")
                        
                        # Calculer l'offset nécessaire pour corriger
                        offset_heuristic = correct_start_for_anchor_row - implied_start_of_anchor_row
                        print(f"[{filename}][HeuristiqueV2] Offset nécessaire calculé: {correct_start_for_anchor_row} (correct) - {implied_start_of_anchor_row} (impliqué) = {offset_heuristic}")
                        print(f"[{filename}][HeuristiqueV2] L'offset {offset_heuristic} sera appliqué aux numéros prédits par l'ancre.")
                    else:
                        print(f"[{filename}][HeuristiqueV2] Le début de ligne implicite {implied_start_of_anchor_row} est correct (num % 5 == 1). Pas d'ajustement (offset=0).")
                    # --- FIN HEURISTIQUE V2 --- #

                else:
                    print(f"[{filename}] La série la plus fréquente ('{potential_major_series}') n'apparaît que {count} fois, seuil minimum {MIN_CONSISTENT_CARDS_FOR_ANCHOR} non atteint. Pas de série principale.")
            else:
                 print(f"[{filename}] Aucune série n'a pu être comptée parmi les candidats.")
        else:
            print(f"[{filename}] Pas de candidats fiables trouvés pour déterminer l'ancre et la série principale.")

        # 5. Construction de la Grille Contextuelle (MODIFIÉ)
        print(f"[{filename}] Construction de la grille contextuelle (ancre: {anchor_info['status']})...")
        self.image_name = filename # Nécessaire pour les logs dans _build...
        final_grid_slots = self._build_contextual_grid(
            grid_rows=num_rows, 
            grid_cols=grid_cols_used, 
            anchor_info=anchor_info, # Contient le numéro ORIGINAL de l'ancre
            major_series=major_series, 
            series_info=self.series_info,
            yolo_detections_map=yolo_detections_map,
            offset_heuristic=offset_heuristic # Passer l'offset calculé
        )
        del self.image_name # Nettoyer l'attribut temporaire

        # Compter les slots vides (inchangé - mais le statut pourrait changer)
        # Note: Le comptage pourrait être affiné pour distinguer les vides détectés des vides prédits
        empty_slots_count = sum(1 for slot_info in final_grid_slots.values() if "Manquant" in slot_info.get('status', ''))

        # 6. DESSIN DE L'IMAGE ANNOTÉE (À AJUSTER ÉVENTUELLEMENT)
        # Les couleurs/textes pourraient être mis à jour pour refléter les nouveaux statuts
        print(f"[{filename}] Génération de l'image annotée (basée sur grille contextuelle)...")
        annotated_img = image_np.copy()
        for pos, slot_info in final_grid_slots.items():
            if slot_info.get('bbox'): # Vérifier si bbox existe (pas le cas pour slots prédits?)
                x1, y1, x2, y2 = slot_info['bbox']
            # else:
            #     # Gérer le cas où il n'y a pas de bbox (slots prédits) - Peut-être dessiner un cadre par défaut?
            #     # Pour l'instant, on ne dessine rien s'il n'y a pas de bbox
            #     continue 
                
            # Ajuster les couleurs et labels en fonction des NOUVEAUX statuts potentiels
            color = (128, 128, 128) # Gris par défaut
            label_text = f"({slot_info.get('row', '?')},{slot_info.get('col', '?')})?"
            status = slot_info.get('status', 'Inconnu')
            pred_num = slot_info.get('predicted_number')
            pred_series = slot_info.get('predicted_series')

            if status == "Présent (Confirmé)":
                color = (0, 128, 0) # Vert foncé
                label_text = f"{pred_series} {pred_num}"
            elif status == "Présent (Corrigé par Ancre)":
                color = (0, 200, 100) # Vert clair
                label_text = f"{pred_series} {pred_num} (C)" # (C) pour Corrigé
            elif status == "Présent (ID Initiale)": # Gardé pour le cas sans ancre ou si l'ancre n'a pas corrigé
                color = (0, 200, 0) 
                label_text = f"{pred_series} {pred_num}"
            elif status == "Présent (ID Incertaine)":
                color = (0, 255, 255) # Jaune
                label_text = f"{pred_series} {pred_num}?"
            elif status == "Présent (Conflit)":
                color = (255, 165, 0) # Orange
                init_num = slot_info.get('initial_id_data', {}).get('number', 'N/A')
                label_text = f"{pred_series} {pred_num} vs {init_num} X" # X pour Conflit
            elif status == "Présent (ID Inconnue)":
                color = (255, 0, 255) # Magenta
                label_text = "ID?"
            elif status == "Manquant (Vide Détecté)":
                color = (0, 0, 255) # Rouge
                label_text = "Vide"
            elif status == "Manquant (Prédit)":
                color = (100, 100, 255) # Rouge clair / Rose
                label_text = f"{pred_series} {pred_num} (P)" # (P) pour Prédit
            # Pas de dessin pour "Hors Série" pour l'instant
            elif status == "Hors Série ou Vide Inconnu": 
                 continue # Ne rien dessiner pour ces cas
                 
            # Dessiner uniquement si on a une bbox
            if slot_info.get('bbox'):
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                font_scale = 0.4
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(annotated_img, (x1, y1 - text_height - baseline - 2), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

        # 7. ENCODAGE BASE64 (inchangé)
        print(f"[{filename}] Encodage de l'image annotée...")
        annotated_image_base64 = None; encoding_error = None
        try:
            retval, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if retval: annotated_image_base64 = base64.b64encode(buffer).decode('utf-8'); print(f"[{filename}] Encodage JPEG OK.")
            else: encoding_error = "imencode JPEG a échoué"; print(f"[{filename}] {encoding_error}")
        except Exception as e: encoding_error = f"Exception encodage: {e}"; print(f"[{filename}] {encoding_error}"); traceback.print_exc()

        # 8. Convertir final_grid_slots en liste de détections pour l'API (À AJUSTER)
        detections = []
        for pos, slot_info in final_grid_slots.items():
            r, c = pos
            # Assurer la cohérence des clés retournées par l'API
            initial_data = slot_info.get('initial_id_data') # Peut être None
            detection = {
                "id": len(detections), # Ou utiliser un ID plus persistant si nécessaire
                "row": r,
                "col": c,
                "predicted_number": slot_info.get('predicted_number'),
                "predicted_series": slot_info.get('predicted_series'),
                "status": slot_info.get('status', 'Inconnu'),
                "bbox": slot_info.get('bbox'), # Peut être None pour slots prédits
                "initial_series": initial_data.get('series') if initial_data else None,
                "initial_number": initial_data.get('number') if initial_data else None,
                "initial_similarity": initial_data.get('similarity') if initial_data else 0.0,
                "is_card": slot_info.get('yolo_detection_type', False), # Vrai si carte détectée par YOLO
                "is_virtual": slot_info.get('status') == "Manquant (Prédit)" # Indiquer si slot prédit
            }
            detections.append(detection)

        # 9. RÉSULTAT FINAL
        grid_info = {"rows": num_rows, "cols": grid_cols_used, "cols_used": num_cols}
        
        # Log des séries initiales détectées
        detected_initial_series = Counter(d.get('initial_series') for d in detections if d.get('initial_series'))
        if detected_initial_series:
            series_summary = ', '.join([f'{s}: {c}' for s, c in detected_initial_series.items()])
            print(f"[{filename}] Résumé des séries initialement identifiées (nombre d'occurrences): {series_summary}")
        else:
            print(f"[{filename}] Aucune série initiale n'a pu être identifiée sur cette image.")
            
        final_result = {
            "detections": detections,
            "grid_info": grid_info,
            "anchor_info": anchor_info, # Maintenant contient les infos de l'ancre trouvée
            "annotated_image_base64": annotated_image_base64,
            "error": encoding_error,
            "filename": filename,
            "empty_slots_count": empty_slots_count,
            "major_series": major_series, # Ajout explicite de la série principale déterminée
            "offset_heuristic": offset_heuristic # AJOUT DE L'OFFSET
        } 
        
        # --- NOUVEAU: Log de la grille finale détectée --- 
        print(f"[{filename}] --- GRILLE FINALE DÉTECTÉE ({anchor_info.get('series') or major_series or 'N/A'}) ---") # Utiliser aussi major_series pour le log si ancre['series'] est None
        if final_grid_slots:
            max_r = max(r for r, c in final_grid_slots.keys()) 
            max_c = max(c for r, c in final_grid_slots.keys())
            # S'assurer qu'on couvre bien jusqu'à grid_info['cols'] même si la dernière colonne est vide
            num_cols_to_print = max(max_c + 1, grid_info.get('cols', 0))
             
            for r in range(max_r + 1):
                row_str = ""
                for c in range(num_cols_to_print):
                    pos = (r, c)
                    slot = final_grid_slots.get(pos)
                    if slot and slot.get('predicted_number') is not None:
                        # Afficher le numéro prédit, paddé sur 3 caractères
                        row_str += f"{slot['predicted_number']: >3} " 
                    elif slot and "Manquant (Vide Détecté)" in slot.get('status', ''):
                        row_str += "VID " # Indiquer un vide détecté
                    else:
                        row_str += "--- " # Placeholder pour slot non trouvé ou prédit manquant
                print(f"[{filename}][Ligne {r}] {row_str.rstrip()}")
        else:
             print(f"[{filename}] Aucune grille finale à afficher (final_grid_slots est vide).")
        print(f"[{filename}] --- FIN GRILLE --- ")
        # --- FIN NOUVEAU LOG ---
        
        print(f"[{filename}] Analyse terminée.")
        return final_result

    def _build_contextual_grid(self, grid_rows, grid_cols, anchor_info, major_series, series_info, yolo_detections_map, offset_heuristic):
        """
        Construit la grille finale en utilisant l'ancre pour prédire/corriger les numéros.

        Args:
            grid_rows (int): Nombre de lignes dans la grille déterminée.
            grid_cols (int): Nombre de colonnes dans la grille déterminée.
            anchor_info (dict): Informations sur l'ancre trouvée (status, row, col, series, number ORIGINAL, similarity).
            major_series (str | None): La série principale identifiée pour cette image.
            series_info (dict): Infos sur les numéros existants pour chaque série (venant de _derive_series_info).
            yolo_detections_map (dict): Dictionnaire {(row, col): detection_data} des détections YOLO avec position.
            offset_heuristic (int): Offset calculé par l'heuristique (0 si non applicable).

        Returns:
            dict: Grille complète sous forme de {(row, col): FinalSlotInfo}.
        """
        print(f"[{self.image_name}] Construction de la grille contextuelle (Ancre: {anchor_info['status']}, Série Majeure: {major_series}, Offset Heuristique: {offset_heuristic})...")
        final_grid_slots = {}
        has_anchor = anchor_info['status'] == "Trouvée" and major_series is not None

        if has_anchor:
            anchor_r, anchor_c = anchor_info['row'], anchor_info['col']
            anchor_num_original = anchor_info['number'] # Utilise le numéro original non ajusté
            print(f"[{self.image_name}][Grid Build] Ancre valide trouvée en ({anchor_r},{anchor_c}), Numéro Original {anchor_num_original}, Série '{major_series}'.")
        else:
            print(f"[{self.image_name}][Grid Build] Pas d'ancre valide ou de série majeure. Construction sans prédiction/correction.")
            offset_heuristic = 0 # Assurer que l'offset est 0 s'il n'y a pas d'ancre

        for r in range(grid_rows):
            for c in range(grid_cols):
                pos = (r, c)
                slot_info = {
                    'row': r, 'col': c,
                    'predicted_number': None, 'predicted_series': None,
                    'is_plausible_in_set': None, 'is_common': None,
                    'yolo_detection_type': None, # Sera True si carte détectée, False si vide
                    'initial_id_data': None,
                    'bbox': None,
                    'status': "Inconnu" # Statut par défaut
                }

                expected_number_relative = None # Renommé pour clarté
                final_predicted_number = None # Numéro final après offset
                is_plausible = False
                # 1. Calculer le numéro attendu RELATIF à l'ancre si on a une ancre
                if has_anchor:
                    expected_number_relative = anchor_num_original + (r * grid_cols + c) - (anchor_r * grid_cols + anchor_c)
                    # Appliquer l'offset pour obtenir le numéro final prédit
                    final_predicted_number = expected_number_relative + offset_heuristic
                    
                    # Vérifier si le NUMÉRO FINAL PRÉDIT existe dans la série majeure
                    if major_series in series_info and final_predicted_number in series_info[major_series]:
                        is_plausible = True
                        slot_info['is_plausible_in_set'] = True
                        print(f"[{self.image_name}][Grid({r},{c})] Attendu Relatif={expected_number_relative}, Offset={offset_heuristic} => Numéro Final Prédit: {final_predicted_number} (Série: {major_series}). Plausible: {is_plausible}.")
                    else:
                        slot_info['is_plausible_in_set'] = False
                        print(f"[{self.image_name}][Grid({r},{c})] Attendu Relatif={expected_number_relative}, Offset={offset_heuristic} => Numéro Final Prédit: {final_predicted_number} (Série: {major_series}). Plausible: {is_plausible} (non trouvé dans series_info).")
                else:
                    print(f"[{self.image_name}][Grid({r},{c})] Pas d'ancre, impossible de calculer le numéro attendu.")

                # 2. Regarder la détection YOLO pour cette position
                yolo_detection = yolo_detections_map.get(pos)
                initial_data = None # Pour stocker les données d'ID initiale si elles existent
                
                if yolo_detection:
                    slot_info['bbox'] = yolo_detection.get('bbox')
                    slot_info['yolo_detection_type'] = yolo_detection['is_card']
                    if yolo_detection.get('initial_number') is not None:
                        initial_data = {
                            'series': yolo_detection.get('initial_series'),
                            'number': yolo_detection.get('initial_number'),
                            'similarity': yolo_detection.get('initial_similarity', 0.0)
                        }
                        slot_info['initial_id_data'] = initial_data
                    print(f"[{self.image_name}][Grid({r},{c})] Détection YOLO trouvée: is_card={yolo_detection['is_card']}, ID Initiale={initial_data}")
                else:
                    print(f"[{self.image_name}][Grid({r},{c})] Aucune détection YOLO à cette position.")

                # 3. Décider du statut et du numéro final (UTILISE MAINTENANT final_predicted_number)
                if yolo_detection:
                    if not yolo_detection['is_card']: # Slot vide détecté
                        slot_info['status'] = "Manquant (Vide Détecté)"
                        slot_info['predicted_series'] = None
                        slot_info['predicted_number'] = None
                        print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (Basé sur YOLO)")
                    else: # C'est une carte détectée par YOLO
                        if has_anchor and is_plausible:
                            slot_info['predicted_series'] = major_series # On force la série majeure
                            slot_info['predicted_number'] = final_predicted_number # On utilise le numéro final prédit
                            if initial_data and initial_data['number'] == final_predicted_number and initial_data['series'] == major_series:
                                slot_info['status'] = "Présent (Confirmé)"
                                print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (ID initiale correspond au numéro final prédit plausible)")
                            else:
                                slot_info['status'] = "Présent (Corrigé par Ancre)"
                                if initial_data:
                                     print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (ID initiale '{initial_data['series']}_{initial_data['number']}' != prédit '{major_series}_{final_predicted_number}', mais prédit plausible. Correction appliquée.)")
                                else:
                                     print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (Pas d'ID initiale, mais numéro final prédit plausible. Correction appliquée.)")
                        elif initial_data:
                            slot_info['predicted_series'] = initial_data['series']
                            slot_info['predicted_number'] = initial_data['number']
                            if initial_data['similarity'] >= HIGH_CONFIDENCE_SIMILARITY:
                                slot_info['status'] = "Présent (ID Initiale)"
                                print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (Pas d'ancre/plausibilité, ID initiale fiable)")
                            else:
                                slot_info['status'] = "Présent (ID Incertaine)"
                                print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (Pas d'ancre/plausibilité, ID initiale peu fiable)")
                        else:
                            slot_info['status'] = "Présent (ID Inconnue)"
                            slot_info['predicted_series'] = None
                            slot_info['predicted_number'] = None
                            print(f"[{self.image_name}][Grid({r},{c})] Décision: {slot_info['status']} (Pas d'ancre/plausibilité, pas d'ID initiale)")
                else: # Pas de détection YOLO
                    if has_anchor and is_plausible:
                        # Cas où on prédit un slot manquant basé sur l'ancre
                        # On calcule les infos mais on ne l'ajoute PAS à la grille finale
                        predicted_series = major_series
                        predicted_number = final_predicted_number # Utilise le numéro final prédit
                        print(f"[{self.image_name}][Grid({r},{c})] Prédit Manquant: N°{predicted_number} (Série: {major_series}). Slot ignoré (pas de détection YOLO).")
                        # slot_info['status'] = "Manquant (Prédit)" # Statut théorique
                        # slot_info['predicted_series'] = predicted_series
                        # slot_info['predicted_number'] = predicted_number 
                        # NE PAS AJOUTER : final_grid_slots[pos] = slot_info 
                        continue # Passer au slot suivant
                    else:
                        # Si pas de détection et pas de prédiction plausible, on ignore
                        print(f"[{self.image_name}][Grid({r},{c})] Décision: Ignorer le slot (Pas de YOLO, pas de prédiction plausible)")
                        continue # Ne pas ajouter ce slot à la grille finale

                # Si on arrive ici, c'est qu'on a une détection YOLO (carte ou vide) 
                # ou un cas non géré (ce qui ne devrait pas arriver)
                final_grid_slots[pos] = slot_info

        print(f"[{self.image_name}] Grille contextuelle construite avec {len(final_grid_slots)} slots (slots 'Manquant (Prédit)' exclus).")
        return final_grid_slots

    def _determine_grid_structure(self, all_detections, force_columns=True, debug_image=None):
        """
        Détermine la structure de la grille (lignes x colonnes) à partir des détections.
        Utilise le clustering pour trouver les lignes et colonnes.
        
        Args:
            all_detections (list): Liste des détections (nettoyées)
            force_columns (bool): Si True, force 5 colonnes comme standard
            debug_image (np.ndarray, optional): Image pour debug visuel
            
        Returns:
            tuple: (rows, cols, detections_with_grid_positions)
        """
        print(f"[{self.image_name}] Détermination de la structure de grille ({5 if force_columns else 'auto'} colonnes forcées) sur {len(all_detections)} détections...")
        
        # Extraire les coordonnées centrales des boîtes
        centers = []
        for i, det in enumerate(all_detections):
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y, i))  # Inclure l'indice de détection
            print(f"DEBUG Grid: Détection {i}: Centre=({center_x:.1f}, {center_y:.1f}), Taille=({x2-x1:.1f}, {y2-y1:.1f})")
        
        if len(centers) < 2:
            print(f"[{self.image_name}] Pas assez de détections pour déterminer une grille.")
            # Assigner des positions par défaut au cas où
            for det in all_detections:
                det['row'] = 0
                det['col'] = 0
            return 1, 1, all_detections
        
        # Tries les centres par coordonnées y croissantes (lignes)
        centers.sort(key=lambda c: c[1])
        
        # Fonction améliorée pour identifier les clusters dans une dimension
        def find_clusters_1d(coords, max_gap=None, pct_based_gap=0.1):
            """
            Regroupe des coordonnées 1D en clusters selon une distance maximale.
            
            Args:
                coords (list): Liste des coordonnées à regrouper
                max_gap (float, optional): Distance maximale entre deux points d'un même cluster
                pct_based_gap (float): Pourcentage de la plage totale à utiliser comme gap si max_gap=None
                
            Returns:
                list: Liste de clusters, chaque cluster contenant des tuples (valeur, index_original)
            """
            if not coords or len(coords) == 0:
                return []
                
            # Trier les coordonnées en conservant leur index original
            coords_with_indices = sorted([(value, index) for index, value in enumerate(coords)])
            sorted_values = [value for value, _ in coords_with_indices]
            
            # Si max_gap n'est pas spécifié, le calculer comme pourcentage de la plage totale
            if max_gap is None and len(sorted_values) > 1:
                total_range = sorted_values[-1] - sorted_values[0]
                max_gap = max(total_range * pct_based_gap, 5.0) if total_range > 0 else 10.0
                print(f"DEBUG Grid: Gap auto-calculé: {max_gap:.1f} (plage totale={total_range:.1f}, pct={pct_based_gap})")
            elif max_gap is None:
                max_gap = 10.0  # Valeur par défaut si un seul point
            
            # Former les clusters
            clusters = [[coords_with_indices[0]]]  # Commencer avec le premier point
            for value, index in coords_with_indices[1:]:
                prev_value = clusters[-1][-1][0]  # Valeur du dernier point du dernier cluster
                if value - prev_value > max_gap:
                    # Créer un nouveau cluster si l'écart est trop grand
                    clusters.append([])
                # Ajouter le point au dernier cluster
                clusters[-1].append((value, index))
            
            return clusters
        
        # Clustering des coordonnées Y pour identifier les lignes
        y_coords = [c[1] for c in centers]
        y_clusters = find_clusters_1d(y_coords, pct_based_gap=0.15)  # Gap plus grand pour y
        print(f"DEBUG Grid: Clusters Y (lignes) détectés: {len(y_clusters)} clusters")
        
        # Calculer les moyennes des clusters pour debug
        y_cluster_means = []
        for i, cluster in enumerate(y_clusters):
            cluster_values = [val for val, _ in cluster]
            mean_y = sum(cluster_values) / len(cluster_values) if cluster_values else 0
            min_y = min(cluster_values) if cluster_values else 0
            max_y = max(cluster_values) if cluster_values else 0
            y_cluster_means.append(mean_y)
            print(f"  Ligne {i}: {len(cluster)} détections, Y_Moyen={mean_y:.1f}, Min={min_y:.1f}, Max={max_y:.1f}")
        
        # Mapper chaque Y (via son index original) à son indice de cluster (row)
        y_coord_index_to_row = {}
        for row_index, cluster in enumerate(y_clusters):
            for y_value, original_y_index in cluster:
                y_coord_index_to_row[original_y_index] = row_index
        
        num_rows = len(y_clusters)
        
        # Assigner la ligne à chaque détection en utilisant l'index original des centres
        for center_index, (center_x, center_y, original_detection_index) in enumerate(centers):
             # Trouver l'index original de y_coord qui correspond à center_y
             # Ceci est un peu fragile si les y sont identiques, mais devrait fonctionner
             try:
                 original_y_index = y_coords.index(center_y)
                 row_assigned = y_coord_index_to_row.get(original_y_index, -1)
                 all_detections[original_detection_index]['row'] = row_assigned
             except ValueError:
                  all_detections[original_detection_index]['row'] = -1 # Not found
             except IndexError:
                  all_detections[original_detection_index]['row'] = -1 # Out of bounds

        # --- NOUVELLE LOGIQUE POUR ASSIGNER LES COLONNES ---
        num_cols = GRID_COLS_EXPECTED # Forcé à 5
        print(f"[{self.image_name}] Assignation des colonnes basée sur {num_cols} centres de colonnes attendus...")
        
        # 1. Collecter les coordonnées X des centres des détections avec une ligne valide
        valid_centers_x = []
        detection_indices_with_valid_row = []
        for i, det in enumerate(all_detections):
            if det.get('row', -1) != -1:
                # Récupérer le centre X associé à cette détection originale
                center_x = None
                for cx, cy, orig_idx in centers:
                    if orig_idx == i:
                        center_x = cx
                        break
                if center_x is not None:
                     valid_centers_x.append(center_x)
                     # Associer le centre X à l'index de la détection DANS all_detections
                     detection_indices_with_valid_row.append(i)
                else:
                     print(f"WARN: Impossible de retrouver center_x pour détection index {i}")
        
        if not valid_centers_x:
            print(f"[{self.image_name}] AVERTISSEMENT: Aucune coordonnée X valide trouvée pour déterminer les centres de colonnes. Assignation échouée.")
            # Mettre col=-1 pour toutes
            for det in all_detections:
                 det['col'] = -1
        else:
            # 2. Calculer les centres de colonnes attendus
            min_x = min(valid_centers_x)
            max_x = max(valid_centers_x)
            print(f"[{self.image_name}] Plage X des centres valides: [{min_x:.1f}, {max_x:.1f}]")
            
            expected_col_centers = []
            if num_cols == 1:
                 expected_col_centers = [(min_x + max_x) / 2] # Centre de la plage
            elif num_cols > 1 and max_x > min_x:
                 # Interpolation linéaire entre min et max
                 col_step = (max_x - min_x) / (num_cols - 1)
                 expected_col_centers = [min_x + i * col_step for i in range(num_cols)]
            else: # Si toutes les cartes sont alignées verticalement ou num_cols <= 0
                 center_x_avg = sum(valid_centers_x) / len(valid_centers_x)
                 expected_col_centers = [center_x_avg] * num_cols # Répéter la moyenne
                 print(f"[{self.image_name}] AVERTISSEMENT: Plage X nulle ou num_cols<=1. Utilisation de la moyenne X ({center_x_avg:.1f}) pour tous les centres.")
            
            print(f"[{self.image_name}] Centres de colonnes attendus (X coords): {['{:.1f}'.format(c) for c in expected_col_centers]}")
            
            # 3. Assigner chaque détection à la colonne la plus proche
            for i, detection_index in enumerate(detection_indices_with_valid_row):
                center_x = valid_centers_x[i] # Récupérer le centre X correspondant
                # Trouver l'index de la colonne la plus proche
                distances = [abs(center_x - col_center) for col_center in expected_col_centers]
                assigned_col = np.argmin(distances) # Index de la distance minimale
                
                # Assigner la colonne à la détection originale dans all_detections
                all_detections[detection_index]['col'] = assigned_col
                print(f"DEBUG Grid: Détection index {detection_index} (X={center_x:.1f}) assignée à Colonne {assigned_col} (Centre={expected_col_centers[assigned_col]:.1f}, Dist={distances[assigned_col]:.1f})")

            # Mettre col=-1 pour celles qui n'ont pas pu être assignées (pas de row valide au départ)
            for det in all_detections:
                if 'col' not in det:
                     det['col'] = -1
        
        # --- FIN NOUVELLE LOGIQUE COLONNES --- #

        print(f"Structure grille finale: {num_rows} lignes, {num_cols} colonnes.")
        
        # Visualiser la grille si debug_image est fourni
        if debug_image is not None:
            debug_img = debug_image.copy()
            height, width = debug_img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Tracer les positions de grille
            for det in all_detections:
                 # Vérifier si row et col existent et sont valides
                 if det.get('row', -1) != -1 and det.get('col', -1) != -1:
                     x1, y1, x2, y2 = map(int, det['bbox'])
                     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     label = f"R{det['row']}C{det['col']}"
                     cv2.putText(debug_img, label, (x1, y1-5), font, 1, (0, 255, 0), 2)
                 else:
                     # Marquer les détections non placées
                     x1, y1, x2, y2 = map(int, det['bbox'])
                     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1) # Rouge fine
                     cv2.putText(debug_img, "NoPos", (x1, y1-5), font, 0.8, (0, 0, 255), 1)
                     
            # Enregistrer l'image debug
            cv2.imwrite('grid_debug.jpg', debug_img)
            print("Image grid_debug.jpg générée.")
            
        # Retourner les détections avec les clés 'row' et 'col' ajoutées/mises à jour
        return num_rows, num_cols, all_detections

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