import numpy as np
import cv2
import os
import pickle
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import json
import matplotlib.pyplot as plt

class PokemonCardIdentifier:
    def __init__(self, model_type='resnet50', use_pca=True, pca_components=512):
        """
        Initialise le système d'identification de cartes
        
        Args:
            model_type: 'resnet50' ou 'mobilenetv2'
            use_pca: Réduire la dimension des embeddings avec PCA
            pca_components: Nombre de composantes à conserver après PCA
        """
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.model_type = model_type
        
        # Charger le modèle de base (sans les couches de classification)
        if model_type == 'resnet50':
            self.base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = preprocess_resnet
            self.embedding_size = 2048
        elif model_type == 'mobilenetv2':
            self.base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = preprocess_mobilenet
            self.embedding_size = 1280
        
        # Structures pour stocker la base de données d'embeddings
        self.embeddings = []
        self.card_labels = []
        self.card_images = []  # Optionnel: stocker des miniatures des cartes
        self.pca = None
        self.knn = None
        
    def compute_embedding(self, image):
        """
        Calcule l'embedding d'une image de carte
        
        Args:
            image: Image de la carte (BGR format de OpenCV)
            
        Returns:
            Vecteur d'embedding
        """
        # Redimensionner l'image à la taille d'entrée du modèle
        target_size = (224, 224)
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            image = cv2.resize(image, target_size)
        
        # Convertir BGR à RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prétraiter l'image pour le modèle
        img_array = np.expand_dims(image_rgb, axis=0)
        img_preprocessed = self.preprocess(img_array)
        
        # Calculer l'embedding
        features = self.base_model.predict(img_preprocessed, verbose=0)
        
        return features[0]
    
    def add_card(self, image, label, store_thumbnail=True):
        """
        Ajoute une carte à la base de données
        
        Args:
            image: Image de la carte
            label: Nom ou identifiant de la carte
            store_thumbnail: Stocker une miniature de l'image
        """
        # Calculer l'embedding
        embedding = self.compute_embedding(image)
        
        # Ajouter à la base de données
        self.embeddings.append(embedding)
        self.card_labels.append(label)
        
        # Stocker une miniature
        if store_thumbnail:
            thumbnail = cv2.resize(image, (112, 156))  # Taille standard carte Pokémon réduite
            self.card_images.append(thumbnail)
        
        # Marquer le modèle KNN comme nécessitant une reconstruction
        self.knn = None
    
    def build_index(self):
        """
        Construit l'index KNN pour la recherche rapide
        """
        if not self.embeddings:
            print("La base de données est vide")
            return
        
        # Convertir en tableau numpy
        X = np.array(self.embeddings)
        
        # Appliquer PCA si demandé
        if self.use_pca and len(self.embeddings) > self.pca_components:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components)
                X_reduced = self.pca.fit_transform(X)
            else:
                X_reduced = self.pca.transform(X)
        else:
            X_reduced = X
        
        # Construire l'index KNN
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
        self.knn.fit(X_reduced)
        
        print(f"Index construit avec {len(self.embeddings)} cartes")
    
    def identify_card(self, image, threshold=0.8):
        """
        Identifie une carte
        
        Args:
            image: Image de la carte à identifier
            threshold: Seuil de confiance (0-1)
            
        Returns:
            label: Étiquette de la carte la plus similaire
            confidence: Score de confiance (0-100%)
            matches: Liste des meilleures correspondances avec leurs scores
        """
        if not self.embeddings or self.knn is None:
            self.build_index()
            
        if not self.embeddings:
            return "Aucune carte dans la base de données", 0, []
        
        # Calculer l'embedding
        embedding = self.compute_embedding(image)
        
        # Appliquer PCA si utilisé
        if self.use_pca and self.pca is not None:
            embedding = self.pca.transform(embedding.reshape(1, -1))[0]
        
        # Trouver les k plus proches voisins
        distances, indices = self.knn.kneighbors([embedding])
        
        # Convertir distance cosinus en similitude (0-1)
        similarities = 1 - distances[0]
        
        # Préparer les résultats
        matches = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
            matches.append({
                'label': self.card_labels[idx],
                'confidence': float(sim * 100),  # Pourcentage
                'index': int(idx)
            })
        
        # Vérifier si le meilleur match dépasse le seuil
        if similarities[0] < threshold:
            return "Carte inconnue", float(similarities[0] * 100), matches
        
        return self.card_labels[indices[0][0]], float(similarities[0] * 100), matches
    
    def save_database(self, filename):
        """
        Sauvegarde la base de données d'embeddings
        
        Args:
            filename: Nom du fichier de sauvegarde
        """
        data = {
            'model_type': self.model_type,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'embeddings': self.embeddings,
            'card_labels': self.card_labels,
            'pca': self.pca
        }
        
        # Sauvegarder les données
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        # Sauvegarder les miniatures séparément si elles existent
        if self.card_images:
            thumbnails_file = os.path.splitext(filename)[0] + '_thumbnails.pkl'
            with open(thumbnails_file, 'wb') as f:
                pickle.dump(self.card_images, f)
        
        print(f"Base de données sauvegardée dans {filename}")
    
    def load_database(self, filename):
        """
        Charge la base de données d'embeddings
        
        Args:
            filename: Nom du fichier de sauvegarde
        """
        # Charger les données principales
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.model_type = data['model_type']
        self.use_pca = data['use_pca']
        self.pca_components = data['pca_components']
        self.embeddings = data['embeddings']
        self.card_labels = data['card_labels']
        self.pca = data['pca']
        
        # Charger les miniatures si elles existent
        thumbnails_file = os.path.splitext(filename)[0] + '_thumbnails.pkl'
        if os.path.exists(thumbnails_file):
            with open(thumbnails_file, 'rb') as f:
                self.card_images = pickle.load(f)
        
        # Reconstruire l'index
        self.build_index()
        
        print(f"Base de données chargée: {len(self.embeddings)} cartes")
    
    def build_database_from_directory(self, directory, extension='.jpg'):
        """
        Construit la base de données à partir d'un répertoire d'images
        
        Args:
            directory: Chemin vers le répertoire contenant les images de cartes
            extension: Extension des fichiers d'images
        """
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    # Utiliser le nom du fichier comme étiquette
                    label = os.path.splitext(file)[0]
                    
                    # Charger l'image
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        # Ajouter à la base de données
                        self.add_card(image, label)
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"Traitement de {count} cartes...")
        
        # Construire l'index
        self.build_index()
        
        print(f"Base de données construite avec {count} cartes")
    
    def visualize_similar_cards(self, query_image, num_results=5):
        """
        Visualise les cartes les plus similaires à une carte requête
        
        Args:
            query_image: Image de la carte requête
            num_results: Nombre de résultats à afficher
        """
        # Identifier la carte
        _, _, matches = self.identify_card(query_image)
        
        # Limiter le nombre de résultats
        matches = matches[:min(num_results, len(matches))]
        
        # Créer une figure
        fig, axs = plt.subplots(1, len(matches) + 1, figsize=(3 * (len(matches) + 1), 4))
        
        # Afficher l'image requête
        axs[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Requête")
        axs[0].axis('off')
        
        # Afficher les résultats
        for i, match in enumerate(matches):
            idx = match['index']
            if idx < len(self.card_images):
                axs[i + 1].imshow(cv2.cvtColor(self.card_images[idx], cv2.COLOR_BGR2RGB))
                axs[i + 1].set_title(f"{match['label']}\n{match['confidence']:.1f}%")
                axs[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()