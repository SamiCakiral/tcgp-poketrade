import sqlite3
import os
import tempfile
from google.cloud import storage
import shutil
import functools

# Nom du bucket défini dans app.yaml
BUCKET_NAME = os.environ.get('CLOUD_STORAGE_BUCKET')
DB_NAME = 'pokemon_cards.db'
# Mode de développement local (sans Cloud Storage)
IS_LOCAL_DEV = os.environ.get('LOCAL_DEV', 'False').lower() == 'true'

def download_db_from_cloud():
    """Télécharge la base de données depuis Cloud Storage"""
    if IS_LOCAL_DEV:
        # En mode développement local, utiliser le fichier local
        local_db_path = os.path.join(os.getcwd(), DB_NAME)
        
        # Vérifier si le fichier existe
        if not os.path.exists(local_db_path):
            # Créer une base de données vide
            conn = sqlite3.connect(local_db_path)
            conn.close()
        
        return local_db_path
    
    # Mode cloud normal
    local_db_path = os.path.join(tempfile.gettempdir(), DB_NAME)
    
    # Vérifier si le fichier existe déjà en local
    if os.path.exists(local_db_path):
        return local_db_path
        
    # Créer le client Cloud Storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DB_NAME)
    
    # Télécharger s'il existe dans le bucket
    if blob.exists():
        blob.download_to_filename(local_db_path)
    
    return local_db_path

def upload_db_to_cloud(local_db_path):
    """Uploade la base de données vers Cloud Storage"""
    if IS_LOCAL_DEV:
        # En mode développement local, ne rien faire
        return
        
    # Mode cloud normal
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DB_NAME)
    blob.upload_from_filename(local_db_path)

def auto_upload_db(conn, local_db_path):
    """Fonction utilitaire pour uploader la DB après les modifications"""
    conn.commit()
    upload_db_to_cloud(local_db_path)

def init_db():
    """Initialise la base de données"""
    local_db_path = download_db_from_cloud()
    
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()
    
    # Table des cartes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        set_id TEXT,
        card_number INTEGER,
        card_name TEXT,
        french_name TEXT,
        image_url TEXT,
        UNIQUE(set_id, card_number)
    )
    ''')
    
    # Table des utilisateurs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        birthdate TEXT,
        UNIQUE(name, birthdate)
    )
    ''')
    
    # Table des cartes en trop
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS extra_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        set_id TEXT,
        card_number INTEGER,
        quantity INTEGER DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, set_id, card_number)
    )
    ''')
    
    # Table des cartes recherchées
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS wanted_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        set_id TEXT,
        card_number INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, set_id, card_number)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    # Uploader la BD vers Cloud Storage
    upload_db_to_cloud(local_db_path)

def get_db_connection():
    """Récupère une connexion à la BD"""
    local_db_path = download_db_from_cloud()
    conn = sqlite3.connect(local_db_path)
    conn.row_factory = sqlite3.Row
    
    # Au lieu de modifier conn.commit, nous retournons aussi le chemin
    # pour que les fonctions qui utilisent la connexion puissent faire l'upload
    return conn, local_db_path 