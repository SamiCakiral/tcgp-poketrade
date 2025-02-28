import sqlite3
import os
import tempfile
from google.cloud import storage
import shutil

# Nom du bucket défini dans app.yaml
BUCKET_NAME = os.environ.get('CLOUD_STORAGE_BUCKET')
DB_NAME = 'pokemon_cards.db'

def download_db_from_cloud():
    """Télécharge la base de données depuis Cloud Storage"""
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
    """Télécharge la base de données vers Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(DB_NAME)
    blob.upload_from_filename(local_db_path)

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
    
    # Enregistrer un hook pour uploader la BD après commit
    original_commit = conn.commit
    def commit_and_upload():
        original_commit()
        upload_db_to_cloud(local_db_path)
    conn.commit = commit_and_upload
    
    return conn 