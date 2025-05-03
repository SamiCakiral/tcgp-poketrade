from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from database.databaseManager import init_db, get_db_connection, upload_db_to_cloud
import os
from functools import wraps
import cv2
import numpy as np
from collections import defaultdict, Counter
import io # Pour lire les fichiers en mémoire
import concurrent.futures
import time # Pour simuler/mesurer le temps si besoin

# --- Import de l'analyseur ---
try:
    # L'import semble correct, l'erreur précédente pouvait venir d'ailleurs
    from card_detector.pokedex_analyzer import PokedexScreenshotAnalyzer
except ImportError as e:
    print(f"ERREUR: Impossible d'importer PokedexScreenshotAnalyzer depuis card_detector: {e} e.stacktrace: {str(e.__traceback__)}")
    print("Vérifiez la structure du projet et les __init__.py.")
    # Classe Factice pour éviter un crash complet
    class PokedexScreenshotAnalyzer:
        def __init__(self, *args, **kwargs): print("Analyseur factice initialisé - CORRIGEZ L'IMPORT")
        def analyze_image(self, *args, **kwargs): return {"error": "Analyseur non chargé"}

app = Flask(__name__)
# Il est préférable de définir une clé secrète plus robuste via les variables d'environnement
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "une_cle_secrete_par_defaut_vraiment_peu_sure_a_changer")

# --- Initialisation de l'analyseur IA ---
try:
    print("Initialisation de l'analyseur de cartes Pokémon (chargement différé)...")
    # CORRECTION 2: Mettre preload_models=False pour un démarrage plus rapide
    card_analyzer = PokedexScreenshotAnalyzer(preload_models=False)
    print("Analyseur configuré (modèles seront chargés au premier usage).")
except Exception as e:
    print(f"ERREUR CRITIQUE lors de la configuration de l'analyseur: {e}")
    card_analyzer = PokedexScreenshotAnalyzer(preload_models=False) # Initialiser la version factice/non chargée


# Initialisation de la base de données SQLite
# S'assurer que le chemin vers la DB est correct si databaseManager.py est dans un sous-dossier
init_db()

# --- Fonction pour convertir les types NumPy en types Python ---
def convert_numpy_types(obj):
    """Convertit récursivement les types NumPy en types Python natifs pour la sérialisation JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convertit les arrays NumPy en listes Python
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    # Gérer SimpleCounter si nécessaire (il devrait déjà être sérialisable s'il ne contient que des clés/valeurs sérialisables)
    # elif isinstance(obj, Counter): # Ou SimpleCounter si vous l'avez défini
    #     return {str(key): convert_numpy_types(value) for key, value in obj.items()} # Convertit clés en str pour JSON
    else:
        return obj

# Décorateur pour vérifier si l'utilisateur est connecté
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.', 'warning')
            # Si c'est une requête API, retourner une erreur JSON
            if request.endpoint and (request.endpoint.startswith('api_') or 'api/' in request.path):
                 return jsonify(success=False, message="Authentification requise"), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        birthdate = request.form['birthdate']
        
        conn, db_path = get_db_connection()
        
        try:
            conn.execute('INSERT INTO users (name, birthdate) VALUES (?, ?)',
                         (name, birthdate))
            conn.commit()
            upload_db_to_cloud(db_path)  # Upload après le commit
            flash('Inscription réussie! Vous pouvez maintenant vous connecter.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Un utilisateur avec ce nom et cette date de naissance existe déjà!')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        birthdate = request.form['birthdate']
        
        conn, db_path = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE name = ? AND birthdate = ?',
                           (name, birthdate)).fetchone()
        
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            
            # Vérifier les correspondances d'échange
            trade_matches = conn.execute('''
                SELECT COUNT(*) as match_count
                FROM wanted_cards w
                JOIN extra_cards e ON w.set_id = e.set_id AND w.card_number = e.card_number
                WHERE w.user_id = ? AND e.user_id != ?
            ''', (user['id'], user['id'])).fetchone()
            
            if trade_matches and trade_matches['match_count'] > 0:
                session['show_trade_notification'] = True
                session['trade_matches_count'] = trade_matches['match_count']
            
            conn.close()
            return redirect(url_for('dashboard'))
        else:
            conn.close()
            flash('Nom ou date de naissance incorrects!')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn, db_path = get_db_connection()
    
    # Récupérer toutes les cartes d'une série
    active_set = request.args.get('set', 'A1')
    
    # Récupérer toutes les cartes de la série active pour l'affichage Collection
    all_cards = conn.execute('''
        SELECT id, set_id, card_number, card_name, french_name, image_url, rarity
        FROM cards 
        WHERE set_id = ?
        ORDER BY card_number
    ''', (active_set,)).fetchall()
    
    # Récupérer les cartes en trop de l'utilisateur
    extra_cards = conn.execute('''
        SELECT c.set_id, c.card_number, e.quantity
        FROM extra_cards e
        JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
        WHERE e.user_id = ?
    ''', (session['user_id'],)).fetchall()
    
    # Récupérer les cartes recherchées de l'utilisateur
    wanted_cards = conn.execute('''
        SELECT c.set_id, c.card_number
        FROM wanted_cards w
        JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
        WHERE w.user_id = ?
    ''', (session['user_id'],)).fetchall()
    
    # NOUVEAU: Récupérer toutes les séries disponibles avec leurs cartes
    all_sets = conn.execute('SELECT DISTINCT set_id FROM cards ORDER BY set_id').fetchall()
    all_sets_with_cards = []
    
    # Pour chaque série, récupérer toutes les cartes
    for set_row in all_sets:
        set_id = set_row['set_id']
        
        # Récupérer le nom de la série
        set_name = {
            'A1': 'Puissance Génétique',
            'A1a': 'Île Fabuleuse',
            'A2': 'Choc Spatio-Temporelle',
            'A2a': 'Lumière Triomphale',
            'A2b': 'Réjouissances Rayonnantes'
        }.get(set_id, set_id)
        
        # Récupérer toutes les cartes de cette série
        set_cards = conn.execute('''
            SELECT id, set_id, card_number, card_name, french_name, image_url, rarity
            FROM cards 
            WHERE set_id = ?
            ORDER BY card_number
        ''', (set_id,)).fetchall()
        
        all_sets_with_cards.append({
            'set_id': set_id,
            'name': set_name,
            'cards': set_cards
        })
    
    # NOUVEAU: Vérifier les cartes recherchées qui sont disponibles chez d'autres utilisateurs
    available_wanted_cards = conn.execute('''
        SELECT w.set_id, w.card_number, GROUP_CONCAT(u.name) as owners
        FROM wanted_cards w
        JOIN extra_cards e ON w.set_id = e.set_id AND w.card_number = e.card_number
        JOIN users u ON e.user_id = u.id
        WHERE w.user_id = ? AND e.user_id != ?
        GROUP BY w.set_id, w.card_number
    ''', (session['user_id'], session['user_id'])).fetchall()
    
    # Convertir en dictionnaires pour faciliter la vérification
    extra_dict = {}
    for card in extra_cards:
        card_key = f"{card['set_id']}-{card['card_number']}"
        extra_dict[card_key] = card['quantity']
    
    wanted_dict = {f"{card['set_id']}-{card['card_number']}": True for card in wanted_cards}
    available_dict = {f"{card['set_id']}-{card['card_number']}": card['owners'] for card in available_wanted_cards}
    
    # Vérifier si des échanges sont possibles pour la notification
    has_trade_matches = len(available_wanted_cards) > 0
    has_wanted_cards = len(wanted_cards) > 0
    
    # Récupérer toutes les séries disponibles pour le sélecteur
    available_sets = all_sets
    
    # Compter le nombre total de cartes dans la série active
    total_cards_in_set = len(all_cards)
    
    # Compter combien de cartes l'utilisateur possède dans cette série
    owned_cards_count = len([c for c in all_cards if f"{c['set_id']}-{c['card_number']}" in extra_dict])
    
    conn.close()
    
    return render_template('dashboard.html', 
                          all_cards=all_cards,
                          extra_dict=extra_dict,
                          wanted_dict=wanted_dict,
                          available_dict=available_dict,
                          active_set=active_set,
                          available_sets=available_sets,
                          all_sets_with_cards=all_sets_with_cards,
                          total_cards_in_set=total_cards_in_set,
                          owned_cards_count=owned_cards_count,
                          has_trade_matches=has_trade_matches,
                          has_wanted_cards=has_wanted_cards)

@app.route('/add_card', methods=['GET', 'POST'])
@login_required
def add_card():
    if request.method == 'POST':
        set_id = request.form['set_id']
        card_number = request.form['card_number']
        card_type = request.form['card_type']  # 'extra' ou 'wanted'
        
        conn, db_path = get_db_connection()
        
        # Vérifier si la carte existe
        card = conn.execute('SELECT * FROM cards WHERE set_id = ? AND card_number = ?',
                          (set_id, card_number)).fetchone()
        
        if not card:
            flash('Cette carte n\'existe pas dans la base de données!')
        else:
            try:
                if card_type == 'extra':
                    # Vérifier si l'utilisateur a déjà cette carte en trop
                    existing = conn.execute('''
                        SELECT * FROM extra_cards 
                        WHERE user_id = ? AND set_id = ? AND card_number = ?
                    ''', (session['user_id'], set_id, card_number)).fetchone()
                    
                    if existing:
                        conn.execute('''
                            UPDATE extra_cards 
                            SET quantity = quantity + 1 
                            WHERE user_id = ? AND set_id = ? AND card_number = ?
                        ''', (session['user_id'], set_id, card_number))
                    else:
                        conn.execute('''
                            INSERT INTO extra_cards (user_id, set_id, card_number, quantity)
                            VALUES (?, ?, ?, 1)
                        ''', (session['user_id'], set_id, card_number))
                    
                    flash('Carte ajoutée à votre collection de cartes en trop!')
                
                elif card_type == 'wanted':
                    conn.execute('''
                        INSERT INTO wanted_cards (user_id, set_id, card_number)
                        VALUES (?, ?, ?)
                    ''', (session['user_id'], set_id, card_number))
                    
                    flash('Carte ajoutée à votre liste de recherche!')
                
                conn.commit()
            except sqlite3.IntegrityError:
                if card_type == 'wanted':
                    flash('Cette carte est déjà dans votre liste de recherche!')
            
        conn.close()
        return redirect(url_for('dashboard'))
    
    return render_template('add_card.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        set_id = request.form['set_id']
        card_number = request.form['card_number']
        
        conn, db_path = get_db_connection()
        
        # Chercher les utilisateurs qui ont cette carte en trop
        users_with_extra = conn.execute('''
            SELECT u.name, e.quantity
            FROM extra_cards e
            JOIN users u ON e.user_id = u.id
            WHERE e.set_id = ? AND e.card_number = ?
        ''', (set_id, card_number)).fetchall()
        
        # Chercher les utilisateurs qui recherchent cette carte
        users_wanting = conn.execute('''
            SELECT u.name
            FROM wanted_cards w
            JOIN users u ON w.user_id = u.id
            WHERE w.set_id = ? AND w.card_number = ?
        ''', (set_id, card_number)).fetchall()
        
        # Obtenir les infos de la carte
        card = conn.execute('''
            SELECT * FROM cards
            WHERE set_id = ? AND card_number = ?
        ''', (set_id, card_number)).fetchone()
        
        conn.close()
        
        return render_template('search_results.html',
                              card=card,
                              users_with_extra=users_with_extra,
                              users_wanting=users_wanting)
    
    return render_template('search.html')

@app.route('/card_info')
def card_info():
    set_id = request.args.get('set_id')
    card_number = request.args.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'exists': False})
    
    conn, db_path = get_db_connection()
    card = conn.execute('''
        SELECT * FROM cards
        WHERE set_id = ? AND card_number = ?
    ''', (set_id, card_number)).fetchone()
    conn.close()
    
    if card:
        return jsonify({
            'exists': True,
            'id': card['id'],
            'set_id': card['set_id'],
            'card_number': card['card_number'],
            'card_name': card['card_name'],
            'french_name': card['french_name'],
            'image_url': card['image_url'],
            'rarity': card['rarity']
        })
    else:
        return jsonify({'exists': False})

@app.route('/delete_account', methods=['GET', 'POST'])
@login_required
def delete_account():
    if request.method == 'POST':
        user_id = session['user_id']
        conn, db_path = get_db_connection()
        
        try:
            # Supprimer d'abord les cartes en trop et recherchées de l'utilisateur
            conn.execute('DELETE FROM extra_cards WHERE user_id = ?', (user_id,))
            conn.execute('DELETE FROM wanted_cards WHERE user_id = ?', (user_id,))
            
            # Ensuite supprimer l'utilisateur lui-même
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            
            # Commit et upload vers Cloud Storage
            conn.commit()
            upload_db_to_cloud(db_path)
            
            # Déconnecter l'utilisateur
            session.pop('user_id', None)
            session.pop('user_name', None)
            
            flash('Votre compte a été supprimé avec succès.')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Erreur lors de la suppression du compte: {str(e)}')
        finally:
            conn.close()
    
    # GET request - afficher la page de confirmation
    return render_template('delete_account.html')

@app.route('/get_set_cards')
def get_set_cards():
    set_id = request.args.get('set_id')
    if not set_id:
        return jsonify([])
    
    conn, db_path = get_db_connection()
    cards = conn.execute('SELECT * FROM cards WHERE set_id = ? ORDER BY card_number', (set_id,)).fetchall()
    conn.close()
    
    result = []
    for card in cards:
        result.append({
            'id': card['id'],
            'set_id': card['set_id'],
            'card_number': card['card_number'],
            'card_name': card['card_name'],
            'french_name': card['french_name'],
            'image_url': card['image_url'],
            'rarity': card['rarity']
        })
    
    return jsonify(result)

@app.route('/add_multiple_cards', methods=['POST'])
def add_multiple_cards():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
    data = request.json
    cards = data.get('cards', [])
    
    if not cards:
        return jsonify({'success': False, 'message': 'Aucune carte sélectionnée'})
    
    conn, db_path = get_db_connection()
    added_count = 0
    
    try:
        for card in cards:
            set_id = card['set_id']
            card_number = card['card_number']
            card_type = card['type']
            # Nouvelle fonctionnalité: gérer la quantité
            quantity = card.get('quantity', 1)
            
            # Vérifier si la carte existe
            existing_card = conn.execute('SELECT * FROM cards WHERE set_id = ? AND card_number = ?',
                          (set_id, card_number)).fetchone()
            
            if not existing_card:
                continue
                
            if card_type == 'extra':
                # Vérifier si l'utilisateur a déjà cette carte en trop
                existing = conn.execute('''
                    SELECT * FROM extra_cards 
                    WHERE user_id = ? AND set_id = ? AND card_number = ?
                ''', (session['user_id'], set_id, card_number)).fetchone()
                
                if existing:
                    # Si elle existe déjà, mettre à jour la quantité
                    conn.execute('''
                        UPDATE extra_cards 
                        SET quantity = quantity + ? 
                        WHERE user_id = ? AND set_id = ? AND card_number = ?
                    ''', (quantity, session['user_id'], set_id, card_number))
                else:
                    # Si elle n'existe pas, l'ajouter avec la quantité spécifiée
                    conn.execute('''
                        INSERT INTO extra_cards (user_id, set_id, card_number, quantity)
                        VALUES (?, ?, ?, ?)
                    ''', (session['user_id'], set_id, card_number, quantity))
                
            elif card_type == 'wanted':
                try:
                    conn.execute('''
                        INSERT INTO wanted_cards (user_id, set_id, card_number)
                        VALUES (?, ?, ?)
                    ''', (session['user_id'], set_id, card_number))
                except sqlite3.IntegrityError:
                    # Ignorer si la carte est déjà dans la liste de recherche
                    pass
                    
            added_count += 1
            
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True, 'count': added_count})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/increment_card', methods=['POST'])
@login_required
def increment_card():
    data = request.json
    set_id = data.get('set_id')
    card_number = data.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'success': False, 'message': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        conn.execute('''
            UPDATE extra_cards 
            SET quantity = quantity + 1 
            WHERE user_id = ? AND set_id = ? AND card_number = ?
        ''', (session['user_id'], set_id, card_number))
        
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/remove_card', methods=['POST'])
@login_required
def remove_card():
    data = request.json
    set_id = data.get('set_id')
    card_number = data.get('card_number')
    is_wanted = data.get('is_wanted', False)
    
    if not set_id or not card_number:
        return jsonify({'success': False, 'message': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        if is_wanted:
            conn.execute('''
                DELETE FROM wanted_cards 
                WHERE user_id = ? AND set_id = ? AND card_number = ?
            ''', (session['user_id'], set_id, card_number))
        else:
            conn.execute('''
                DELETE FROM extra_cards 
                WHERE user_id = ? AND set_id = ? AND card_number = ?
            ''', (session['user_id'], set_id, card_number))
        
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/add_to_collection', methods=['POST'])
@login_required
def add_to_collection():
    data = request.json
    set_id = data.get('set_id')
    card_number = data.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'success': False, 'message': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        # Vérifier si la carte existe déjà dans la collection
        existing = conn.execute('''
            SELECT * FROM extra_cards 
            WHERE user_id = ? AND set_id = ? AND card_number = ?
        ''', (session['user_id'], set_id, card_number)).fetchone()
        
        if existing:
            # Si elle existe déjà, incrémenter la quantité
            conn.execute('''
                UPDATE extra_cards 
                SET quantity = quantity + 1 
                WHERE user_id = ? AND set_id = ? AND card_number = ?
            ''', (session['user_id'], set_id, card_number))
        else:
            # Sinon, l'ajouter
            conn.execute('''
                INSERT INTO extra_cards (user_id, set_id, card_number, quantity)
                VALUES (?, ?, ?, 1)
            ''', (session['user_id'], set_id, card_number))
        
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/add_to_wanted', methods=['POST'])
@login_required
def add_to_wanted():
    data = request.json
    set_id = data.get('set_id')
    card_number = data.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'success': False, 'message': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        # Ajouter la carte aux recherches
        conn.execute('''
            INSERT OR IGNORE INTO wanted_cards (user_id, set_id, card_number)
            VALUES (?, ?, ?)
        ''', (session['user_id'], set_id, card_number))
        
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/get_all_sets')
@login_required
def get_all_sets():
    """Récupère toutes les séries disponibles"""
    conn, db_path = get_db_connection()
    sets = conn.execute('SELECT DISTINCT set_id FROM cards ORDER BY set_id').fetchall()
    conn.close()
    
    # Associer un nom à chaque série (à adapter selon vos données)
    set_names = {
        'A1': 'Puissance Génétique',
        'A1a': 'Île Fabuleuse',
        'A2': 'Choc Spatio-Temporelle',
        'A2a': 'Lumière Triomphale',
        'A2b': 'Réjouissances Rayonnantes'
    }
    
    result = []
    for set_row in sets:
        set_id = set_row['set_id']
        result.append({
            'set_id': set_id,
            'name': set_names.get(set_id, '')
        })
    
    return jsonify(result)

@app.route('/hide_trade_notification', methods=['POST'])
@login_required
def hide_trade_notification():
    session.pop('show_trade_notification', None)
    
    return jsonify({'success': True})

@app.route('/convert_to_collection', methods=['POST'])
@login_required
def convert_to_collection():
    data = request.json
    set_id = data.get('set_id')
    card_number = data.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'success': False, 'message': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        # Supprimer de la liste des cartes recherchées
        conn.execute('''
            DELETE FROM wanted_cards 
            WHERE user_id = ? AND set_id = ? AND card_number = ?
        ''', (session['user_id'], set_id, card_number))
        
        # Vérifier si la carte existe déjà dans la collection
        existing = conn.execute('''
            SELECT * FROM extra_cards 
            WHERE user_id = ? AND set_id = ? AND card_number = ?
        ''', (session['user_id'], set_id, card_number)).fetchone()
        
        if existing:
            # Si la carte existe déjà, incrémenter sa quantité
            conn.execute('''
                UPDATE extra_cards 
                SET quantity = quantity + 1 
                WHERE user_id = ? AND set_id = ? AND card_number = ?
            ''', (session['user_id'], set_id, card_number))
        else:
            # Sinon, l'ajouter comme nouvelle carte
            conn.execute('''
                INSERT INTO extra_cards (user_id, set_id, card_number, quantity)
                VALUES (?, ?, ?, 1)
            ''', (session['user_id'], set_id, card_number))
        
        conn.commit()
        upload_db_to_cloud(db_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/get_card_owners')
@login_required
def get_card_owners():
    set_id = request.args.get('set_id')
    card_number = request.args.get('card_number')
    
    if not set_id or not card_number:
        return jsonify({'owners': [], 'debug': 'Paramètres manquants'})
    
    conn, db_path = get_db_connection()
    
    try:
        # Récupérer les propriétaires qui ont cette carte en trop
        owners = conn.execute('''
            SELECT u.name 
            FROM extra_cards e
            JOIN users u ON e.user_id = u.id
            WHERE e.set_id = ? AND e.card_number = ? AND e.user_id != ?
        ''', (set_id, card_number, session['user_id'])).fetchall()
        
        # Convertir le résultat en liste simple et ajouter du débogage
        owners_list = [owner['name'] for owner in owners]
        
        print(f"Propriétaires trouvés pour {set_id}-{card_number}: {owners_list}")
        
        return jsonify({
            'owners': owners_list, 
            'debug': {
                'count': len(owners_list),
                'card': f"{set_id}-{card_number}",
                'user_id': session['user_id']
            }
        })
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({'owners': [], 'error': str(e)})
    finally:
        conn.close()

@app.route('/trade_assistant')
@login_required
def trade_assistant():
    return render_template('trade_assistant.html')

@app.route('/api/potential_trades')
@login_required
def get_potential_trades():
    # Récupérer les paramètres de filtre et de tri
    filter_type = request.args.get('filter', 'all')
    sort_type = request.args.get('sort', 'relevance')
    # NOUVEAU : Récupérer l'ID de l'utilisateur cible pour filtrer
    target_user_id = request.args.get('user_id', 'all')

    # Récupérer l'utilisateur courant
    user_id = session.get('user_id')

    # MODIFIÉ: Utiliser la fonction get_db_connection
    conn, _ = get_db_connection() # _ car on n'a pas besoin du db_path ici
    # conn.row_factory = sqlite3.Row est déjà défini dans get_db_connection (supposé)
    # Pas besoin de créer un curseur séparé

    try:
        # 1. Récupérer les cartes recherchées par l'utilisateur
        # MODIFIÉ: Utiliser conn.execute et ne pas convertir en dict immédiatement
        wanted_cards_rows = conn.execute("""
            SELECT c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity
            FROM cards c
            JOIN wanted_cards w ON c.set_id = w.set_id AND c.card_number = w.card_number
            WHERE w.user_id = ?
        """, (user_id,)).fetchall()
        wanted_cards = wanted_cards_rows # Les objets Row se comportent déjà comme des dicts

        # 2. Récupérer les cartes en double des autres utilisateurs
        # MODIFIÉ: Filtrer par target_user_id si nécessaire
        other_extras_query = """
            SELECT e.user_id as owner_id, c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity, u.name as owner_name
            FROM extra_cards e
            JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
            JOIN users u ON e.user_id = u.id
            WHERE e.user_id != ? AND e.quantity > 1
        """
        params = [user_id]
        if target_user_id != 'all':
            other_extras_query += " AND e.user_id = ?"
            params.append(target_user_id)

        other_users_extras_rows = conn.execute(other_extras_query, params).fetchall()
        other_users_extras = other_users_extras_rows

        # 3. Récupérer les cartes en double de l'utilisateur courant
        # MODIFIÉ: Utiliser conn.execute et ne pas convertir en dict immédiatement
        user_extras_rows = conn.execute("""
            SELECT c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity
            FROM extra_cards e
            JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
            WHERE e.user_id = ? AND e.quantity > 1
        """, (user_id,)).fetchall()
        user_extras = user_extras_rows

        # 4. Récupérer les cartes recherchées par les autres utilisateurs
        # MODIFIÉ: Filtrer par target_user_id si nécessaire
        other_wanted_query = """
            SELECT w.user_id as owner_id, c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity, u.name as owner_name
            FROM wanted_cards w
            JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
            JOIN users u ON w.user_id = u.id
            WHERE w.user_id != ?
        """
        params_wanted = [user_id]
        if target_user_id != 'all':
            other_wanted_query += " AND w.user_id = ?"
            params_wanted.append(target_user_id)

        other_users_wanted_rows = conn.execute(other_wanted_query, params_wanted).fetchall()
        other_users_wanted = other_users_wanted_rows

    finally:
        # MODIFIÉ: Fermer la connexion obtenue via get_db_connection
        conn.close()

    # Trouver les échanges potentiels
    potential_trades = []

    # Pour chaque carte recherchée par l'utilisateur (Row object)
    for wanted_card in wanted_cards:
        # Trouver les utilisateurs qui ont cette carte en double (Row object)
        for extra_card in other_users_extras:
            # Comparaison via ['id'] car ce sont des objets Row
            # (Déjà filtré par target_user_id si nécessaire au niveau de la requête SQL)
            if extra_card['id'] == wanted_card['id']:
                # Pour chaque carte en double de l'utilisateur (Row object)
                for user_extra in user_extras:
                    # Vérifier si l'autre utilisateur recherche cette carte (Row object)
                    for other_wanted in other_users_wanted:
                         # Comparaison via ['id'] et ['owner_id'] car ce sont des objets Row
                         # (Déjà filtré par target_user_id si nécessaire au niveau de la requête SQL)
                        if other_wanted['id'] == user_extra['id'] and other_wanted['owner_id'] == extra_card['owner_id']:
                            # --- DEBUGGING START ---
                            print(f"--- Potential Match Found Before Rarity Check ---")
                            print(f"My Extra: {user_extra['set_id']}-{user_extra['card_number']} (Rarity: '{user_extra['rarity']}')")
                            print(f"Their Extra: {extra_card['set_id']}-{extra_card['card_number']} (Rarity: '{extra_card['rarity']}')")
                            print(f"Owner: {extra_card['owner_name']}")
                            print(f"Checking Rarity: '{user_extra['rarity']}' == '{extra_card['rarity']}' -> {user_extra['rarity'] == extra_card['rarity']}")
                            # --- DEBUGGING END ---

                            # ---> ICI : Vérification de la rareté <---
                            # Est-ce que la rareté de MA carte (user_extra) est la même que celle de SA carte (extra_card) ?
                            if user_extra['rarity'] == extra_card['rarity']:
                                # Si oui, calculer la qualité et ajouter l'échange
                                match_quality = calculate_match_quality(user_extra, extra_card)

                                # Créer l'échange potentiel en accédant aux colonnes via ['...']
                                # Convertir explicitement en dict pour le JSON final
                                trade = {
                                    'your_card': dict(user_extra), # Convertir en dict
                                    'their_card': dict(extra_card), # Convertir en dict
                                    'other_user': {
                                        'id': extra_card['owner_id'],
                                        'username': extra_card['owner_name']
                                    },
                                    'match_quality': match_quality
                                }

                                potential_trades.append(trade)

    # --- DEBUGGING: Check list before filtering ---
    print(f"--- Final potential_trades list BEFORE filtering (Count: {len(potential_trades)}) ---")
    # print(potential_trades) # Uncomment carefully if needed, might be long

    # Appliquer les filtres
    filtered_trades = filter_trades(potential_trades, filter_type)

    # --- DEBUGGING: Check list after filtering ---
    print(f"--- Trade list AFTER filtering (Count: {len(filtered_trades)}) ---")
    # print(filtered_trades)

    # Appliquer le tri
    sorted_trades = sort_trades(filtered_trades, sort_type)

    # --- DEBUGGING: Check list after sorting ---
    print(f"--- Trade list AFTER sorting (Count: {len(sorted_trades)}) ---")
    # print(sorted_trades)

    # Renvoyer les résultats
    return jsonify({
        'success': True,
        'trades': sorted_trades
    })

def calculate_match_quality(user_extra_card, their_extra_card):
    # Vérifier si les cartes sont de la même série
    # Note: On compare la carte de l'utilisateur (user_extra_card)
    # avec la carte de l'autre (their_extra_card)
    same_set = user_extra_card['set_id'] == their_extra_card['set_id']

    # La rareté est déjà garantie d'être la même par la logique précédente
    # Calculer la qualité du match
    if same_set:
        return 'Excellent' # Même série et même rareté (implicite)
    else:
        return 'Good' # Même rareté (implicite) mais séries différentes

def filter_trades(trades, filter_type):
    if filter_type == 'all':
        return trades

    filtered = []
    for trade in trades:
        if filter_type == 'same-set' and trade['your_card']['set_id'] == trade['their_card']['set_id']:
            filtered.append(trade)
        elif filter_type == 'best-match' and trade['match_quality'] == 'Excellent':
            filtered.append(trade)

    return filtered

def sort_trades(trades, sort_type):
    # Fonction pour convertir une rareté en valeur numérique pour le tri
    def rarity_value(rarity):
        if not rarity:
            return 0
        if rarity == 'Common':
            return 1
        if rarity.startswith('Diamond'):
            return 2 + int(rarity.split(' ')[1])
        if rarity.startswith('Star'):
            return 6 + int(rarity.split(' ')[1])
        if rarity == 'Crown Rare':
            return 10
        return 0
    
    if sort_type == 'rarity-desc':
        return sorted(trades, key=lambda t: rarity_value(t['their_card']['rarity']), reverse=True)
    elif sort_type == 'rarity-asc':
        return sorted(trades, key=lambda t: rarity_value(t['their_card']['rarity']))
    elif sort_type == 'set':
        return sorted(trades, key=lambda t: t['their_card']['set_id'])
    else:  # sort_type == 'relevance' (par défaut)
        # Trier par qualité de match (Excellent > Good > Fair)
        quality_order = {'Excellent': 3, 'Good': 2, 'Fair': 1}
        return sorted(trades, key=lambda t: quality_order.get(t['match_quality'], 0), reverse=True)

@app.route('/api/propose_trade', methods=['POST'])
@login_required
def propose_trade():
    data = request.json
    
    # Récupérer les données nécessaires
    your_card_id = data.get('your_card_id')
    their_card_id = data.get('their_card_id')
    other_user_id = data.get('user_id')
    user_id = session.get('user_id')
    
    # Vérifier que toutes les données nécessaires sont présentes
    if not your_card_id or not their_card_id or not other_user_id:
        return jsonify({'success': False, 'message': 'Données manquantes'})
    
    # Connexion à la base de données
    conn = sqlite3.connect('pokemon_cards.db')
    cursor = conn.cursor()
    
    try:
        # Créer une table pour les propositions d'échange si elle n'existe pas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id INTEGER,
                receiver_id INTEGER,
                sender_card_id INTEGER,
                receiver_card_id INTEGER,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sender_id) REFERENCES users (id),
                FOREIGN KEY (receiver_id) REFERENCES users (id),
                FOREIGN KEY (sender_card_id) REFERENCES cards (id),
                FOREIGN KEY (receiver_card_id) REFERENCES cards (id)
            )
        ''')
        
        # Insérer la proposition d'échange
        cursor.execute('''
            INSERT INTO trade_proposals 
            (sender_id, receiver_id, sender_card_id, receiver_card_id) 
            VALUES (?, ?, ?, ?)
        ''', (user_id, other_user_id, your_card_id, their_card_id))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Proposition d\'échange envoyée avec succès!'
        })
    
    except Exception as e:
        conn.rollback()
        return jsonify({
            'success': False,
            'message': f'Erreur lors de la création de la proposition d\'échange: {str(e)}'
        })
    
    finally:
        conn.close()

@app.route('/api/user_cards_status')
@login_required
def get_user_cards_status():
    set_id = request.args.get('set_id', None)
    
    conn, db_path = get_db_connection()
    
    # Récupérer les cartes en trop de l'utilisateur
    if set_id:
        extra_cards = conn.execute('''
            SELECT c.set_id, c.card_number, e.quantity
            FROM extra_cards e
            JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
            WHERE e.user_id = ? AND c.set_id = ?
        ''', (session['user_id'], set_id)).fetchall()
        
        # Récupérer les cartes recherchées de l'utilisateur
        wanted_cards = conn.execute('''
            SELECT c.set_id, c.card_number
            FROM wanted_cards w
            JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
            WHERE w.user_id = ? AND c.set_id = ?
        ''', (session['user_id'], set_id)).fetchall()
    else:
        extra_cards = conn.execute('''
            SELECT c.set_id, c.card_number, e.quantity
            FROM extra_cards e
            JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
            WHERE e.user_id = ?
        ''', (session['user_id'],)).fetchall()
        
        wanted_cards = conn.execute('''
            SELECT c.set_id, c.card_number
            FROM wanted_cards w
            JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
            WHERE w.user_id = ?
        ''', (session['user_id'],)).fetchall()
    
    # Convertir en dictionnaires pour faciliter la vérification
    extra_dict = {}
    for card in extra_cards:
        card_key = f"{card['set_id']}-{card['card_number']}"
        extra_dict[card_key] = card['quantity']
    
    wanted_dict = {f"{card['set_id']}-{card['card_number']}": True for card in wanted_cards}
    
    conn.close()
    
    return jsonify({
        'extra_dict': extra_dict,
        'wanted_dict': wanted_dict
    })

# --- Fonction Helper pour traiter une seule image ---
def process_single_image(file_storage, analyzer):
    """Traite une seule image et retourne les résultats ou une erreur."""
    filename = file_storage.filename
    try:
        start_time = time.time()
        # Lire l'image en mémoire depuis le FileStorage uploadé
        in_memory_file = io.BytesIO()
        file_storage.save(in_memory_file)
        in_memory_file.seek(0)
        file_bytes = np.frombuffer(in_memory_file.read(), np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_np is None:
            return {"filename": filename, "error": f"Impossible de décoder l'image."}

        # Analyser l'image
        # print(f"Début analyse pour: {filename}")
        results = analyzer.analyze_image(image_np) # Utilise le seuil par défaut
        end_time = time.time()
        # print(f"Fin analyse pour: {filename} ({end_time - start_time:.2f}s)")

        # Ajouter le nom du fichier aux résultats pour référence si nécessaire
        results["filename"] = filename
        return results

    except Exception as e:
        # import traceback
        # print(f"Erreur détaillée pour {filename}: {traceback.format_exc()}")
        return {"filename": filename, "error": f"Erreur interne lors du traitement: {str(e)}"}

# --- NOUVELLE ROUTE pour analyse individuelle ---
@app.route('/api/analyze_pokedex_single', methods=['POST'])
@login_required
def api_analyze_pokedex_single():
    if 'screenshot' not in request.files: # Attend un seul fichier nommé 'screenshot'
        return jsonify(success=False, message="Aucun fichier 'screenshot' trouvé."), 400

    file = request.files['screenshot']
    if not file or file.filename == '':
        return jsonify(success=False, message="Aucun fichier sélectionné."), 400

    result_raw = process_single_image(file, card_analyzer)

    # *** AJOUT : Conversion des types NumPy avant jsonify ***
    try:
        result_serializable = convert_numpy_types(result_raw)
    except Exception as conversion_error:
        # Si la conversion elle-même échoue, renvoyer une erreur
        print(f"Erreur lors de la conversion des types pour JSON: {conversion_error}")
        # Renvoyer l'erreur originale si possible, ou une erreur générique
        error_detail = result_raw.get("error", f"Erreur de conversion JSON: {conversion_error}")
        return jsonify(success=False, result={"filename": result_raw.get("filename", file.filename), "error": error_detail}), 500

    # Déterminer le succès basé sur la présence d'une clé 'error' DANS LE RÉSULTAT ORIGINAL
    is_success = not result_raw.get("error")

    # Renvoyer le résultat converti et sérialisable
    return jsonify(success=is_success, result=result_serializable)

@app.route('/pokedex_analyzer')
@login_required
def pokedex_analyzer_view():
    """Affiche la page de l'assistant d'analyse Pokédex IA."""
    return render_template('pokedex_analyzer.html')

@app.route('/api/update_collection_from_scan', methods=['POST'])
@login_required
def api_update_collection_from_scan():
    data = request.json
    cards_to_update = data.get('cards_to_update', [])
    cards_to_mark_as_wanted = data.get('cards_to_mark_as_wanted', []) # NOUVEAU : Récupérer la liste
    user_id = session['user_id']

    if not cards_to_update and not cards_to_mark_as_wanted: # Vérifier les deux listes
        return jsonify(success=False, message="Aucune carte à traiter fournie.")

    conn, db_path = get_db_connection()
    updated_count = 0
    wanted_added_count = 0 # NOUVEAU : Compteur pour les cartes recherchées
    update_errors = []

    try:
        # --- Traitement des mises à jour de quantité (existant) ---
        for card_info in cards_to_update:
            set_id = card_info.get('set_id')
            card_number_str = str(card_info.get('card_number', ''))
            quantity = card_info.get('quantity')

            if not set_id or not card_number_str or quantity is None or quantity < 0:
                update_errors.append(f"Données invalides (update): {card_info}")
                continue

            try: card_number_db = int(card_number_str)
            except ValueError: card_number_db = card_number_str

            card_exists = conn.execute('SELECT 1 FROM cards WHERE set_id = ? AND card_number = ?',
                                     (set_id, card_number_db)).fetchone()
            if not card_exists:
                update_errors.append(f"Carte {set_id}-{card_number_str} non trouvée.")
                continue

            if quantity > 0:
                 result = conn.execute('''
                    INSERT INTO extra_cards (user_id, set_id, card_number, quantity)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, set_id, card_number) DO UPDATE SET
                    quantity = excluded.quantity
                 ''', (user_id, set_id, card_number_db, quantity))
                 # On ne peut pas facilement savoir si c'était un insert ou un update ici sans requête supplémentaire
                 # On compte juste l'opération
                 updated_count += 1
            else: # quantity == 0 signifie suppression
                result = conn.execute('''
                    DELETE FROM extra_cards
                    WHERE user_id = ? AND set_id = ? AND card_number = ?
                ''', (user_id, set_id, card_number_db))
                if result.rowcount > 0:
                    updated_count += 1

        # --- NOUVEAU : Traitement des cartes à marquer comme recherchées ---
        for card_info in cards_to_mark_as_wanted:
            set_id = card_info.get('set_id')
            card_number_str = str(card_info.get('card_number', ''))

            if not set_id or not card_number_str:
                 update_errors.append(f"Données invalides (wanted): {card_info}")
                 continue

            try: card_number_db = int(card_number_str)
            except ValueError: card_number_db = card_number_str

            # Vérifier si la carte maîtresse existe (redondant si déjà fait avant mais plus sûr)
            card_exists = conn.execute('SELECT 1 FROM cards WHERE set_id = ? AND card_number = ?',
                                     (set_id, card_number_db)).fetchone()
            if not card_exists:
                update_errors.append(f"Carte {set_id}-{card_number_str} non trouvée (pour wanted).")
                continue

            # Ajouter à la liste des recherches, ignorer si déjà présente
            result = conn.execute('''
                INSERT OR IGNORE INTO wanted_cards (user_id, set_id, card_number)
                VALUES (?, ?, ?)
            ''', (user_id, set_id, card_number_db))

            if result.rowcount > 0: # Compter seulement si une ligne a été insérée
                wanted_added_count += 1

        # --- Fin des traitements ---

        conn.commit()
        if updated_count > 0 or wanted_added_count > 0: # Upload si changement dans extra OU wanted
             upload_db_to_cloud(db_path)

        # Construire le message de succès final
        success_messages = []
        if updated_count > 0:
            success_messages.append(f"{updated_count} carte(s) de la collection mise(s) à jour.")
        if wanted_added_count > 0:
            success_messages.append(f"{wanted_added_count} carte(s) ajoutée(s) aux recherches.")
        if not success_messages:
             success_messages.append("Aucune modification effectuée.")

        return jsonify(success=True,
                       message=" ".join(success_messages),
                       updated_count=updated_count,
                       wanted_added_count=wanted_added_count, # Renvoyer le compte au front
                       errors=update_errors)

    except Exception as e:
        conn.rollback()
        import traceback
        print(f"Erreur lors de la mise à jour de la collection/wanted: {e}\n{traceback.format_exc()}")
        return jsonify(success=False, message=f"Erreur serveur: {str(e)}", errors=update_errors), 500
    finally:
        conn.close()

# NOUVELLE ROUTE pour lister les partenaires d'échange potentiels
@app.route('/api/trade_partners')
@login_required
def get_trade_partners():
    user_id = session['user_id']
    conn, _ = get_db_connection()
    partners = []
    try:
        # Trouver les utilisateurs qui ont des cartes que JE VEUX
        partners_having_my_wants = conn.execute("""
            SELECT DISTINCT u.id, u.name
            FROM users u
            JOIN extra_cards e ON u.id = e.user_id
            JOIN wanted_cards w ON e.set_id = w.set_id AND e.card_number = w.card_number
            WHERE w.user_id = ? AND e.user_id != ?
        """, (user_id, user_id)).fetchall()

        # Trouver les utilisateurs qui veulent des cartes que J'AI EN DOUBLE
        partners_wanting_my_extras = conn.execute("""
            SELECT DISTINCT u.id, u.name
            FROM users u
            JOIN wanted_cards w ON u.id = w.user_id
            JOIN extra_cards e ON w.set_id = e.set_id AND w.card_number = e.card_number
            WHERE e.user_id = ? AND w.user_id != ? AND e.quantity > 1
        """, (user_id, user_id)).fetchall()

        # Combiner et dédupliquer les partenaires
        partner_ids = set()
        for row in partners_having_my_wants:
            if row['id'] not in partner_ids:
                partners.append({'id': row['id'], 'name': row['name']})
                partner_ids.add(row['id'])
        for row in partners_wanting_my_extras:
             if row['id'] not in partner_ids:
                partners.append({'id': row['id'], 'name': row['name']})
                partner_ids.add(row['id'])

        # Trier par nom
        partners.sort(key=lambda x: x['name'])

    finally:
        conn.close()

    return jsonify({'users': partners})

if __name__ == '__main__':
    # Assure-toi d'avoir init_db() appelé quelque part avant si nécessaire
    # init_db()

    # Modifie cette ligne : ajoute ou change use_reloader=False
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080) 
    # J'ai gardé debug=True, mais tu peux aussi le mettre à False si besoin.
    # J'ai aussi gardé host et port s'ils étaient là. L'important est use_reloader=False. 