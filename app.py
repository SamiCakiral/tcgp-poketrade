from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from database import init_db, get_db_connection, upload_db_to_cloud
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = "y'en a pas"
# Initialisation de la base de données
init_db()

# Décorateur pour vérifier si l'utilisateur est connecté
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.')
            return redirect(url_for('login'))
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
            
            # NOUVEAU: Vérifier si l'utilisateur a des correspondances de cartes
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
        'A2a': 'Ile Fabuleuse',
        'A2b': 'Choc Spatio-Temporelle',
        'A2c': 'Lumière Triomphale'
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
    
    # Récupérer l'utilisateur courant
    user_id = session.get('user_id')
    
    # Connexion à la base de données
    conn = sqlite3.connect('pokemon_cards.db')
    conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
    cursor = conn.cursor()
    
    # 1. Récupérer les cartes recherchées par l'utilisateur
    cursor.execute("""
        SELECT c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity
        FROM cards c
        JOIN wanted_cards w ON c.set_id = w.set_id AND c.card_number = w.card_number
        WHERE w.user_id = ?
    """, (user_id,))
    wanted_cards = [dict(row) for row in cursor.fetchall()]
    
    # 2. Récupérer les cartes en double des autres utilisateurs
    cursor.execute("""
        SELECT e.user_id as owner_id, c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity, u.name as owner_name
        FROM extra_cards e
        JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
        JOIN users u ON e.user_id = u.id
        WHERE e.user_id != ? AND e.quantity > 1
    """, (user_id,))
    other_users_extras = [dict(row) for row in cursor.fetchall()]
    
    # 3. Récupérer les cartes en double de l'utilisateur courant
    cursor.execute("""
        SELECT c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity
        FROM extra_cards e
        JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
        WHERE e.user_id = ? AND e.quantity > 1
    """, (user_id,))
    user_extras = [dict(row) for row in cursor.fetchall()]
    
    # 4. Récupérer les cartes recherchées par les autres utilisateurs
    cursor.execute("""
        SELECT w.user_id as owner_id, c.id, c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, c.rarity, u.name as owner_name
        FROM wanted_cards w
        JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
        JOIN users u ON w.user_id = u.id
        WHERE w.user_id != ?
    """, (user_id,))
    other_users_wanted = [dict(row) for row in cursor.fetchall()]
    
    # Fermer la connexion à la base de données
    conn.close()
    
    # Trouver les échanges potentiels
    potential_trades = []
    
    # Pour chaque carte recherchée par l'utilisateur
    for wanted_card in wanted_cards:
        # Trouver les utilisateurs qui ont cette carte en double
        for extra_card in other_users_extras:
            if extra_card['id'] == wanted_card['id']:
                # Pour chaque carte en double de l'utilisateur
                for user_extra in user_extras:
                    # Vérifier si l'autre utilisateur recherche cette carte
                    for other_wanted in other_users_wanted:
                        if other_wanted['id'] == user_extra['id'] and other_wanted['owner_id'] == extra_card['owner_id']:
                            # Calculer la qualité du match
                            match_quality = calculate_match_quality(wanted_card, user_extra, extra_card, other_wanted)
                            
                            # Créer l'échange potentiel
                            trade = {
                                'your_card': user_extra,
                                'their_card': extra_card,
                                'other_user': {
                                    'id': extra_card['owner_id'],
                                    'username': extra_card['owner_name']
                                },
                                'match_quality': match_quality
                            }
                            
                            potential_trades.append(trade)
    
    # Appliquer les filtres
    filtered_trades = filter_trades(potential_trades, filter_type)
    
    # Appliquer le tri
    sorted_trades = sort_trades(filtered_trades, sort_type)
    
    # Renvoyer les résultats
    return jsonify({
        'success': True,
        'trades': sorted_trades
    })

def calculate_match_quality(wanted_card, user_extra, extra_card, other_wanted):
    # Vérifier si les cartes sont de la même série
    same_set = wanted_card['set_id'] == user_extra['set_id']
    
    # Vérifier si les cartes sont de même rareté
    same_rarity = wanted_card['rarity'] == user_extra['rarity']
    
    # Calculer la qualité du match
    if same_set and same_rarity:
        return 'Excellent'
    elif same_rarity:
        return 'Good'
    else:
        return 'Fair'

def filter_trades(trades, filter_type):
    if filter_type == 'all':
        return trades
    
    filtered = []
    for trade in trades:
        if filter_type == 'same-set' and trade['your_card']['set_id'] == trade['their_card']['set_id']:
            filtered.append(trade)
        elif filter_type == 'same-rarity' and trade['your_card']['rarity'] == trade['their_card']['rarity']:
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

if __name__ == '__main__':
    app.run(debug=True) 