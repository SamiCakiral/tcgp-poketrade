from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from database import init_db, get_db_connection, upload_db_to_cloud
import os

app = Flask(__name__)
app.secret_key = "y'en a pas"
# Initialisation de la base de données
init_db()

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
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn, db_path = get_db_connection()
    
    # Récupérer toutes les cartes d'une série
    active_set = request.args.get('set', 'A1')
    
    # Récupérer toutes les cartes de la série active pour l'affichage Collection
    all_cards = conn.execute('''
        SELECT id, set_id, card_number, card_name, french_name, image_url
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
            'A2a': 'Lumière Triomphale'
        }.get(set_id, set_id)
        
        # Récupérer toutes les cartes de cette série
        set_cards = conn.execute('''
            SELECT id, set_id, card_number, card_name, french_name, image_url
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
def add_card():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
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
            'image_url': card['image_url']
        })
    else:
        return jsonify({'exists': False})

@app.route('/delete_account', methods=['GET', 'POST'])
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
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
            'image_url': card['image_url']
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
def increment_card():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
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
def remove_card():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
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
def add_to_collection():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
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
def add_to_wanted():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
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
def get_all_sets():
    """Récupère toutes les séries disponibles"""
    if 'user_id' not in session:
        return jsonify([])
    
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
def hide_trade_notification():
    if 'user_id' not in session:
        return jsonify({'success': False})
    
    session.pop('show_trade_notification', None)
    
    return jsonify({'success': True})

@app.route('/convert_to_collection', methods=['POST'])
def convert_to_collection():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Utilisateur non connecté'})
    
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
def get_card_owners():
    if 'user_id' not in session:
        return jsonify({'owners': [], 'debug': 'Non connecté'})
    
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

if __name__ == '__main__':
    app.run(debug=True) 