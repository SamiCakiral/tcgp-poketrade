from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
from database import init_db, get_db_connection
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
        
        conn = get_db_connection()
        
        try:
            conn.execute('INSERT INTO users (name, birthdate) VALUES (?, ?)',
                         (name, birthdate))
            conn.commit()
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
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE name = ? AND birthdate = ?',
                           (name, birthdate)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))
        else:
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
    
    conn = get_db_connection()
    extra_cards = conn.execute('''
        SELECT c.set_id, c.card_number, c.card_name, c.french_name, c.image_url, e.quantity
        FROM extra_cards e
        JOIN cards c ON e.set_id = c.set_id AND e.card_number = c.card_number
        WHERE e.user_id = ?
    ''', (session['user_id'],)).fetchall()
    
    wanted_cards = conn.execute('''
        SELECT c.set_id, c.card_number, c.card_name, c.french_name, c.image_url
        FROM wanted_cards w
        JOIN cards c ON w.set_id = c.set_id AND w.card_number = c.card_number
        WHERE w.user_id = ?
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                          extra_cards=extra_cards, 
                          wanted_cards=wanted_cards)

@app.route('/add_card', methods=['GET', 'POST'])
def add_card():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        set_id = request.form['set_id']
        card_number = request.form['card_number']
        card_type = request.form['card_type']  # 'extra' ou 'wanted'
        
        conn = get_db_connection()
        
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
        
        conn = get_db_connection()
        
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
    
    conn = get_db_connection()
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

if __name__ == '__main__':
    app.run(debug=True) 