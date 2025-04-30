from flask import Flask, jsonify, render_template, request, redirect, url_for, session
import tensorflow as tf
from models.realTimeGesture import detect_gesture
from models.realTimeAlphabet import detect_alphabet
import threading
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Secret key for session management (keep this safe)
app.secret_key = 'this_is_my_secret_project'

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'sl_db'
}

app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load models
gesture_model = tf.keras.models.load_model("models/gesture_model.keras")
alphabet_model = tf.keras.models.load_model("models/alphabet_model.keras")

# Shared variable for recognition results
recognition_result = {"alphabet": "", "status": "ready"}

def login_required(func):
    """Decorator to protect routes requiring login."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('signin'))
        return func(*args, **kwargs)
    return wrapper

@app.route('/')
@login_required
def index():
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = None
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            query = "SELECT * FROM users WHERE email = %s"
            cursor.execute(query, (email,))
            user = cursor.fetchone()

            if not user:
                return render_template('signin.html', error="User with this email does not exist. Please sign up.")

            if check_password_hash(user['password'], password):
                # Set session variables
                session['user_id'] = user['id']
                session['user_name'] = user['full_name']
                return redirect(url_for('index'))
            else:
                return render_template('signin.html', error="Incorrect password.")

        except mysql.connector.Error as err:
            return render_template('signin.html', error=f"Database Error: {err}")

        finally:
            if conn is not None and conn.is_connected():
                cursor.close()
                conn.close()

    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']

        conn = None
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            hashed_password = generate_password_hash(password)

            query = "INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)"
            cursor.execute(query, (full_name, email, hashed_password))
            conn.commit()

            # Log user in immediately after signup
            session['user_id'] = cursor.lastrowid  # last inserted id
            session['user_name'] = full_name

            return redirect(url_for('index'))

        except mysql.connector.Error as err:
            return render_template('signup.html', error=f"Database error: {err}")

        finally:
            if conn is not None and conn.is_connected():
                cursor.close()
                conn.close()

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html')

@app.route('/acquire')
@login_required
def acquire():
    return render_template('acquire.html')

@app.route('/alphabet')
@login_required
def alphabet_page():
    return render_template('alphabet.html')

@app.route('/gesture')
@login_required
def gesture_page():
    return render_template('gesture.html')

@app.route('/start_alphabet_recognition', methods=['POST'])
@login_required
def start_alphabet_recognition():
    if recognition_result["status"] != "processing":
        recognition_result["status"] = "processing"
        recognition_result["alphabet"] = ""

        def recognition_thread():
            recognized_word = detect_alphabet(alphabet_model)
            recognition_result["alphabet"] = recognized_word
            recognition_result["status"] = "ready"

        thread = threading.Thread(target=recognition_thread)
        thread.start()
        return jsonify({'status': 'recognition started'})

    return jsonify({'status': 'already processing'})

@app.route('/get_alphabet_result', methods=['GET'])
@login_required
def get_alphabet_result():
    return jsonify({
        'recognized_alphabet': recognition_result["alphabet"],
        'status': recognition_result["status"]
    })

@app.route('/start_gesture_recognition', methods=['POST'])
@login_required
def start_gesture_recognition():
    gesture = detect_gesture(gesture_model)
    return jsonify({'recognized_gesture': gesture})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, jsonify, render_template
# import tensorflow as tf
# from models.realTimeGesture import detect_gesture
# from models.realTimeAlphabet import detect_alphabet  # Import alphabet recognition

# app = Flask(__name__)
# app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto-reload templates in debug mode

# # Load models
# gesture_model = tf.keras.models.load_model("models/gesture_model.keras")
# alphabet_model = tf.keras.models.load_model("models/alphabet_model.keras")  # Load alphabet model

# # Routes for rendering HTML templates
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/tutorial')
# def tutorial():
#     return render_template('tutorial.html')

# @app.route('/signin')
# def signin():
#     return render_template('signin.html')

# @app.route('/signup')
# def signup():
#     return render_template('signup.html')

# @app.route('/learn')
# def learn():
#     return render_template('learn.html')

# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('404.html'), 404

# @app.route('/acquire')
# def acquire():
#     return render_template('acquire.html')

# # New Routes for Displaying Alphabet & Gesture Pages
# @app.route('/alphabet')
# def alphabet_page():
#     return render_template('alphabet.html')

# @app.route('/gesture')
# def gesture_page():
#     return render_template('gesture.html')

# # API Endpoint for Alphabet Recognition
# @app.route('/start_alphabet_recognition', methods=['POST'])
# def start_alphabet_recognition():
#     alphabet = detect_alphabet(alphabet_model)  # Call alphabet recognition function
#     return jsonify({'recognized_alphabet': alphabet})

# # API Endpoint for Gesture Recognition
# @app.route('/start_gesture_recognition', methods=['POST'])
# def start_gesture_recognition():
#     gesture = detect_gesture(gesture_model)  # Call gesture recognition function
#     return jsonify({'recognized_gesture': gesture})

# if __name__ == '__main__':
#     app.run(debug=True)
