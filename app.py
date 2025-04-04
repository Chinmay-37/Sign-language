from flask import Flask, jsonify, render_template
import tensorflow as tf
from models.realTimeGesture import detect_gesture
from models.realTimeAlphabet import detect_alphabet  # Import alphabet recognition

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto-reload templates in debug mode

# Load models
gesture_model = tf.keras.models.load_model("models/gesture_model.keras")
alphabet_model = tf.keras.models.load_model("models/alphabet_model.keras")  # Load alphabet model

# Routes for rendering HTML templates
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/acquire')
def acquire():
    return render_template('acquire.html')

# New Routes for Displaying Alphabet & Gesture Pages
@app.route('/alphabet')
def alphabet_page():
    return render_template('alphabet.html')

@app.route('/gesture')
def gesture_page():
    return render_template('gesture.html')

# API Endpoint for Alphabet Recognition
@app.route('/start_alphabet_recognition', methods=['POST'])
def start_alphabet_recognition():
    alphabet = detect_alphabet(alphabet_model)  # Call alphabet recognition function
    return jsonify({'recognized_alphabet': alphabet})

# API Endpoint for Gesture Recognition
@app.route('/start_gesture_recognition', methods=['POST'])
def start_gesture_recognition():
    gesture = detect_gesture(gesture_model)  # Call gesture recognition function
    return jsonify({'recognized_gesture': gesture})

if __name__ == '__main__':
    app.run(debug=True)
