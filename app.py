from flask import Flask, request, jsonify, render_template
import joblib
import re
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer

# Load  model

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove links
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = emoji.demojize(text)  # Convert emojis to text
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    else:
        text = ''
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cleaned_text = clean_text(data['text'])
    
    
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    label = 'hate' if prediction[0] == 'hate' else 'not hate'
    
    return jsonify({'label': label})

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
