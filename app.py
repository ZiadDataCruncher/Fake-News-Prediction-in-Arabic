from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


model = pickle.load(open('modele.pkl', 'rb'))
vectorizer = pickle.load(open('vecteur.pkl', 'rb'))


def preprocess_text(text):
    stop_words = set(stopwords.words('arabic'))
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        cleaned_news = preprocess_text(news)
        vectorized_news = vectorizer.transform([cleaned_news])
        prediction = model.predict(vectorized_news)
        result = "Fausses nouvelles 📰" if prediction[0] == 1 else "Nouvelles véridiques ✅"
        return render_template('prediction.html', prediction_text=result)
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
