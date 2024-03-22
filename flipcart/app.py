from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickled model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define preprocess_text function
def preprocess_text(text):
    # Implement your text preprocessing logic here
    # Example: Tokenization, lowercasing, removing punctuation, etc.
    processed_text = text.lower()  # Convert text to lowercase
    # Other preprocessing steps...
    return processed_text

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input_text = request.form['review']
    processed_input_text = preprocess_text(user_input_text)
    predicted_sentiment = model.predict([processed_input_text])
    predicted_sentiment_label = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    return render_template('result.html', sentiment=predicted_sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
