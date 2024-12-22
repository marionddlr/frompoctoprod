from flask import Flask, request, jsonify, render_template_string
import logging
import os
from run import TextPredictionModel

app = Flask(__name__)

logger = logging.getLogger(__name__)

MODEL_PATH = "./train/data/artefacts/2024-12-11-12-12-24"
model = TextPredictionModel.from_artefacts(MODEL_PATH)

@app.route('/')
def home():
    """
    Route for the home page.
    """
    return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Text Prediction API</title>
          </head>
          <body>
            <div class="container">
              <h1>Welcome to the Text Prediction API</h1>
              <form id="predict-form">
                <div class="form-group">
                  <label for="text">Enter text:</label>
                  <input type="text" id="text" name="text" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-primary">Predict</button>
              </form>
              <div id="prediction-result"></div>
            </div>
            <script>
              document.getElementById('predict-form').addEventListener('submit', function(event) {
                event.preventDefault();
                const text = document.getElementById('text').value;
                fetch('/predict', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({ text: text })
                })
                .then(response => response.json())
                .then(data => {
                  document.getElementById('prediction-result').innerHTML = '<h2>Predictions:</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                })
                .catch(error => {
                  console.error('Error:', error);
                });
              });
            </script>
          </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route to predict tags for given text.
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "No text provided for prediction"}), 400
        
        texts = data['text']
        if isinstance(texts, str):
            texts = [texts]
        
        top_k = data.get('top_k', 5)
        top_k_predictions = model.predict(texts, top_k)
        
        return jsonify(top_k_predictions)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)