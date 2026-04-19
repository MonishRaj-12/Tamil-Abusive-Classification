from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store prediction history for dashboard
prediction_history = []

def load_model():
    global model, tokenizer
    model_path = './models/muril_tamil_abuse'
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True
    else:
        print("Model not found. Please run train_model.py first!")
        return False

def predict_comment(text):
    """Predict if a comment is abusive or not"""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
    
    return {
        'label': 'Abusive' if prediction == 1 else 'Non-abusive',
        'abusive_score': probabilities[0][1].item(),
        'non_abusive_score': probabilities[0][0].item(),
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data.get('comment', '')
    
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    
    # Get prediction
    result = predict_comment(comment)
    
    # Store in history (keep last 100)
    prediction_history.append({
        'text': comment,
        'result': result,
        'timestamp': result['timestamp']
    })
    if len(prediction_history) > 100:
        prediction_history.pop(0)
    
    return jsonify(result)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(prediction_history)

@app.route('/stats', methods=['GET'])
def get_stats():
    if not prediction_history:
        return jsonify({'total': 0, 'abusive_count': 0, 'non_abusive_count': 0, 'avg_confidence': 0})
    
    total = len(prediction_history)
    abusive_count = sum(1 for p in prediction_history if p['result']['label'] == 'Abusive')
    avg_confidence = sum(p['result']['confidence'] for p in prediction_history) / total
    
    return jsonify({
        'total': total,
        'abusive_count': abusive_count,
        'non_abusive_count': total - abusive_count,
        'avg_confidence': round(avg_confidence, 3),
        'abusive_percentage': round((abusive_count / total) * 100, 1)
    })

if __name__ == '__main__':
    if load_model():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Please train the model first by running: python train_model.py")