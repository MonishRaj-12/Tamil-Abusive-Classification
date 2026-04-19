import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset
import os
import warnings
warnings.filterwarnings('ignore')

# Custom Dataset class
class TamilDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# Import preprocessing
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import load_and_prepare_data

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Load and prepare data
print("\n📂 Loading datasets...")
train_texts, val_texts, test_texts, train_labels, val_labels = load_and_prepare_data(
    'data/trainV2.csv',
    'data/TestV2 - testV2.csv'
)

# Check if we have both classes
unique_train = np.unique(train_labels)
if len(unique_train) < 2:
    print("\n❌ ERROR: Training data has only ONE class!")
    print("Cannot train model. Please check your dataset.")
    exit(1)

# Load MuRIL tokenizer and model
print("\n🔧 Loading MuRIL model...")
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create datasets
print("\n📝 Creating datasets...")
train_dataset = TamilDataset(train_texts, train_labels, tokenizer)
val_dataset = TamilDataset(val_texts, val_labels, tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments optimized for your dataset
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increased epochs for better learning
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=50,
    fp16=False,
    warmup_ratio=0.1,
    save_total_limit=2,
    report_to='none',  # Disable external logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("\n🎯 Starting training...")
print("="*50)
trainer.train()

# Evaluate
print("\n📊 Evaluating model...")
eval_results = trainer.evaluate()
print("="*50)
print(f"✅ Evaluation Results:")
print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"   F1-Score: {eval_results['eval_f1']:.4f}")
print(f"   Precision: {eval_results['eval_precision']:.4f}")
print(f"   Recall: {eval_results['eval_recall']:.4f}")
print("="*50)

# Save model
model_save_path = './models/muril_tamil_abuse'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n💾 Model saved to {model_save_path}")

# Test on unseen data
print("\n🧪 Testing on unseen data...")
test_dataset = TamilDataset(test_texts, None, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

model.eval()
predictions = []
all_probs = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Save test predictions
test_results = pd.DataFrame({
    'text': test_texts,
    'prediction': predictions,
    'prediction_label': ['Abusive' if p == 1 else 'Non-abusive' for p in predictions],
    'confidence': [max(prob) for prob in all_probs]
})
os.makedirs('./results', exist_ok=True)
test_results.to_csv('./results/test_predictions.csv', index=False)
print(f"📄 Test predictions saved to ./results/test_predictions.csv")

# Show sample predictions
print("\n📝 Sample Test Predictions:")
print(test_results.head(10).to_string())

# Summary
print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
print("="*50)
print(f"✅ Best Model Performance:")
print(f"   • Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
print(f"   • F1-Score: {eval_results['eval_f1']:.4f}")
print(f"   • Test samples processed: {len(test_texts)}")
print(f"\n📊 Test set statistics:")
print(f"   • Predicted Abusive: {(predictions == 1).sum()}")
print(f"   • Predicted Non-abusive: {(predictions == 0).sum()}")
print("\n🚀 You can now run: python app.py")
print("="*50)