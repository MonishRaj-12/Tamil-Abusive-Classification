import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_tamil_text(text):
    """Clean and normalize Tamil text"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags but keep words
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters (keep Tamil unicode range and basic punctuation)
    # Tamil Unicode range: \u0B80-\u0BFF
    tamil_pattern = r'[^\u0B80-\u0BFF\s\w\.\,\!\!\?\-]'
    text = re.sub(tamil_pattern, '', text)
    return text

def load_and_prepare_data(train_path, test_path):
    """Load and prepare datasets"""
    # Load training data
    train_df = pd.read_csv(train_path)
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # Assuming columns: 'text' (contains Tamil sentences) and 'class' (abusive/non-abusive)
    # Map labels to binary values
    label_map = {'abusive': 1, 'non-abusive': 0, 'Abusive': 1, 'Non-abusive': 0}
    
    train_df['label'] = train_df['class'].map(label_map)
    
    # Clean text
    train_df['clean_text'] = train_df['text'].apply(clean_tamil_text)
    
    # Remove empty rows
    train_df = train_df[train_df['clean_text'].str.len() > 0]
    
    # Split into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['clean_text'].values,
        train_df['label'].values,
        test_size=0.1,
        random_state=42,
        stratify=train_df['label'].values
    )
    
    # Load test data
    test_df = pd.read_csv(test_path)
    test_df['clean_text'] = test_df['text'].apply(clean_tamil_text)
    test_texts = test_df['clean_text'].values
    
    print(f"Train: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}")
    print(f"Class distribution - Abusive: {sum(train_labels)}, Non-abusive: {len(train_labels)-sum(train_labels)}")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels