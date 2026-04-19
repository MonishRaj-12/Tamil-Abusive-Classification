import pandas as pd

# Test the conversion logic
def convert_label(label):
    label_str = str(label).strip()
    if label_str.lower() == 'non-abusive':
        return 0
    elif label_str.lower() == 'abusive':
        return 1
    else:
        return None

# Load data
train_df = pd.read_csv('data/trainV2.csv')

print("Testing conversion on first 20 rows:")
print("="*50)
for i in range(min(20, len(train_df))):
    original = train_df.iloc[i]['Class']
    converted = convert_label(original)
    print(f"Row {i}: '{original}' -> {converted}")

print("\n" + "="*50)
print("Full conversion:")
train_df['label'] = train_df['Class'].apply(convert_label)
print(f"Non-Abusive (0): {(train_df['label'] == 0).sum()}")
print(f"Abusive (1): {(train_df['label'] == 1).sum()}")
print(f"Null: {train_df['label'].isna().sum()}")