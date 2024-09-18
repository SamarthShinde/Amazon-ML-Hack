# predict.py

import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pytesseract
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# First Check src/constants.py contains ALLOWED_UNITS dictionary
from src.constants import ALLOWED_UNITS

# Check if MPS(MACBOOK GPU) is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

def perform_ocr(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"Error performing OCR on image {image_path}: {e}")
        text = ''
    return text

def preprocess_data(df, image_dir):
    df['image_path'] = df['index'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    tqdm.pandas(desc='Performing OCR')
    df['ocr_text'] = df['image_path'].progress_apply(perform_ocr)
    df['input_texts'] = df['entity_name'] + ' : ' + df['ocr_text']
    return df

def generate_predictions(df, tokenizer, model):
    model.eval()
    predictions = []
    for input_text in tqdm(df['input_texts'], desc='Generating predictions'):
        input_encoding = tokenizer(
            input_text,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=20,
                num_beams=2,
                early_stopping=True,
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
    df['raw_prediction'] = predictions
    return df

def post_process_prediction(pred, entity_name):
    pred = pred.lower().strip()
    match = re.match(r'^([\d\.]+)\s*(\w+)$', pred)
    if match:
        number, unit = match.groups()
        allowed_units = ALLOWED_UNITS.get(entity_name, [])
        if unit in allowed_units:
            return f"{number} {unit}"
    return ""

if __name__ == '__main__':
    # Load test data
    test_df = pd.read_csv('dataset/test.csv')

    # Preprocess data
    test_df = preprocess_data(test_df, image_dir='images/test')

    # Load the trained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('trained_model')
    model = T5ForConditionalGeneration.from_pretrained('trained_model').to(device)

    # Generate predictions
    test_df = generate_predictions(test_df, tokenizer, model)

    # Post-process predictions
    tqdm.pandas(desc='Post-processing predictions')
    test_df['prediction'] = test_df.progress_apply(
        lambda row: post_process_prediction(row['raw_prediction'], row['entity_name']),
        axis=1
    )

    # Prepare the submission file
    output_df = test_df[['index', 'prediction']]
    output_df.to_csv('test_out.csv', index=False)
    print("Predictions saved to 'test_out.csv'.")

    # Optionally, run the sanity checker
    # os.system('python src/sanity.py --submission_file test_out.csv')