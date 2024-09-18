import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import pytesseract
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

# Ensure pytesseract is installed: pip install pytesseract
# Ensure transformers is installed: pip install transformers

# Check GPU of MACBOOK Available 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

class EntityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.input_texts = df['input_texts']
        self.target_texts = df['target_texts']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = str(self.input_texts[idx])
        target_text = str(self.target_texts[idx])

        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )

        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=20,
            return_tensors='pt',
        )

        labels = target_encoding['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100  # Important for ignoring pad tokens in loss

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

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
    df['target_texts'] = df['entity_value'].fillna('')
    return df

if __name__ == '__main__':
    # Load training data
    train_df = pd.read_csv('dataset/train.csv')

    # Preprocess data
    train_df = preprocess_data(train_df, image_dir='images/train')

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    # Create dataset and dataloader
    train_dataset = EntityDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    epochs = 3
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc='Training'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Average loss for epoch {epoch + 1}: {avg_epoch_loss:.4f}")

    # Save the trained model
    model.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')
    print("Model and tokenizer saved to 'trained_model' directory.")