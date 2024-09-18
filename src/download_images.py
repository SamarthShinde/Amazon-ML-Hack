import os
import pandas as pd
from tqdm import tqdm
from src.utils import download_images  

def download_images_from_dataframe(df, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Downloading images'):
        image_link = row['image_link']
        image_path = os.path.join(image_dir, f"{row['index']}.jpg")
        if not os.path.exists(image_path):
            try:
                download_images(image_link, image_path)
            except Exception as e:
                print(f"Error downloading image {row['index']}: {e}")

if __name__ == '__main__':
    
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    # Download images for training data
    download_images_from_dataframe(train_df, image_dir='images/train')

    # Download images for test data
    download_images_from_dataframe(test_df, image_dir='images/test')