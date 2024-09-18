
# Amazon ML Hack

## Overview

**Amazon ML Hack** is a machine learning project developed to extract entity values from product images. This capability is essential for various domains such as healthcare, e-commerce, and content moderation, where accurate product information is crucial. As digital marketplaces grow, many products lack detailed textual descriptions, making it necessary to derive key details directly from images. This project leverages Optical Character Recognition (OCR) and advanced machine learning models to extract vital information like weight, volume, voltage, wattage, dimensions, and more.

## Project Structure

```
student_resource_three/
├── CSV_data_set/
│   ├── clean_data.csv/
│   │   └── train.csv
│   ├── output_prediction.csv/
│   │   └── output_prediction.csv
│   └── data_set/
│       ├── train.csv
│       ├── test.csv
│       ├── sample_test.csv
│       ├── sample_test_out.csv
│       └── sample_test_out_fail.csv
├── models/
│   └── best_model.keras
├── SRC/
│   ├── download_image.py
│   ├── inference.py
│   ├── sanity.py
│   ├── test.py
│   ├── train.py
│   ├── constants.py
│   └── utils.py
├── requirements.txt
└── README.md
```

- **CSV_data_set/**: Contains all CSV files related to the dataset, including training data, test data, and prediction outputs.
- **models/**: Stores the trained machine learning model (`best_model.keras`).
- **SRC/**: Contains all source code scripts used for downloading images, training the model, running inference, and utilities.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **README.md**: Project documentation.

## Setup Instructions

### Prerequisites

- **Operating System**: macOS (developed on MacBook with M1 chip)
- **Python Version**: 3.10.10 (64-bit)
- **Git**: Installed on your system
- **Homebrew**: Installed for managing packages (if not already installed)

### Step 1: Clone the Repository

1. Open your terminal.
2. Navigate to the directory where you want to clone the repository.
3. Clone the repository using Git:

   ```bash
   git clone https://github.com/SamarthShinde/Amazon-ML-Hack.git
   cd Amazon-ML-hack
   ```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)

Creating a virtual environment ensures that dependencies are managed separately from your global Python installation.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

Install all required Python packages using the `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR

The project uses Tesseract OCR for text extraction from images.

1. **Install via Homebrew:**

   ```bash
   brew install tesseract
   ```

2. **Verify Installation:**

   ```bash
   tesseract --version
   ```

### Step 5: Download Images

Use the `download_image.py` script to download images for both training and testing datasets.

```bash
python SRC/download_image.py
```

**Script Functionality:**

- Downloads images based on URLs provided in the CSV files.
- Saves images in designated directories within the project structure.

### Step 6: Train the Model

Train your machine learning model using the `train.py` script.

```bash
python SRC/train.py
```

**Script Functionality:**

- Performs OCR on training images to extract textual data.
- Preprocesses data and trains the `best_model.keras`.
- Saves the trained model in the `models/` directory.

### Step 7: Run Inference

Generate predictions on the test dataset using the `inference.py` script.

```bash
python SRC/inference.py
```

**Script Functionality:**

- Performs OCR on test images.
- Loads the trained model to generate predictions.
- Post-processes predictions to ensure they meet the required format.
- Saves output predictions in `CSV_data_set/output_prediction.csv`.

### Step 8: Run Sanity Checker

Ensure that your output predictions are correctly formatted using the `sanity.py` script.

```bash
python SRC/sanity.py --submission_file CSV_data_set/output_prediction.csv
```

**Script Functionality:**

- Validates the format and correctness of the prediction CSV file.
- Outputs a success message if the file passes all checks.

## Usage Notes

- **MacBook-Specific Instructions:** This project is developed and tested on a MacBook with an M1 chip utilizing GPU acceleration via Metal Performance Shaders (MPS). If you're using a different system or environment, you might need to adjust certain configurations, especially related to GPU support.
- **Environment Differences:** Ensure that all dependencies are compatible with your system. Some packages might have different installation procedures or dependencies based on the operating system.



### **. PyTorch MPS Support Issues**

**Problem:** PyTorch doesn't recognize the MPS device on MacBook.

**Solution:** Ensure you're using PyTorch version `1.12.0` or higher. Install or upgrade PyTorch:

```bash
pip install --upgrade torch torchvision torchaudio
```

Verify MPS availability in Python:

```python
import torch
print(torch.backends.mps.is_available())
```

Should output `True`.

---

## **. References**


- [PyTorch MPS Backend](https://pytorch.org/docs/stable/backends.html#torch.backends.mps)
- [Tesseract OCR Documentation](https://github.com/tesseract-ocr/tesseract)

---

## **8. Contact**

For any questions or assistance, feel free to contact me at [samarth.shinde505@gmail.com](mailto:samarth.shinde505@gmail.com).

---
