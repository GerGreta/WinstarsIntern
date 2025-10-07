# AnimalNER_CV_ModelDemo

## Overview

This project demonstrates a pipeline for recognizing animals in images and extracting animal names from text, and checks if the text matches the image. It uses:
- Computer Vision (CV): PyTorch + ResNet18
- Named Entity Recognition (NER): HuggingFace Transformers (BERT)
- Pipeline: Combines text and image analysis

## Project Structure
```
Test_2/
├── data/
│   ├── images/        # Image dataset (subfolders for each animal class)
│   └── ner_data.json  # NER dataset
├── models/            # Trained models are saved here
├── notebooks/
│   └── demo_pipeline.ipynb  # Demonstration notebook
├── src/
│   ├── cv_inference.py      # Image classification
│   ├── ner_inference.py     # Text analysis
│   ├── pipeline.py          # Main pipeline logic
│   ├── cv_train.py          # Training script for CV model
│   └── ner_train.py         # Training script for NER model
├── requirements.txt         # Python dependencies
└── README.md
```

## Dataset
- Each animal class contains 108 images.
- This number is small for real training, but sufficient for demonstration purposes.
- Images are located in data/images/, with subfolders for each class.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/GerGreta/WinstarsIntern.git
cd Winstars_Intern
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the environment:

  ..Windows PowerShell: .\.venv\Scripts\Activate.ps1

  ..Bash/Linux/Mac: source .venv/bin/activate

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Ensure your data/ folder contains the images and ner_data.json.

## How to Use
1. Run CV Model Training
```
python src/cv_train.py
```
2. Run NER Model Training
```
python src/ner_train.py
```
3. Run the Pipeline
```
python src/pipeline.py
```
4. Demonstration Notebook
Open notebooks/ner_cv_demo.ipynb in Jupyter or PyCharm to see:
- Sample text and image inputs
- Detected animals in text
- Classified animals from images
- Matching results
- Images displayed inline

## Notes

All paths in the code use relative paths, so the project can run on any computer after cloning.

GPU is used if available for faster training and inference.
