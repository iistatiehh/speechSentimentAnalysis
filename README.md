# Speech Emotion Recognition

Deep learning project for classifying emotions from speech audio using multiple approaches including traditional feature extraction and transformer-based models.
[Speech Sentiment Recognition Project Documentation.pdf](https://github.com/user-attachments/files/21293534/Speech.Sentiment.Recognition.Project.Report2.pdf)

## Overview

This project explores four different approaches for speech emotion recognition:
- 1D vector features with ANN/CNN
- Custom CNN with image-based features
- ResNet50 with spectrogram images
- Wav2Vec2 fine-tuning (best performing)

## Dataset

Combined dataset from RAVDESS, CREMA, SAVEE, and TESS containing ~12,161 audio samples across 8 emotion classes: happy, sad, angry, fear, disgust, calm, neutral, and surprise.

## Results

Best performance achieved with Wav2Vec2 fine-tuning (5 emotion classes):
- **Accuracy**: 85.78%
- **Macro F1-Score**: 87.21%

## Important Note

**Large files not included**: Processed datasets and trained models (including 3 versions of Wav2Vec2 models) are not included in this repository due to size constraints.

To use this project:
1. Run the data processing files first
2. Then run the corresponding training files in order

## Requirements

- Python 3.8+
- PyTorch
- transformers (Hugging Face)
- librosa
- scikit-learn
- numpy
- pandas

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Process data first
python data_processing.py

# Then train models
python train_model.py
```

