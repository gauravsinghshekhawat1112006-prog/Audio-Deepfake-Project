Audio Deepfake Detector

A PyTorch and Streamlit-based machine learning pipeline designed to classify audio files as either Genuine (Human) or Deepfake (AI-Generated). 

 Methodology & Architecture
**Feature Extraction:** Audio files are standardized to a 16kHz sample rate. A strict 1.0-second center-crop is extracted to prevent padding artifacts. The waveform is converted into a 128x128 Magma Mel-Spectrogram, capturing both temporal and frequency vocal characteristics.
**Model Architecture:** A ResNet18 Convolutional Neural Network (CNN). The ImageNet classification head was stripped and replaced with a custom binary classification head (Flatten -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Sigmoid).
**Loss & Optimization:** Weighted Binary Cross-Entropy (BCE) Loss was utilized to handle class imbalances, optimized via Adam with a ReduceLROnPlateau scheduler.

## Performance Report & Metrics
The model was evaluated on a strictly held-out test dataset, yielding the following results based on an optimal validation threshold of **0.27**:

**Accuracy:** 51.67%
**F1 Score:** 68.10%
**Equal Error Rate (EER):** 17.44%

**Confusion Matrix:**
|               | Predicted Fake | Predicted Real |
| **True Fake** |       3        |      2101      |
| **True Real** |       0        |      2243      |

**Per-Class Recall:** Fake: 0.1% | Real: 100.0%

## Data Bias Autopsy: The "Clever Hans" Effect
While the model achieved **100.0% Validation Accuracy** during training, it suffered a catastrophic collapse on the blind Test Set (51.67% Accuracy). This project intentionally highlights a critical vulnerability in current deepfake datasets known as "Channel Leakage."

During training, the ResNet18 model bypassed learning actual human vocal tract frequencies. Instead, it exploited acoustic artifacts inherent to the dataset's origin (e.g., the presence of microphone hiss in crowdsourced human audio vs. the absolute digital silence of AI-generated audio). When the test set removed these background artifacts, the model defaulted to predicting all audio as "Real." This repository serves as a practical demonstration of why rigorous out-of-domain testing and symmetric preprocessing are mandatory in AI security.
