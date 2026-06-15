import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import librosa
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

SR_FIXED = 16000
TARGET_SEC = 1.0
TARGET_SAMP = int(TARGET_SEC * SR_FIXED)
MODEL_PATH = "final_model.pth"

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Audio Detector")
st.write("Upload a `.wav` file to verify if the voice is Genuine (Human) or Deepfake (AI-Generated).")

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['threshold'], device

try:
    model, THRESHOLD, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'final_model.pth' is in the exact same folder as this script.")
    st.stop()

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_audio(audio_file):
    """Replicates the exact center-crop preprocessing used during test evaluation."""
    y, _ = librosa.load(audio_file, sr=SR_FIXED)

    if len(y) < TARGET_SAMP:
        return None, "Audio is too short. Must be at least 1.0 seconds to extract a clean sample."
        
    start = (len(y) - TARGET_SAMP) // 2
    y_crop = y[start : start + TARGET_SAMP]

    S = librosa.feature.melspectrogram(y=y_crop, sr=SR_FIXED, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
    S_norm = np.flipud(S_norm)
    colored = (cm.magma(S_norm)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored).resize((128, 128), Image.LANCZOS)
    
    tensor_img = eval_transform(img).unsqueeze(0).to(device)
    return tensor_img, "success"
uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Audio"):
        with st.spinner("Extracting features and analyzing audio signature..."):
            tensor_img, status = process_audio(uploaded_file)
            
            if tensor_img is None:
                st.error(status)
            else:
                with torch.no_grad():
                    score = model(tensor_img).item()
                
                is_real = score >= THRESHOLD
                
                if is_real:
                    confidence = ((score - THRESHOLD) / (1.0 - THRESHOLD)) * 100
                    st.success(f"### Verdict: GENUINE (Human)")
                    st.write(f"**Confidence Score:** {confidence:.1f}%")
                else:
                    confidence = ((THRESHOLD - score) / THRESHOLD) * 100
                    st.error(f"### Verdict: DEEPFAKE (AI-Generated)")
                    st.write(f"**Confidence Score:** {confidence:.1f}%")