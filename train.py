import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="肺部影像診斷系統", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fusion_model_streamlit.pth"
DATA_DIR_TB = "./TB_Chest"
TARGET_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class MultiDiseaseDataset(Dataset):
    def __init__(self, data_source, is_mnist=False):
        self.data_source = data_source
        self.is_mnist = is_mnist
    def __len__(self):
        return len(self.data_source)
    def __getitem__(self, idx):
        if self.is_mnist:
            img, label = self.data_source[idx]
            img = img.convert('RGB')
            label = int(label[0])
        else:
            img = Image.open(self.data_source.iloc[idx]['path']).convert('RGB')
            label = self.data_source.iloc[idx]['label']
        return TRANSFORM(img), torch.tensor(label, dtype=torch.long)


def get_model(model_type):
    if model_type == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 3)
    elif model_type == "ViT-B/16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 3)
    elif model_type == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 3)
    elif model_type == "ConvNeXt-Tiny":
        model = models.convnext_tiny(weights=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    
    return model.to(DEVICE)


def train():

    pass


def test():
    pass




def inference():
    pass