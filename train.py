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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR_TB = "./TB_Chest"
TARGET_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']
TRAIN_SET, TEST_SET = None, None

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


def get_model(model_type, num_classes = 3):
    if model_type == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "ViT-B/16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_type == "EfficientNet-B0":
        model = models.efficientnet_b0(pretrained=True)
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "ConvNeXt-Tiny":
        model = models.convnext_tiny(weights=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model.to(DEVICE)


def load_data():
    label_map = {
        "Normal": 0,
        "Tuberculosis": 2
    }

    tb_paths = []
    tb_labels = []
    for dir in os.listdir(DATA_DIR_TB):
        if dir not in label_map:
            continue

        label = label_map[dir]
        for f in os.listdir(os.path.join(DATA_DIR_TB, dir)):
            if f.lower().endswith(('.png', '.jpg')):
                tb_paths.append(os.path.join(DATA_DIR_TB, dir, f))
                tb_labels.append(label)
    
    tb_df = pd.DataFrame({
        "path": tb_paths,
        "label": tb_labels
    })
    
    train_tb_df, test_tb_df = train_test_split(tb_df, test_size=0.2, random_state=42)  

    mnist_train = PneumoniaMNIST(split="train", download=True)
    train_set = ConcatDataset([
        MultiDiseaseDataset(mnist_train, is_mnist=True),
        MultiDiseaseDataset(train_tb_df, is_mnist=False)
    ])

    tb_df = pd.DataFrame({'path': tb_paths, 'label': 2})
    mnist_test = PneumoniaMNIST(split="test", download=True)
    test_set = ConcatDataset([
        MultiDiseaseDataset(mnist_test, is_mnist=True),
        MultiDiseaseDataset(test_tb_df, is_mnist=False)
    ])

    return train_set, test_set



def train(model_type, lr, epochs, batch_size, model_path, progress_bar, status_text):
    global TRAIN_SET, TEST_SET
    if TRAIN_SET is None:
        TRAIN_SET, TEST_SET = load_data()
    loader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True)

    model = get_model(model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        status_text.text(f"Epoch {epoch+1}/{epochs} - 平均損失: {avg_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    torch.save(model.state_dict(), model_path)




def test(model_type, model_path, st):
    model = get_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    global TRAIN_SET, TEST_SET
    if TEST_SET is None:
        TRAIN_SET, TEST_SET = load_data()
    test_loader = DataLoader(TEST_SET, batch_size=16, shuffle=False)
    
    y_true, y_pred = [], []
    st.info("正在計算測試集預測結果...")        
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # 2. Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: Normal vs Pneumonia vs TB')
    st.pyplot(fig)
    
    # 3. Show classification report
    st.text("分類報告 (Classification Report):")
    st.code(classification_report(y_true, y_pred, target_names=TARGET_NAMES))




def inference(img, model_type, model_path):
    model = get_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
   
    return probs