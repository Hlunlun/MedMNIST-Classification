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

# ==========================================
# 1. 設定與模型路徑
# ==========================================
st.set_page_config(page_title="肺部影像診斷系統", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "fusion_model_streamlit.pth"
DATA_DIR_TB = "./TB_Chest"
TARGET_NAMES = ['Normal', 'Pneumonia', 'Tuberculosis']

# 影像轉換
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==========================================
# 2. 輔助類別與模型載入
# ==========================================
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


st.title("🫁 肺部疾病 AI 診斷與訓練平台")
st.markdown("支援 **正常 (Normal)**、**肺炎 (Pneumonia)** 與 **肺結核 (Tuberculosis)** 三類辨識。")

# --- 側邊欄：訓練參數 ---
st.sidebar.header("🛠️ Training Setting")
model_type = st.sidebar.selectbox("Model", ["ResNet50", "ViT-B/16", "EfficientNet-B0", "ConvNeXt-Tiny"], index=0)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-3, 1e-4, 5e-5], value=1e-4)
train_btn = st.sidebar.button("🚀 Start Training!")

# --- 主畫面：分頁設計 ---
tab1, tab2, tab3 = st.tabs(["🔍 Inference", "📊 Evaluation", "📝 Training Log"])

# --- Tab 1: 即時推論 ---
with tab1:
    st.header("上傳 X 光片進行預測")
    uploaded_file = st.file_uploader("選擇影像檔...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption='上傳的影像', use_container_width=True)
        
        with col2:
            if not os.path.exists(MODEL_PATH):
                st.warning("⚠️ 請先在側邊欄啟動訓練，產生模型權重。")
            else:
                model = get_model(model_type)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.eval()
                
                input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                
                # 顯示結果
                for i, name in enumerate(TARGET_NAMES):
                    st.write(f"**{name}**")
                    st.progress(float(probs[i]))
                    st.write(f"機率: {probs[i]*100:.2f}%")

# --- Tab 3: 訓練日誌 (實作訓練邏輯) ---
with tab3:
    if train_btn:
        st.info("正在準備資料中...")
        # 準備資料
        tb_paths = [os.path.join(root, f) for root, _, files in os.walk(DATA_DIR_TB) 
                    for f in files if "Tuberculosis" in root and f.lower().endswith(('.png', '.jpg'))]
        tb_df = pd.DataFrame({'path': tb_paths, 'label': 2})
        
        mnist_train = PneumoniaMNIST(split="train", download=True)
        train_set = ConcatDataset([
            MultiDiseaseDataset(mnist_train, is_mnist=True),
            MultiDiseaseDataset(tb_df, is_mnist=False)
        ])
        loader = DataLoader(train_set, batch_size=16, shuffle=True)
        
        model = get_model(model_type)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
        
        torch.save(model.state_dict(), MODEL_PATH)
        st.success("✅ 訓練完成！權重已儲存。")

# --- Tab 2: 效能評估 (完整三分類混淆矩陣) ---
with tab2:
    if st.button("📈 評估模型效能"):
        if not os.path.exists(MODEL_PATH):
            st.error("找不到模型，請先訓練。")
        else:
            model = get_model(model_type)
            # 確保此處也使用 map_location 防止裝置錯誤
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            
            # 1. 準備測試資料：MNIST (Normal/Pneumonia) + 本地 TB 資料
            mnist_test = PneumoniaMNIST(split="test", download=True)
            
            tb_paths = [os.path.join(root, f) for root, _, files in os.walk(DATA_DIR_TB) 
                        for f in files if "Tuberculosis" in root and f.lower().endswith(('.png', '.jpg'))]
            
            if len(tb_paths) == 0:
                st.error("找不到本地 TB 測試資料，請確認 TB_Chest 目錄結構。")
            else:
                tb_df = pd.DataFrame({'path': tb_paths, 'label': 2})
                # 取一部份 TB 資料作為測試評估 (例如最後 20%)
                _, test_tb_df = train_test_split(tb_df, test_size=0.2, random_state=42)
                
                test_set = ConcatDataset([
                    MultiDiseaseDataset(mnist_test, is_mnist=True),
                    MultiDiseaseDataset(test_tb_df, is_mnist=False)
                ])
                test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
                
                y_true, y_pred = [], []
                st.info("正在計算測試集預測結果...")
                
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs = imgs.to(DEVICE)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        y_true.extend(labels.numpy())
                        y_pred.extend(preds.cpu().numpy())
                
                # 2. 繪製完整的混淆矩陣
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.title('Confusion Matrix: Normal vs Pneumonia vs TB')
                st.pyplot(fig)
                
                # 3. 顯示分類報告
                st.text("分類報告 (Classification Report):")
                st.code(classification_report(y_true, y_pred, target_names=TARGET_NAMES))
