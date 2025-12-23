import streamlit as st
import os
from PIL import Image
from train import inference, train, test, TARGET_NAMES


# Streamlit App Configuration
st.set_page_config(page_title="肺部影像診斷系統", layout="wide")
st.title("🫁 肺部疾病 AI 診斷與訓練平台")
st.markdown("支援 **正常 (Normal)**、**肺炎 (Pneumonia)** 與 **肺結核 (Tuberculosis)** 三類分類。")

# Training Setting
st.sidebar.header("🛠️ Training Setting")
model_type = st.sidebar.selectbox("Model", ["ResNet50", "ViT-B16", "EfficientNet-B0", "ConvNeXt-Tiny"], index=0)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=3)
epochs = st.sidebar.slider("Epochs", 1, 10, 3)
lr = st.sidebar.select_slider("Learning Rate", options=[1e-3, 1e-4, 5e-5], value=1e-4)
train_btn = st.sidebar.button("🚀 Start Training!")

#  Tabs for Inference, Evaluation, and Training Log
tab1, tab2, tab3 = st.tabs(["🔍 Inference", "📊 Evaluation", "📝 Training Log"])
MODEL_PATH = f"weights/{model_type.lower()}.pth"


# --- Tab 1: Inference ---
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
                probs = inference(img, model_type, MODEL_PATH)
                for i, name in enumerate(TARGET_NAMES):
                    st.write(f"**{name}**")
                    st.progress(float(probs[i]))
                    st.write(f"機率: {probs[i]*100:.2f}%")


# --- Tab 3:  Train Log ---
with tab3:
    if train_btn:
        st.info("正在準備資料中...")      
        progress_bar = st.progress(0)        
        status_text = st.empty()
        train(model_type, lr, epochs, batch_size, MODEL_PATH, progress_bar, status_text)
        st.success("✅ 訓練完成！權重已儲存。")

# --- Tab 2: Testing ---
with tab2:
    if st.button("📈 評估模型效能"):
        try:
            test(model_type, MODEL_PATH, st)
        except FileNotFoundError:
            st.error("找不到本地 TB 測試資料，請確認 TB_Chest 目錄結構。")
