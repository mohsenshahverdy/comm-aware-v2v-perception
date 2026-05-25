# 🚗 V2V Cooperative Perception (V2VAM Extended)

> End-to-End Multi-Agent LiDAR-Based Cooperative Perception System  
> Designed for scalable, modular, and efficient Vehicle-to-Vehicle (V2V) perception.

---

## 🧠 Overview

Autonomous vehicles rely heavily on perception systems to understand their surroundings. However, **single-vehicle perception suffers from occlusions, blind spots, and limited field of view**.

This project implements a **cooperative perception pipeline**, where multiple connected vehicles share information to improve detection accuracy and environmental awareness.

### 🔥 Key Idea
Instead of relying only on ego-vehicle sensors, we **fuse features from multiple vehicles using attention mechanisms**.

---


---
## 🏗️ System Architecture

![Architecture](https://github.com/Mohammad-Amirifard/V2V_Cooperative_Perception/blob/main/images/Architecture.png)

### 🔄 Pipeline Overview

1. Input LiDAR Point Clouds  
2. Feature Extraction (BEV backbone)  
3. Feature Sharing between vehicles  
4. Attention-based Fusion (Criss-Cross Attention)  
5. Detection Head (Classification + Regression)  

---

## ⚙️ Key Features

- 🧠 Attention-based feature fusion  
- 🚗 Multi-agent cooperative perception  
- ⚡ Distributed multi-GPU training  
- 🔁 Checkpoint-based reproducibility  
- 📊 Cross-domain evaluation (CARLA & Culver City)  
- 🧩 Modular architecture design  

---

## 📦 Dataset

We use the **OPV2V Dataset (Open Perception for V2V)**

🔗 https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V  

### Includes:
- Multi-agent LiDAR data  
- CARLA simulated towns  
- Real-world Culver City dataset  

---

## 🧪 Experimental Pipeline

### 📌 Steps

1. Environment Setup  
2. Install Dependencies  
3. Clone Repository  
4. Install Project  
5. Configure Dataset Paths  
6. Train Model  
7. Resume Training  
8. Run Inference  

---

## 🚀 Quick Start (Kaggle)

You can directly run this project on Kaggle using the notebook below:

👉 **[Open Kaggle Notebook](https://www.kaggle.com/code/mohammadamirifard/v2v-cooperative-perception)**

---

### 🖥️ Requirements

- GPU: **T4 × 2 (recommended)**  
- Internet: ON  

---

### ⚠️ Notes

- Make sure to enable **GPU acceleration** in Kaggle:

### 🔧 Setup

```bash
git clone https://github.com/Mohammad-Amirifard/V2V_Cooperative_Perception.git
cd V2V_Cooperative_Perception

pip install -e .
