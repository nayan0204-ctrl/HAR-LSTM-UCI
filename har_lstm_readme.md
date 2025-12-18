# 🧠 Human Activity Recognition using Deep Learning (LSTM)

This project implements a **Human Activity Recognition (HAR)** system using a **2-layer LSTM deep learning model** trained on the **UCI HAR Dataset**. The model classifies human activities from multivariate time-series sensor data collected via smartphone accelerometers and gyroscopes.

---

## 📌 Project Overview

- **Problem Type:** Multiclass time-series classification
- **Dataset:** UCI Human Activity Recognition (HAR)
- **Model:** 2-Layer Long Short-Term Memory (LSTM)
- **Framework:** TensorFlow / Keras

---

## 📊 Dataset Description

The UCI HAR dataset contains sensor signals collected from **30 subjects** performing **6 daily activities** while carrying a smartphone.

### Activities:
1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

### Input Data Shape:
- **Samples:** 10,299 (Train: 7,352 | Test: 2,947)
- **Timesteps:** 128
- **Features:** 9 (3-axis accelerometer + 3-axis gyroscope)

```
X_train shape: (7352, 128, 9)
X_test shape:  (2947, 128, 9)
```

Labels are one-hot encoded:
```
y_train shape: (7352, 6)
y_test shape:  (2947, 6)
```

---

## 🏗️ Model Architecture

```text
Input (128 timesteps × 9 features)
   ↓
LSTM (64 units, return_sequences=True)
   ↓
Dropout (0.5)
   ↓
LSTM (64 units)
   ↓
Dropout (0.5)
   ↓
Dense (6 units, Softmax)
```

---

## ⚙️ Training Details

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 64
- **Epochs:** 30
- **Validation Split:** 20%
- **Regularization:** Dropout

---

## 📈 Results

### Overall Performance:
- **Test Accuracy:** 92.26%

### Classification Report:

| Activity | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| WALKING | 0.94 | 0.93 | 0.93 |
| WALKING_UPSTAIRS | 0.98 | 0.98 | 0.98 |
| WALKING_DOWNSTAIRS | 0.92 | 1.00 | 0.96 |
| SITTING | 0.86 | 0.77 | 0.81 |
| STANDING | 0.84 | 0.88 | 0.86 |
| LAYING | 1.00 | 1.00 | 1.00 |

---

## 📉 Evaluation Techniques

- Confusion Matrix (visualized using Seaborn heatmap)
- Precision, Recall, F1-score
- Training vs Validation Accuracy curves

---

## 💾 Model Saving

The trained model is saved in the modern Keras format:

```
HAR_2Layer_LSTM.keras
```

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/HAR-LSTM-UCI.git
cd HAR-LSTM-UCI

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# Run notebook / script
jupyter notebook
```

---

## 🧠 Key Learning Outcomes

- Handling multivariate time-series data
- Building stacked LSTM architectures
- Preventing overfitting using dropout
- Evaluating deep learning models using confusion matrices and F1-score

---

## 🔮 Future Improvements

- CNN-LSTM hybrid model
- Attention-based LSTM
- Real-time HAR inference
- Deployment on edge devices

---

## 📚 References

- UCI Machine Learning Repository: Human Activity Recognition Dataset
- TensorFlow/Keras Documentation

---

## 👤 Author

**Nayan Suhane**  
MSc / BTech – Machine Learning & Signal Processing Enthusiast  

---

⭐ If you like this project, give it a star!

