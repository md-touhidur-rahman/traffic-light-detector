# 🚦 Traffic Light State Detection with CNN (CARLA Dataset)

![Model Accuracy](https://img.shields.io/badge/accuracy-99.22%25-brightgreen)
![Model Type](https://img.shields.io/badge/model-CNN-blue)
![Framework](https://img.shields.io/badge/framework-TensorFlow-orange)

This project implements a Convolutional Neural Network (CNN) to detect and classify traffic light states using simulated data from the CARLA simulator.

## 🔍 Objective
To simulate real-time perception in autonomous vehicles by classifying traffic light states (Red, Green, Yellow, and Back) using RGB images, contributing to safer and smarter connected mobility.

## 📂 Dataset
- Source: CARLA traffic light dataset (manually labeled)
- Classes: `red`, `green`, `yellow`, `back`
- Structure: Pre-split into `train/` and `val/` folders

## 🧠 Model Architecture
- CNN using TensorFlow/Keras
- Input size: 64x64 RGB
- Layers: Conv2D, MaxPooling, Dense, Dropout
- Activation: ReLU + Softmax
- Output: 4-class classification

## 📊 Results
- **Validation Accuracy:** 99.22%
- **F1 Scores:** Near-perfect on all classes
- **Confusion Matrix & Predictions:** See `sample_outputs/`

## 📁 Files
| File/Folder | Description |
|-------------|-------------|
| `traffic_light_classifier.ipynb` | Colab notebook (full training + evaluation) |
| `sample_outputs/` | Sample predictions, confusion matrix |
| `model.h5` | (Optional) Trained model weights |
| `README.md` | Project summary |
| `requirements.txt` | Dependencies (if added) |

## 🧪 Evaluation
- Classification report with precision/recall
- Confusion matrix
- Visual inspection of predictions

## 🛠️ Tools Used
- Python (Colab)
- TensorFlow / Keras
- OpenCV, Matplotlib
- Seaborn, Scikit-learn

## 🚀 Future Work
- Apply model on real-world traffic footage
- Integrate with ROS or a live perception pipeline
- Extend to traffic sign recognition

---


