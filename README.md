# Medical & Statistical Machine Learning Pipelines

This repository contains machine learning pipelines developed for the **AI-MED AGH** recruitment process. The project evaluates classical statistical learning on coordinate trajectory data and deep learning for multi-class peripheral blood cell classification, integrated with Explainable AI (XAI) for clinical interpretability.

---

## 🛠 Tech Stack

* **Environment & Package Management:** `uv`, Python 3.14
* **Deep Learning & Computer Vision:** PyTorch, torchvision, Optuna, OpenCV
* **Statistical & Classical Machine Learning:** scikit-learn, XGBoost
* **Data Processing & Analytics:** NumPy, Pandas, Matplotlib, Seaborn

---

## 🔬 Project Modules

### 1. PenDigits Classification (Statistical & Classical ML)
* **Objective:** Benchmark classical statistical classifiers and evaluate the trade-off between model complexity, inference latency, and predictive performance.
* **Dataset:** 16-dimensional coordinate trajectories normalized into $[0, 100]$ across 10 digit classes ($7\,494$ training and $3\,498$ test samples).
* **Models Evaluated:**
  * **Multinomial Logistic Regression:** `saga` solver, tuned with Stratified 5-Fold GridSearch (`C=50.0`, `penalty='l1'`).
  * **Decision Tree Classifier:** Optimized with Stratified 5-Fold CV (`max_depth=20`, `min_samples_split=2`).
  * **Random Forest Classifier:** Randomized Search tuning (`n_estimators=500`, `max_depth=20`, `max_features=6`).
  * **$k$-Nearest Neighbors:** Distance metric optimization (`n_neighbors=3`, Manhattan distance) — **Top Accuracy: 97.5%**.
  * **Gradient Boosting (`XGBClassifier`):** Fine-tuned tree depth and learning rate (`learning_rate=0.1`, `max_depth=4`, `n_estimators=500`) — **Best latency-accuracy trade-off: 96.5% accuracy, sub-13ms inference**.

### 2. BloodMNIST Classification (Deep Learning & Computer Vision)
* **Objective:** Classify 8 peripheral blood cell types from microscopic imagery and interpret internal network representations using model explainability.
* **Dataset:** $11\,959$ training, $1\,712$ validation, and $3\,421$ test color images ($28 \times 28 \times 3$).
* **Pipeline Highlights:**
  * **Imbalance Mitigation:** Calculated inverse-frequency loss weighting integrated into `nn.CrossEntropyLoss`.
  * **Data Augmentation:** Stochastic horizontal/vertical flips, affine rotations ($\pm 30^\circ$), and channel-wise normalization.
  * **Automated Hyperparameter Tuning:** Automated search via **Optuna** across convolutional depths, channel counts, linear layer dimensions, dropout rates, and learning rate.
  * **Evaluation Metrics:** **95.0% Accuracy**, **0.997 AUROC**, **94.6% Macro F1-score**.
* **Model Explainability (Grad-CAM):**
  * Implemented Gradient-weighted Class Activation Mapping on the final convolutional layer.
  * Generated class-discriminative saliency maps superimposed over raw microscopic inputs to isolate morphological features (nuclear contours vs. cytoplasm) for both true positives and misclassifications.

---

## ⚙️ Setup & Execution

### Prerequisites
Ensure `uv` is installed on your system. If not, install it via:
```bash
brew install uv
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/juliet2112/Rekrutacja-AI-Med-2026-2027.git
   cd Rekrutacja-AI-Med-2026-2027
   ```

2. Synchronize dependencies and virtual environment:
   ```bash
   uv sync
   ```

3. Run JupyterLab to explore and execute the notebooks:
   ```bash
   uv run jupyter lab
   ```

---

## 📚 References & Attribution
* **Grad-CAM:** Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017.
* **Datasets:** Hugging Face Hub (`AI-MED-AGH/Recruitment-Task-1`), MedMNIST v2 (`bloodmnist`).
