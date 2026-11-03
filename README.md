# AI-MED AGH Recruitment — Machine Learning Projects

This repository contains two classification projects developed for the AI-MED AGH recruitment process. The PenDigits notebook compares five classifiers on handwritten digit trajectories. The BloodMNIST notebook trains a convolutional neural network and uses HiResCAM to investigate which image regions support different blood cell predictions.

## 1. PenDigits — handwritten digit classification

Each digit is represented by eight pairs of pen coordinates scaled to the range $[0, 100]$. The dataset contains 7,494 training examples and 3,498 test examples across 10 classes.

I compared multinomial logistic regression, a decision tree, a random forest, k-nearest neighbors, and XGBoost. I evaluated the models on both classification accuracy and inference time.

| Model | Selected configuration |
|---|---|
| Multinomial logistic regression | L1 regularization, `C=50`, `saga` solver |
| Decision tree | `max_depth=20`, `min_samples_split=2` |
| Random forest | 500 trees, `max_depth=20`, `max_features=6` |
| k-nearest neighbors | 3 neighbors, Manhattan distance |
| XGBoost | 500 estimators, `max_depth=4`, `learning_rate=0.1` |

**k-nearest neighbors achieved the highest test accuracy (97.5%).** XGBoost reached **96.5%** and recorded an inference time below 13 ms in the notebook benchmark.

[Open the PenDigits notebook](./Zadanie-1/Rozwiązanie-1.ipynb)

## 2. BloodMNIST — blood cell classification

The task was to classify eight types of peripheral blood cells from **28 × 28 RGB images**. I used the official BloodMNIST split of 11,959 training images, 1,712 validation images, and 3,421 test images.

I computed channel-wise normalization statistics on the training split and applied random flips and rotations during training. To account for differences in class frequency, I used weighted cross-entropy. Optuna selected the network depth, channel counts, hidden layer size, dropout rate, and learning rate based on validation accuracy. I then trained the selected two-block CNN on the combined training and validation sets and evaluated it on the held-out test set.

### Test results

| Accuracy | Macro precision | Macro recall | Macro F1 | Macro AUROC (one-vs-rest) |
|---:|---:|---:|---:|---:|
| 95.03% | 0.9453 | 0.9491 | 0.9462 | 0.9973 |

The model correctly classified **3,251 of 3,421 images**. Two error types accounted for 73 of the 170 mistakes: 42 images of class 6 were classified as class 3, and 31 images of class 3 were classified as class 5.

### Visual explanations with HiResCAM

I compared HiResCAM maps for correct and incorrect predictions. For selected mistakes, I generated two maps for the **same image**, one for the true class and one for the predicted class. This made it possible to compare the regions associated with each class score while keeping the input fixed.

For example, on images 109 and 122, the class 3 map highlighted the interior of a dark-stained region, while the class 5 map highlighted parts of its boundary. A similar pattern appeared on a correctly classified image, so the location of the highlighted region alone could not explain the mistake. At this resolution, the maps also provide limited detail about the underlying cell structures.

[Open the BloodMNIST notebook](./Zadanie-2A/Rozwiązanie-2A.ipynb)

## Running the notebooks

The project uses Python 3.14 and `uv`. To install the dependencies recorded in `uv.lock` and launch JupyterLab, run:

```bash
git clone https://github.com/juliet2112/Rekrutacja-AI-Med-2026-2027.git
cd Rekrutacja-AI-Med-2026-2027
uv sync --frozen
uv run jupyter lab
```

To reproduce the reported BloodMNIST results, run the data preparation cells and the model definition, then load `bloodmnist_model.pt` using the checkpoint cell before test evaluation. You can then run the metrics, confusion matrix, and HiResCAM cells without repeating the Optuna search or training. The original training run did not use a fixed random seed.

## Sources

- [PenDigits recruitment dataset](https://huggingface.co/datasets/AI-MED-AGH/Recruitment-Task-1)
- [BloodMNIST recruitment dataset](https://huggingface.co/datasets/AI-MED-AGH/Recruitment-Task-2A)
- Rachel Lea Draelos and Lawrence Carin, [*Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks*](https://arxiv.org/abs/2011.08891).
