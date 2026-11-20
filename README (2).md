# Parkinson's Disease Detection from Brain MRI Images

A deep learning project that classifies brain MRI images into **Healthy** and **Parkinson** categories using transfer learning with a pretrained DenseNet121 model. The notebook also uses **Grad-CAM** to generate visual explanations of the model's predictions.

> **Disclaimer:** This is an academic/experimental project, not a clinically validated diagnostic system. It must not be used to diagnose Parkinson's disease or make medical decisions.

## Project Overview

The project trains a binary image-classification model on MRI images organized into two classes:

- `HC_new` — Healthy
- `PD_new` — Parkinson

The notebook covers image preprocessing, an 80/20 train-test split, model training, evaluation, single-image prediction, model saving, and Grad-CAM visualization.

## Tech Stack

- Python
- PyTorch and Torchvision
- DenseNet121 (ImageNet-pretrained)
- NumPy and Matplotlib
- Scikit-learn
- Pillow
- Grad-CAM (`grad-cam` package)
- Google Colab / Google Drive

## Dataset Structure

The notebook expects the dataset to be stored in Google Drive with this directory structure:

```text
Disease_Prediction/
├── HC_new/
│   ├── image1.jpg
│   └── ...
└── PD_new/
    ├── image1.jpg
    └── ...
```

The dataset path in the notebook is currently set to:

```text
/content/drive/MyDrive/Disease_Prediction-20231213T083430Z-001/Disease_Prediction
```

Update `DATASET_PATH` in the notebook if your dataset is stored elsewhere. The dataset itself is not included in this repository/notebook.

## Workflow

1. **Load the dataset:** Read the images from the `HC_new` and `PD_new` folders using `torchvision.datasets.ImageFolder`.
2. **Preprocess images:** Resize images to `224 × 224`, convert them to tensors, and normalize them using ImageNet mean and standard deviation. Training transforms also include random horizontal flipping, rotation, and color jitter.
3. **Split the data:** Use an 80% training and 20% testing split.
4. **Prepare data loaders:** Load images in batches of 16.
5. **Build the model:** Load ImageNet-pretrained DenseNet121 and replace its classifier with a binary classification head containing dropout and fully connected layers.
6. **Train:** Train for 20 epochs using `BCEWithLogitsLoss` and the Adam optimizer with a learning rate of `0.0001`.
7. **Evaluate:** Calculate accuracy, precision, recall, and F1-score; display a confusion matrix, classification report, and ROC curve with AUC.
8. **Predict:** Run inference on an individual MRI image. A sigmoid probability above `0.5` is classified as Parkinson; otherwise it is classified as Healthy.
9. **Explain predictions:** Apply Grad-CAM to the final DenseNet121 feature layer and overlay the resulting heatmap on the MRI image.

## Model Architecture and Training Configuration

| Setting | Value |
|---|---|
| Base model | DenseNet121 pretrained on ImageNet |
| Input image size | 224 × 224 |
| Batch size | 16 |
| Train/test split | 80% / 20% |
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Binary Cross Entropy with Logits |
| Output | One logit for binary classification |
| Classification threshold | 0.5 |

The replacement classifier uses dropout (`0.4`), a fully connected layer with 256 units, ReLU, another dropout layer (`0.3`), and a final single-unit output layer.

## Installation

Install the required packages:

```bash
pip install torch torchvision scikit-learn opencv-python matplotlib pillow grad-cam
```

Package versions are not pinned in the notebook.

## Run the Notebook

1. Open `Abhishek_parkinson.ipynb` in [Google Colab](https://colab.research.google.com/) or a compatible Jupyter environment.
2. Make the dataset available and update `DATASET_PATH` to point to the `Disease_Prediction` folder.
3. If using Google Colab, run the Google Drive mount cell and grant access.
4. Run the notebook cells in order to train and evaluate the model.
5. The trained weights are saved as `Parkinson_DenseNet121.pth`.
6. Run the prediction and Grad-CAM cells to inspect an individual image.

For local Jupyter use, remove or replace the Google Colab-specific `google.colab.drive` mount code and provide a local dataset path.

## Evaluation

The notebook calculates the following metrics on its test split:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- ROC curve and ROC-AUC score

The actual metric values depend on the dataset and training run. Run the notebook to generate them; no performance figures are claimed in this README.

## Grad-CAM Visualization

Grad-CAM highlights image regions that contribute to the model's output. In this notebook, the target layer is the final DenseNet121 feature block (`model.features[-1]`). The heatmap is overlaid on the resized MRI image to provide a visual explanation of the prediction.

These heatmaps are exploratory model explanations. They do not establish that the highlighted regions are medically meaningful.

## Output

- `Parkinson_DenseNet121.pth` — saved model weights (`state_dict`)
- Evaluation plots and metric reports displayed in the notebook
- Single-image prediction and confidence
- Original MRI and Grad-CAM heatmap visualization

## Important Implementation Note

The notebook defines separate training and testing transforms, but the `ImageFolder` dataset is created with `train_transform` before the random split. Consequently, the test subset also receives the training augmentations during evaluation. For a more reliable evaluation, create separate dataset instances or apply transforms by subset so that the test data uses only resizing, tensor conversion, and normalization. Also consider using a reproducible, stratified split and an independent validation/test dataset when developing the model.

## Future Improvements

- Use separate training, validation, and test datasets with appropriate transforms.
- Use a stratified split and set random seeds for reproducibility.
- Track validation loss and metrics during training.
- Evaluate on an independent dataset and investigate class balance.
- Save the class mapping and preprocessing configuration alongside the model weights.
- Validate Grad-CAM explanations and model performance with appropriate domain expertise.

## Author

**Abhishek Prasad**

---

If you use or extend this project, please describe the dataset source and evaluation setup clearly.
