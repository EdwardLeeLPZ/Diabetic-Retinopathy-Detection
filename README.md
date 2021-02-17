# Diabetic-Retinopathy-Detection
Diabetic retinopathy (DR) is an eye disease that seriously impairs the vision of diabetic patients. In this project, we mainly developed a detection model based on deep convolutional neural networks. This model is based on IDRID and Kaggle EyePACS datasets, which are also augmented through Sample Pairing. After analysis using deep visualization, we selected various architectures, and the model reached an binary accuracy of nearly 90% through ensemble learning. In the end, we concluded that the size of the dataset is currently the most critical factor that determines the detection ability.

## Content
- Input pipeline with TFRecord of dataset IDRID und EyePACS;
- Various CNN models (VGG, Inception, SEResNeXt, RepVGG) built by ourselves;
- Different kinds of metrics (binary/multi-accuracy, binary/multi-confusion-matrix, precision and recall) and corresponding evaluations;
- Image preprocessing (cutting black edges, resizing, normalization) and different kinds of data augmentation (random rotation, random shear, random crop, random flip, random change of contrast/saturation/hue);
- Deep visualization (Grad-CAM, Guided Backpropagation, Guided Grad-CAM, Integrated Gradients);
- Transfer learning models (Densenet and Efficientnet);
- Ensemble learning models;
- Freely selectable detection models (binary classification, multi-classification, regression)
- Sample Pairing (Data augmentation based on image mixing)

## How to run the code
- Download the original data file and modify the parameter *load.data_dir* in `config.gin` and `tuning_config.gin` to your corresponding data directory
  - IDRID dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
  - Kaggle Challenge Dataset provided by EyePACS: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
- If you want to start training a new model, run `main.py` directly (To evaluate the model, just change the parameter *train* in `main.py` to False)
- If you want to start fine-tuning of the model, run `tune.py` directly
- If you want to use ensemble learning, run directly `ensemble_learning.py`, but make sure that there are already well trained models to be fused (you need to give names and types of these models to the parameter *model_list* in `ensemble_learning.py`)

Note: If you want to change the type and parameters of the model, you need to modify the parameters in `config.gin`, `tuning_config.gin`, `main.py` and `tune.py`

Note: The default parameters are: VGG16 regression model for `main.py`; EfficientNet regression model for `tune.py`

## Results
Single model results:

Model Name|Model Type|Binary Accuracy|5-Class Accuracy
----------|----------|---------------|----------------
VGG16|Regression|85.44%|53.40%
Inception|Regression|83.50%|49.51%
SEResNeXt|Regression|82.52%|51.46%
DenseNet|Regression|85.44%|57.28%
**DenseNet**|**5-Class Classification**|85.44%|**62.14%**
**EfficientNet**|**Regression**|**87.38%**|55.34%
EfficientNet|5-Class Classification|86.41%|54.39%
RepVGG|Regression|82.52%|45.63%

Ensemble learning results:

Metrics|Value
-------|-----
Binary Accuracy|88.35%
Binary Balanced Accuracy Score|89.62%
Precision|96.43%
Recall|84.38%
F1 Score|89.90%
5-class Accuracy|55.40%
5-class Balanced Accuracy Score|44.67%

EfficientNet Regression Model achieves the best binary classification performance, when DenseNet 5-Class Classification Model has the highest 5-Class accuracy.

Ensemble learning can effectively improve the overall performance of diabetic retinopathy detection.