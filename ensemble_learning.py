import gin
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from evaluation.metrics import *
from input_pipeline import datasets
from utils import utils_params
from models.architectures import *
from models.transfer_learning import *

"""Determines the basic information of the ensemble learning

Parameters:
    model_list (dictionary): name and output type of each model 
                             (name list: 'VGG16', 'Simplified Inception', 'Simplified SEResNeXt', 'RepVGG', 'DenseNet201', 'EfficientNetB3') 
                             (type list: 'regression', 'binary_classification', 'multi_classification')
"""

model_list = {}
"""Template:
    model_list = {'run_2020-12-17T03-08-30-164667_VGG16_Rv01(85%&54%)': 'regression',
                  'run_2021-01-01T15-29-54-236026_EfficientNetB3_Rv01-2(87%&46%)': 'regression',
                  'run_2021-01-29T17-28-54-236026_EfficientNetB3_Rv03-4(86%&48%)': 'regression'}
"""

assert model_list != {}

# gin-config
gin.parse_config_files_and_bindings(['configs/config.gin'], [])

# record the predictions and labels of each model
regression_predictions_list = []
regression_label_list = []
multi_predictions_list = []
multi_label_list = []

# evaluate each model
for model_name, model_type in model_list.items():
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(model_type=model_type)
    # generate folder structures
    run_paths = utils_params.gen_run_folder(model_name)
    # setup model
    if model_name.find('VGG16') != -1:
        model = vgg_like(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name.find('Inception') != -1:
        model = simplified_inception(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name.find('SEResNeXt') != -1:
        model = simplified_seresnext(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'RepVGG':
        model = rep_vgg(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name.find('DenseNet201') != -1:
        model = densenet201(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name.find('EfficientNetB3') != -1:
        model = efficientnetb3(input_shape=(256, 256, 3), model_type=model_type)
    else:
        model = vgg_like(input_shape=(256, 256, 3), model_type=model_type)
    model.summary()
    # load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    # compile the model
    if model_type == 'regression':
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(delta=0.3), metrics=[BinaryAccuracy(model_type=model_type)])
    elif model_type == 'multi_classification':
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[BinaryAccuracy(model_type=model_type)])
    # calculate the predictions
    ps = []
    ls = []
    for test_images, test_labels in ds_test:
        predictions = model(test_images, training=False)
        ps.append(predictions.numpy())
        ls.append(test_labels.numpy())
    ps = np.concatenate(ps, axis=0)
    ls = np.concatenate(ls, axis=0)
    # change the original predictions into the standard predictions
    if model_type == 'regression':
        ps += 0.5
        regression_predictions_list.append(np.squeeze(ps))
        regression_label_list.append(np.squeeze(ls))
    elif model_type == 'multi_classification':
        ps = np.argmax(ps, axis=1).astype(np.int32)
        multi_predictions_list.append(np.squeeze(ps))
        multi_label_list.append(np.squeeze(ls))

# evaluate the ensemble learning model
if regression_predictions_list == [] and multi_predictions_list == []:
    print('Error: There is no prediction.')
else:
    print('---Evaluation of Ensemble Learning---')
    # fuse the different models with the same output type together
    if regression_predictions_list != []:
        multi_pred = np.mean(regression_predictions_list, axis=0)
        multi_pred = np.clip(np.floor(multi_pred), a_min=0.0, a_max=4.0).astype(np.int32)
        multi_pred = np.squeeze(multi_pred)
        multi_true = regression_label_list[0].astype(np.int32)
        multi_true = np.squeeze(multi_true)
    elif multi_predictions_list != []:
        multi_pred = np.array(multi_predictions_list)
        for i in range(multi_pred.shape[1]):
            multi_pred[0, i] = np.argmax(np.bincount(multi_pred[:, i]))
        multi_pred = multi_pred[0]
        multi_pred = np.squeeze(multi_pred)
        multi_true = multi_label_list[0].astype(np.int32)
        multi_true = np.squeeze(multi_true)
    # change the standard 5-class predictions into the standard binary predictions
    binary_pred = (multi_pred >= 2).astype(np.int32)
    binary_true = (multi_true >= 2).astype(np.int32)
    # calculate the accuracy, balanced accuracy score, confusion matrix, precision, recall and F1 score of the binary classification
    binary_accuracy = metrics.accuracy_score(binary_true, binary_pred)
    binary_balanced_accuracy = metrics.balanced_accuracy_score(binary_true, binary_pred)
    binary_confusion_matrix = metrics.confusion_matrix(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1_score = metrics.f1_score(binary_true, binary_pred)
    # output the result of the binary classification
    print('Binary-accuracy:\n{}'.format(binary_accuracy))
    print('Balanced Binary-accuracy:\n{}'.format(binary_balanced_accuracy))
    print('Binary-confusion matrix:\n{}'.format(binary_confusion_matrix))
    print('Precision:\n{}'.format(precision))
    print('Recall:\n{}'.format(recall))
    print('F1 score:\n{}'.format(f1_score))
    # calculate the accuracy, balanced accuracy score and confusion matrix of the 5-calss classification
    multi_accuracy = metrics.accuracy_score(multi_true, multi_pred)
    multi_balanced_accuracy = metrics.balanced_accuracy_score(multi_true, multi_pred)
    multi_confusion_matrix = metrics.confusion_matrix(multi_true, multi_pred)
    # output the result of the 5-calss classification
    print('Multi-accuracy:\n{}'.format(multi_accuracy))
    print('Balanced Multi-accuracy:\n{}'.format(multi_balanced_accuracy))
    print('Multi-confusion matrix:\n{}'.format(multi_confusion_matrix))
    print('-------------------------------------')
    # plot the ROC and PRC of the ensemble learning model
    if regression_predictions_list != []:
        pred = np.clip((np.mean(regression_predictions_list, axis=0) + 0.5) / 4.0, a_min=0.0, a_max=1.0).astype(np.float64)
    elif multi_predictions_list != []:
        pred = np.clip(multi_pred.astype(np.float64) / 4.0, a_min=0.0, a_max=1.0).astype(np.float64)
    pred = np.squeeze(pred)
    true = binary_true

    fpr, tpr, _ = metrics.roc_curve(true, pred, pos_label=1.0)
    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    precision_list, recall_list, _ = metrics.precision_recall_curve(true, pred, pos_label=1.0)
    pr_display = metrics.PrecisionRecallDisplay(precision=precision_list, recall=recall_list).plot()

    plt.show()
