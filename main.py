import gin
import logging
import tensorflow as tf
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from models.transfer_learning import *

"""Determines the basic information of the model

Parameters:
    model_name (string): name of the model (name list: 'VGG16', 'Simplified Inception', 'Simplified SEResNeXt', 'RepVGG', 'DenseNet201', 'EfficientNetB3') 
    model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
    train (bool): if you want to train the model, set to True; if you want to evaluate (and visualize) the model, set to False
    folder (string): the folder, which contains the checkpoints, logs, summary, configs and deep visualization images of the model (Default is 'CNN')
"""

model_name = 'VGG16'
model_type = 'regression'
train = True
folder = 'CNN'

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', train, 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(folder)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(model_type=model_type)

    # setup model
    if model_name == 'VGG16':
        model = vgg_like(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'Simplified Inception':
        model = simplified_inception(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'Simplified SEResNeXt':
        model = simplified_seresnext(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'RepVGG':
        model = rep_vgg(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'DenseNet201':
        model = densenet201(input_shape=(256, 256, 3), model_type=model_type)
    elif model_name == 'EfficientNetB3':
        model = efficientnetb3(input_shape=(256, 256, 3), model_type=model_type)
    else:
        model = vgg_like(input_shape=(256, 256, 3), model_type=model_type)
    model.summary()

    if FLAGS.train:
        # set training loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        # train the model
        trainer = Trainer(model, ds_train, ds_val, ds_info, model_type=model_type, run_paths=run_paths)
        for _ in trainer.train():
            continue
    else:
        # set validation loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        # evaluate the model
        evaluate(model, ds_test, ds_info, model_type=model_type, run_paths=run_paths)


if __name__ == "__main__":
    app.run(main)
