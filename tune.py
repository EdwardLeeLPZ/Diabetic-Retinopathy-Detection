import gin
import logging
import tensorflow as tf
import ray
from ray import tune

from train import Trainer
from evaluation.metrics import *
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from models.transfer_learning import *

"""Determines the basic information of the model

Parameters:
    model_name (string): name of the model (name list: 'VGG16', 'Simplified Inception', 'Simplified SEResNeXt', 'RepVGG', 'DenseNet201', 'EfficientNetB3') 
    model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
    config_path (string): path of config.gin (must be an absolute path on the server)
"""

model_name = 'EfficientNetB3'
model_type = 'regression'
config_path = '/home/RUS_CIP/st169530/dl-lab-2020-team09/diabetic_retinopathy(accomplished by Peizheng Li)/configs/tuning_config.gin'


def tuning(config):
    # set hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append('{}={}'.format(str(key), str(value)))

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # gin-config
    gin.parse_config_files_and_bindings([config_path], bindings)
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

    # set training loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # train the model
    trainer = Trainer(model, ds_train, ds_val, ds_info, model_type=model_type, run_paths=run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy * 100)

    # set validation loggers
    utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

    # evaluate the model
    trained_model = trainer.model_output()
    if model_type == 'regression':
        trained_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber(delta=0.3), metrics=[BinaryAccuracy(model_type=model_type)])
    elif model_type == 'binary_classification':
        trained_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[BinaryAccuracy(model_type=model_type)])
    elif model_type == 'multi_classification':
        trained_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[BinaryAccuracy(model_type=model_type)])

    result = trained_model.evaluate(ds_test, return_dict=True)
    test_accuracy = result['binary_accuracy']
    tune.report(test_accuracy=test_accuracy * 100)


# initialize ray
ray.init()

# run the training program
analysis = tune.run(tuning,
                    name="EfficientNetB3_fine_tuning",
                    local_dir="./ray_results",
                    num_samples=1,
                    resources_per_trial={"cpu": 48, "gpu": 1},
                    config={"folder": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            "efficientnetb3.trainable_rate": tune.quniform(0.01, 0.15, 0.01),
                            "Trainer.learning_rate": tune.loguniform(1e-9, 1e-5),
                            "output_block.dropout_rate": tune.quniform(0.3, 0.5, 0.05)})

# print the best result
print("Best config is:", analysis.get_best_config(metric="test_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
