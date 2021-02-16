import gin
import logging

from evaluation.metrics import *
from visualization.deep_visualization import deep_visualize


@gin.configurable
def evaluate(model, ds_test, ds_info, model_type, run_paths):
    """evaluate performance of the model

    Parameters:
        model (keras.Model): keras model object to be evaluated
        ds_test (tf.data.Dataset): test set
        ds_info (dictionary): information and structure of dataset
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        run_paths (dictionary): storage path of model information
    """

    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    step = int(checkpoint.step.numpy())

    # compile the model
    if model_type == 'regression':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.Huber(),
                      metrics=[[BinaryConfusionMatrix(model_type=model_type)],
                               [BinaryAccuracy(model_type=model_type)],
                               [MultiConfusionMatrix(model_type=model_type)],
                               [MultiAccuracy(model_type=model_type)]])
    elif model_type == 'binary_classification':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[[BinaryConfusionMatrix(model_type=model_type)],
                               [BinaryAccuracy(model_type=model_type)]])
    elif model_type == 'multi_classification':
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[[BinaryConfusionMatrix(model_type=model_type)],
                               [BinaryAccuracy(model_type=model_type)],
                               [MultiConfusionMatrix(model_type=model_type)],
                               [MultiAccuracy(model_type=model_type)]])

    # summarize the evaluation results of each batch
    # (because model.evaluate() only returns the result of last batch, we need to calculate the sum or the average value of the whole dataset)
    for idx, (test_image, test_label) in enumerate(ds_test):
        batch_result = model.evaluate(test_image, test_label, return_dict=True)
        for key, value in batch_result.items():
            if (key.find('accuracy') != -1 or key.find('loss') != -1):
                batch_result[key] *= test_label.shape[0]
        if idx == 0:
            result = batch_result
        else:
            for key, value in batch_result.items():
                result[key] += batch_result[key]
    ds_test = ds_test.unbatch().batch(1)
    num_test = sum(1 for _ in ds_test)
    for key, value in result.items():
        if (key.find('accuracy') != -1 or key.find('loss') != -1) and num_test != 0:
            result[key] /= num_test

    # log the evaluation information
    logging.info(f"Evaluating at step: {step}...")
    for key, value in result.items():
        logging.info('{}:\n{}'.format(key, value))

    # perform deep visualization (only for the output of the first 128 images)
    for idx, (test_image, test_label) in enumerate(ds_test):
        deep_visualize(model, test_image, test_label, idx, model_type, run_paths, train=False)
        if idx >= 127:
            break
