import gin
import logging
import tensorflow as tf

from evaluation.metrics import *
from visualization.deep_visualization import deep_visualize


@gin.configurable
class Trainer(object):
    """Trainer for the model training"""

    def __init__(self, model, ds_train, ds_val, ds_info, model_type, run_paths, total_steps, visual_interval, log_interval, ckpt_interval, learning_rate=1e-3):
        """Trainer parameters initialization"""

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.visual_interval = visual_interval  # step size for visualization
        self.log_interval = log_interval  # step size for logging
        self.ckpt_interval = ckpt_interval  # step size for saving checkpoints
        self.model_type = model_type

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # loss objective and metrics
        if self.model_type == 'regression':
            # training metrics
            self.train_binary_accuracy = tf.keras.metrics.Accuracy(name='train_binary_accuracy')
            self.train_multi_accuracy = tf.keras.metrics.Accuracy(name='train_multi_accuracy')
            # validation metrics
            self.val_binary_accuracy = tf.keras.metrics.Accuracy(name='val_binary_accuracy')
            self.val_multi_accuracy = tf.keras.metrics.Accuracy(name='val_multi_accuracy')
            # loss
            self.loss_object = tf.keras.losses.Huber(delta=0.3)
        elif self.model_type == 'binary_classification':
            # training metrics
            self.train_binary_accuracy = tf.keras.metrics.Accuracy(name='train_binary_accuracy')
            # validation metrics
            self.val_binary_accuracy = tf.keras.metrics.Accuracy(name='val_binary_accuracy')
            # loss
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif self.model_type == 'multi_classification':
            # training metrics
            self.train_binary_accuracy = tf.keras.metrics.Accuracy(name='train_binary_accuracy')
            self.train_multi_accuracy = tf.keras.metrics.Accuracy(name='train_multi_accuracy')
            # validation metrics
            self.val_binary_accuracy = tf.keras.metrics.Accuracy(name='val_binary_accuracy')
            self.val_multi_accuracy = tf.keras.metrics.Accuracy(name='val_multi_accuracy')
            # loss
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # other metrics
        self.train_precision = tf.keras.metrics.Precision(name='train_precision')
        self.train_recall = tf.keras.metrics.Recall(name='train_recall')
        self.val_precision = tf.keras.metrics.Precision(name='val_precision')
        self.val_recall = tf.keras.metrics.Recall(name='val_recall')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        # summary writer
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_train'])
        self.val_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_val'])
        self.profiler_summary_path = self.run_paths['path_summary_profiler']

        # checkpoint manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10)

    @tf.function
    def train_step(self, images, labels):
        """one-step training"""

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # calculate the training loss and training metrics
        self.train_loss(loss)
        # for different model output types, choose different postprocessing methods for the model predictions
        if self.model_type == 'regression':
            predictions = tf.cast(tf.clip_by_value(predictions + 0.5, clip_value_min=0, clip_value_max=4), tf.int32)
            self.train_multi_accuracy(labels, predictions)
            predictions = tf.cast(predictions >= 2, dtype=tf.int32)
            labels = tf.cast(labels >= 2, dtype=tf.int32)
        elif self.model_type == 'binary_classification':
            predictions = tf.cast(tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1)), dtype=tf.int32)
        elif self.model_type == 'multi_classification':
            predictions = tf.cast(tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1)), dtype=tf.int32)
            self.train_multi_accuracy(labels, predictions)
            predictions = tf.cast(predictions >= 2, dtype=tf.int32)
            labels = tf.cast(labels >= 2, dtype=tf.int32)
        self.train_binary_accuracy(labels, predictions)
        self.train_precision(labels, predictions)
        self.train_recall(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        """one-step validation"""

        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        v_loss = self.loss_object(labels, predictions)
        # calculate the validation loss and validation metrics
        self.val_loss(v_loss)
        # for different model output types, choose different postprocessing methods for the model predictions
        if self.model_type == 'regression':
            predictions = tf.cast(tf.clip_by_value(predictions + 0.5, clip_value_min=0, clip_value_max=4), tf.int32)
            self.val_multi_accuracy(labels, predictions)
            predictions = tf.cast(predictions >= 2, dtype=tf.int32)
            labels = tf.cast(labels >= 2, dtype=tf.int32)
        elif self.model_type == 'binary_classification':
            predictions = tf.cast(tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1)), dtype=tf.int32)
        elif self.model_type == 'multi_classification':
            predictions = tf.cast(tf.reshape(tf.math.argmax(predictions, axis=1), shape=(-1, 1)), dtype=tf.int32)
            self.val_multi_accuracy(labels, predictions)
            predictions = tf.cast(predictions >= 2, dtype=tf.int32)
            labels = tf.cast(labels >= 2, dtype=tf.int32)
        self.val_binary_accuracy(labels, predictions)
        self.val_precision(labels, predictions)
        self.val_recall(labels, predictions)

    def train(self):
        """Complete training process"""

        # set the profiler (optional)
        # options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3, python_tracer_level=1, device_tracer_level=1)
        # tf.profiler.experimental.start(self.profiler_summary_path, options=options)

        # record the model structure
        logging.info(self.model.summary())

        # record the current optimal accuracy and loss, which is used for early stopping
        max_accuracy_record = 0
        min_loss_record = float("inf")

        # if training is interrupted unexpectedly, resume the model from here and continue training
        # or if it is the first step of training, start training from the beginning
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            tf.print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.step.assign_add(1)
        else:
            tf.print("Initializing from scratch.")

        for idx, (images, labels) in enumerate(self.ds_train):
            step = int(self.ckpt.step.numpy())
            # perform one-step training
            self.train_step(images, labels)

            # write train summary to tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                tf.summary.scalar('train_binary_accuracy', self.train_binary_accuracy.result() * 100, step=step)
                tf.summary.scalar('train_precision', self.train_precision.result(), step=step)
                tf.summary.scalar('train_recall', self.train_recall.result(), step=step)
                if self.model_type == 'regression' or self.model_type == 'multi_classification':
                    tf.summary.scalar('train_multi_accuracy', self.train_multi_accuracy.result() * 100, step=step)

            # check if the model should be validated
            if int(step) % self.log_interval == 0:
                # reset validation loss and metrics
                self.val_loss.reset_states()
                self.val_binary_accuracy.reset_states()
                self.val_precision.reset_states()
                self.val_recall.reset_states()
                if self.model_type == 'regression' or self.model_type == 'multi_classification':
                    self.val_multi_accuracy.reset_states()

                # perform one-step validation
                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                # log the training and validation information
                if self.model_type == 'binary_classification':
                    template = 'Step {} [Training/Validation]: Loss: {:.5f}/{:.5f}, Binary-accuracy: {:.2f}/{:.2f}, Precision: {:.5f}/{:.5f}, Recall: {:.5f}/{:.5f}'
                    logging.info(template.format(step,
                                                 self.train_loss.result(), self.val_loss.result(),
                                                 self.train_binary_accuracy.result() * 100, self.val_binary_accuracy.result() * 100,
                                                 self.train_precision.result(), self.val_precision.result(),
                                                 self.train_recall.result(), self.val_recall.result()))
                    # record the accuracy and loss of this step, which is used for early stopping
                    accuracy_record = self.val_binary_accuracy.result().numpy()
                    loss_record = self.val_loss.result().numpy()
                elif self.model_type == 'regression' or self.model_type == 'multi_classification':
                    template = 'Step {} [Training/Validation]: Loss: {:.5f}/{:.5f}, Binary-accuracy: {:.2f}/{:.2f}, Precision: {:.5f}/{:.5f}, Recall: {:.5f}/{:.5f}, Multi-accuracy: {:.2f}/{:.2f}'
                    logging.info(template.format(step,
                                                 self.train_loss.result(), self.val_loss.result(),
                                                 self.train_binary_accuracy.result() * 100, self.val_binary_accuracy.result() * 100,
                                                 self.train_precision.result(), self.val_precision.result(),
                                                 self.train_recall.result(), self.val_recall.result(),
                                                 self.train_multi_accuracy.result() * 100, self.val_multi_accuracy.result() * 100))
                    # record the accuracy and loss of this step, which is used for early stopping
                    accuracy_record = self.val_multi_accuracy.result().numpy()
                    loss_record = self.val_loss.result().numpy()

                # reset train loss and metrics
                self.train_loss.reset_states()
                self.train_binary_accuracy.reset_states()
                self.train_precision.reset_states()
                self.train_recall.reset_states()
                if self.model_type == 'regression' or self.model_type == 'multi_classification':
                    self.train_multi_accuracy.reset_states()

                # write validation summary to tensorboard
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('val_binary_accuracy', self.val_binary_accuracy.result() * 100, step=step)
                    tf.summary.scalar('val_precision', self.val_precision.result(), step=step)
                    tf.summary.scalar('val_recall', self.val_recall.result(), step=step)
                    if self.model_type == 'regression' or self.model_type == 'multi_classification':
                        tf.summary.scalar('val_multi_accuracy', self.val_multi_accuracy.result() * 100, step=step)

                yield self.val_binary_accuracy.result().numpy()

            # run deep visualization
            if int(step) % self.visual_interval == 0:
                deep_visualize(self.model, images[:1], labels[:1], step, self.model_type, self.run_paths)

            # save checkpoints
            if int(self.ckpt.step) % self.ckpt_interval == 0:
                # only save the model parameters corresponding to the current optimal result
                if accuracy_record > max_accuracy_record:
                    max_accuracy_record = accuracy_record
                    min_loss_record = loss_record
                    # save checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Saved checkpoint for step {int(self.ckpt.step)} to {save_path}.')
                elif accuracy_record == max_accuracy_record and loss_record < min_loss_record:
                    max_accuracy_record = accuracy_record
                    min_loss_record = loss_record
                    # save checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Saved checkpoint for step {int(self.ckpt.step)} to {save_path}.')
                else:
                    logging.info(f'Did not save checkpoint for step {int(self.ckpt.step)}, because the validation accuracy was not high enough.')

            # finish
            if int(step) % self.total_steps == 0:
                # only save the model parameters corresponding to the current optimal result
                if accuracy_record > max_accuracy_record:
                    # save final checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps and saved final checkpoint to {save_path}.')
                elif accuracy_record == max_accuracy_record and loss_record < min_loss_record:
                    # save final checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps and saved final checkpoint to {save_path}.')
                else:
                    logging.info(
                        f'Finished training after {step} steps, but did not save checkpoint for step {int(self.ckpt.step)}, because the validation accuracy was not high enough.')

                # stop the profiler (optional)
                # tf.profiler.experimental.stop()

                return self.val_binary_accuracy.result().numpy()

            self.ckpt.step.assign_add(1)

    def model_output(self):
        """model output interface (used for fine tuning)"""

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            tf.print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            tf.print("Initializing from scratch.")
        return self.model
