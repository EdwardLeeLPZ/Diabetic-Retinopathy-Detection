from models.layers import *


@gin.configurable
def densenet201(input_shape, trainable_rate, model_type='regression'):
    """Defines a DenseNet201 architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        trainable_rate (float): proportion of trainable parameters in the feature extraction module
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """
    # set the input
    inputs = tf.keras.Input(input_shape)
    # preprocess input data
    prep_inputs = tf.keras.applications.densenet.preprocess_input(inputs)

    # build the DenseNet201 model with transfer learning
    base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    # Fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))
    # Freeze all the layers before the 'fine_tune_at' layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    outputs = base_model(prep_inputs)
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='densenet201')


@gin.configurable
def efficientnetb3(input_shape, trainable_rate, model_type='regression'):
    """Defines a DenseNet201 architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        trainable_rate (float): proportion of trainable parameters in the feature extraction module
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)
    # preprocess input data
    prep_inputs = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # build the EfficientNetB3 model with transfer learning
    base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    # Fine-tune from this layer onwards
    fine_tune_at = int(len(base_model.layers) * (1 - trainable_rate))
    # Freeze all the layers before the 'fine_tune_at' layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    outputs = base_model(prep_inputs)
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='efficientnetb3')
