from models.layers import *


@gin.configurable
def vgg_like(input_shape, base_filters, blocks_array, model_type='regression'):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        base_filters (int): number of base filters, which are doubled for every VGG block
        blocks_array (tuple: 1 to infinite): number of different VGG blocks (with different convolutional layers)
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """

    assert max(blocks_array) > 0  # number of blocks has to be at least 1
    # set the input
    inputs = tf.keras.Input(input_shape)
    # normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(inputs)
    block_count = 0

    # establish feature extraction blocks
    for i in range(1, len(blocks_array) + 1):  # i means which vgg_block is used (with how many convolutional layers)
        for j in range(blocks_array[i - 1]):  # j means how many times this vgg_block is repeated.
            if block_count == 0:
                outputs = vgg_block(rescale, base_filters * 2 ** block_count, n_layer=i)
                block_count += 1
            else:
                outputs = vgg_block(outputs, base_filters * 2 ** block_count, n_layer=i)
                block_count += 1
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def rep_vgg(input_shape, rep_vgg_type='B0', base_filters=64, model_type='regression'):
    """Defines a RepVGG architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        rep_vgg_type (string): type of RepVGG model (Default is 'B0')
        base_filters (int): number of base filters, which are doubled for every VGG block (Default is 64)
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """

    rep_vgg_type_dict = {'A0': [(1, 2, 4, 14, 1), 0.75, 2.5],
                         'A1': [(1, 2, 4, 14, 1), 1, 2.5],
                         'A2': [(1, 2, 4, 14, 1), 1.5, 2.5],
                         'B0': [(1, 4, 6, 16, 1), 1, 2.5],
                         'B1': [(1, 4, 6, 16, 1), 2, 4],
                         'B2': [(1, 4, 6, 16, 1), 2.5, 5],
                         'B3': [(1, 4, 6, 16, 1), 3, 5]}
    # set parameters according to the selected model type
    blocks_array = rep_vgg_type_dict[rep_vgg_type][0]
    a = rep_vgg_type_dict[rep_vgg_type][1]
    b = rep_vgg_type_dict[rep_vgg_type][2]

    # set the input
    inputs = tf.keras.Input(input_shape)
    # normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(inputs)

    # establish feature extraction blocks
    outputs = rep_vgg_block(rescale, min(base_filters, a * base_filters), blocks_array[0])
    for i in range(1, len(blocks_array) - 1):
        outputs = rep_vgg_block(outputs, a * base_filters * 2 ** (i - 1), blocks_array[i])
    outputs = rep_vgg_block(outputs, b * base_filters * 8, blocks_array[-1])
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='rep_vgg')


@gin.configurable
def simplified_inception(input_shape, blockset_1_filter_array, blockset_2_filter_array, blockset_3_filter_array, model_type='regression'):
    """Defines a simplified Inception v3 architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        blockset_1_filter_array (tuple: 1 to infinite): base filter numbers of first Inception blocks
        blockset_2_filter_array (tuple: 1 to infinite): base filter numbers of second Inception blocks
        blockset_3_filter_array (tuple: 1 to infinite): base filter numbers of third Inception blocks
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)
    # normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(inputs)

    # establish feature pre-extraction blocks
    outputs = inception_block_pre_conv(rescale)
    # establish first feature extraction blocks
    for i in blockset_1_filter_array:
        outputs = inception_block(outputs, i, (1, 2))
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    # establish second feature extraction blocks
    for i in blockset_2_filter_array:
        outputs = inception_block(outputs, i, (2, 3))
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    # establish third feature extraction blocks
    for i in blockset_3_filter_array:
        outputs = inception_block(outputs, i, (3, 5))
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='simplified_inception')


@gin.configurable
def simplified_seresnext(input_shape, blockset_1_filter_array, blockset_1_number, blockset_2_filter_array, blockset_2_number, blockset_3_filter_array, blockset_3_number,
                         model_type='regression'):
    """Defines a simplified SEResNeXt architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        blockset_1_filter_array (tuple: 1 to infinite): base filter numbers of first SEResNeXt blocks
        blockset_1_number (int): number of first SEResNeXt blocks
        blockset_2_filter_array (tuple: 1 to infinite): base filter numbers of second SEResNeXt blocks
        blockset_2_number (int): number of second SEResNeXt blocks
        blockset_3_filter_array (tuple: 1 to infinite): base filter numbers of third SEResNeXt blocks
        blockset_3_number (int): number of third SEResNeXt blocks
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)
    # normalize input data
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)(inputs)

    # establish feature pre-extraction blocks
    outputs = seresnext_block_pre_conv(rescale)
    # establish first feature extraction blocks
    for i in range(blockset_1_number):
        outputs = seresnext_block(outputs, blockset_1_filter_array, base_kernel_number=1, cardinality=16)
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    # establish second feature extraction blocks
    for i in range(blockset_2_number):
        outputs = seresnext_block(outputs, blockset_2_filter_array, base_kernel_number=2, cardinality=16)
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    # establish third feature extraction blocks
    for i in range(blockset_3_number):
        outputs = seresnext_block(outputs, blockset_3_filter_array, base_kernel_number=4, cardinality=32)
    # establish an identity layer, which is not used for feature extraction, but only for data interface for deep visualization
    outputs = tf.keras.layers.Activation(activation='linear', name='last_conv')(outputs)

    # establish output block composed of dense layers
    outputs = output_block(outputs, model_type=model_type)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='simplified_seresnext')
