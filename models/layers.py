import gin
import tensorflow as tf


@gin.configurable
def basic_conv2D(inputs, filters, kernel_size, strides=1, padding='same', use_bn=True, use_activation=True):
    """A single convolutional layer

    Parameters:
        inputs (Tensor): input of the convolutional layer
        filters (int): number of filters used for the convolutional layer
        kernel_size (tuple: 2): kernel size used for the convolutional layer, e.g. (3, 3)
        strides (int): stride of the convolutional layer (Default is 1)
        padding (string): padding type of the convolutional layer (Default is 'same')
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single convolutional layer
    """

    outputs = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='linear')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.ReLU()(outputs)
    return outputs


@gin.configurable
def rep_conv2D(inputs, filters, strides, padding='same', use_1x1=True, use_identity=True):
    """A single convolutional layer of RepVGG

    Parameters:
        inputs (Tensor): input of the convolutional layer
        filters (int): number of filters used for the convolutional layer
        strides (int): stride of the convolutional layer
        padding (string): padding type of the convolutional layer (Default is 'same')
        use_1x1 (bool): whether the 1x1 convolutional branch is used or not (Default is True)
        use_identity (bool): whether the identity branch is used or not (Default is True)

    Returns:
        (Tensor): output of the single convolutional layer of RepVGG
    """

    outputs = basic_conv2D(inputs, filters=filters, kernel_size=(3, 3), strides=strides, padding=padding)
    if use_1x1:
        outputs_1x1 = basic_conv2D(inputs, filters=filters, kernel_size=(1, 1), strides=strides, padding=padding)
        outputs += outputs_1x1
    if use_identity:
        outputs_identity = tf.keras.layers.BatchNormalization()(inputs)
        outputs += outputs_identity
    return outputs


@gin.configurable
def split_conv2D(inputs, filters, base_kernel_number, strides, padding='same'):
    """A split convolutional layer

    Parameters:
        inputs (Tensor): input of the split convolutional layer
        filters (int): number of filters used for the split convolutional layer
        base_kernel_number (int): number of base kernels for the split convolutional layer
        strides (int): stride of the split convolutional layer
        padding (string): padding type of the split convolutional layer (Default is 'same')

    Returns:
        (Tensor): output of the split convolutional layer
    """

    outputs = basic_conv2D(inputs, filters=filters, kernel_size=(1, 1), strides=1, padding='same')
    for k in range(base_kernel_number):
        outputs = basic_conv2D(outputs, filters=filters, kernel_size=(1, 3), strides=strides, padding=padding)
        outputs = basic_conv2D(outputs, filters=filters, kernel_size=(3, 1), strides=strides, padding=padding)
    return outputs


@gin.configurable
def group_conv2D(inputs, filters, base_kernel_number, strides, padding='same', branch_filters=4, cardinality=32):
    """A group convolutional layer

    Parameters:
        inputs (Tensor): input of the group convolutional layer
        filters (int): number of filters used for the group convolutional layer
        base_kernel_number (int): number of base kernels for the group convolutional layer
        strides (int): stride of the group convolutional layer
        padding (string): padding type of the group convolutional layer (Default is 'same')
        branch_filters (int): number of  branch filters of the group convolutional layer (Default is 4)
        cardinality (int): cardinality of the group convolutional layer (Default is 32)

    Returns:
        (Tensor): output of the group convolutional layer
    """

    split = []
    for i in range(cardinality):
        split_conv = split_conv2D(inputs, branch_filters, base_kernel_number, strides=strides, padding=padding)
        split.append(split_conv)
    outputs = tf.keras.layers.concatenate(split, axis=-1)
    outputs = basic_conv2D(outputs, filters=filters, kernel_size=(1, 1), strides=1, padding='same')
    return outputs


@gin.configurable
def se(inputs, channels, se_rate=16):
    """A Squeeze-and-Excitation layer

    Parameters:
        inputs (Tensor): input of the Squeeze-and-Excitation layer
        channels (int): number of  channels in the Squeeze-and-Excitation layer
        se_rate (int): reduction factor of channels in the Squeeze-and-Excitation layer (Default is 16)

    Returns:
        (Tensor): output of the Squeeze-and-Excitation layer
    """

    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(units=(channels // se_rate), activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(units=channels, activation='sigmoid')(outputs)
    outputs = tf.keras.layers.Reshape((1, 1, channels))(outputs)
    return inputs * outputs


@gin.configurable
def vgg_block(inputs, filters, kernel_size, n_layer=2):
    """A single VGG block consisting of several convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
        n_layer (int): number of convolutional layers

    Returns:
        (Tensor): output of the VGG block
    """

    assert n_layer > 0, 'Number of convolutional layers has to be at least 1.'

    outputs = basic_conv2D(inputs, filters=filters, kernel_size=kernel_size, strides=1, padding='same')
    for i in range(n_layer - 1):
        outputs = basic_conv2D(outputs, filters=filters, kernel_size=kernel_size, strides=1, padding='same')
    outputs = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(outputs)

    return outputs


@gin.configurable
def rep_vgg_block(inputs, filters, n_layer):
    """A single RepVGG block consisting of several convolutional layers

    Parameters:
        inputs (Tensor): input of the RepVGG block
        filters (int): number of filters used for the convolutional layers
        n_layer (int): number of convolutional layers

    Returns:
        (Tensor): output of the RepVGG block
    """

    outputs = rep_conv2D(inputs, filters, strides=2, padding='same', use_1x1=True, use_identity=False)
    for i in range(n_layer - 1):
        outputs = rep_conv2D(outputs, filters, strides=1, padding='same', use_1x1=True, use_identity=True)
    return outputs


@gin.configurable
def inception_block_pre_conv(inputs):
    """A pre-convolutional block of Inception consisting of several convolutional layers and max-pooling layers

    Parameters:
        inputs (Tensor): input of the Inception pre-convolutional block

    Returns:
        (Tensor): output of the Inception pre-convolutional block
    """

    outputs = basic_conv2D(inputs, filters=32, kernel_size=(3, 3), strides=2, padding='same')
    outputs = basic_conv2D(outputs, filters=32, kernel_size=(3, 3), strides=1, padding='same')
    outputs = basic_conv2D(outputs, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    outputs = basic_conv2D(outputs, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    outputs = basic_conv2D(outputs, filters=80, kernel_size=(1, 1), strides=1, padding='same')
    outputs = basic_conv2D(outputs, filters=192, kernel_size=(3, 3), strides=1, padding='same')
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)

    return outputs


@gin.configurable
def inception_block(inputs, filters_number, base_kernel_number):
    """A convolutional block of Inception with four branches consisting of several convolutional layers

    Parameters:
        inputs (Tensor): input of the Inception convolutional block
        filters_number (int): basic number of filters used for the Inception convolutional block
        base_kernel_number (int): number of base kernels for the Inception convolutional block

    Returns:
        (Tensor): output of the Inception convolutional block
    """

    # branch 1 with perception field small
    b1 = basic_conv2D(inputs, filters=filters_number, kernel_size=(1, 1), strides=1, padding='same')

    # branch 2 with perception field medium
    b2 = basic_conv2D(inputs, filters=filters_number, kernel_size=(1, 1), strides=1, padding='same')
    for i in range(base_kernel_number[0]):
        b2 = basic_conv2D(b2, filters=filters_number, kernel_size=(3, 1), strides=1, padding='same')
        b2 = basic_conv2D(b2, filters=filters_number, kernel_size=(1, 3), strides=1, padding='same')
    b2 = basic_conv2D(b2, filters=filters_number, kernel_size=(1, 1), strides=1, padding='same')

    # branch 3 with perception field large
    b3 = basic_conv2D(inputs, filters=filters_number, kernel_size=(1, 1), strides=1, padding='same')
    for i in range(base_kernel_number[1]):
        b3 = basic_conv2D(b3, filters=filters_number, kernel_size=(3, 1), strides=1, padding='same')
        b3 = basic_conv2D(b3, filters=filters_number, kernel_size=(1, 3), strides=1, padding='same')
    b3 = basic_conv2D(b3, filters=filters_number * 2, kernel_size=(1, 1), strides=1, padding='same')

    # branch 4 with average pooling layer
    b4 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    b4 = basic_conv2D(b4, filters=filters_number, kernel_size=(1, 1), strides=1, padding='same')

    outputs = tf.keras.layers.concatenate([b1, b2, b3, b4], axis=-1)

    return outputs


@gin.configurable
def seresnext_block_pre_conv(inputs):
    """A pre-convolutional block of SEResNeXt consisting of several convolutional layers and max-pooling layers

    Parameters:
        inputs (Tensor): input of the SEResNeXt pre-convolutional block

    Returns:
        (Tensor): output of the SEResNeXt pre-convolutional block
    """

    outputs = basic_conv2D(inputs, filters=32, kernel_size=(3, 3), strides=2, padding='same')
    outputs = basic_conv2D(outputs, filters=32, kernel_size=(3, 3), strides=1, padding='same')
    outputs = basic_conv2D(outputs, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    outputs = basic_conv2D(outputs, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)
    outputs = basic_conv2D(outputs, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    outputs = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(outputs)

    return outputs


@gin.configurable
def seresnext_block(inputs, filter_array, base_kernel_number, cardinality):
    """A convolutional block of SEResNeXt consisting of several convolutional layers

    Parameters:
        inputs (Tensor): input of the SEResNeXt convolutional block
        filter_array (tuple: 2): numbers of filters used for the SEResNeXt convolutional block
        base_kernel_number (int): number of base kernels for the SEResNeXt convolutional block
        cardinality (int): cardinality of the SEResNeXt convolutional block

    Returns:
        (Tensor): output of the SEResNeXt convolutional block
    """

    # the group convolution
    outputs = basic_conv2D(inputs, filters=filter_array[0], kernel_size=(1, 1), strides=1, padding='same')
    outputs = group_conv2D(outputs, filters=filter_array[1], base_kernel_number=base_kernel_number, strides=1, padding='same', cardinality=cardinality)
    # the Squeeze-and-Excitation branch in order to calculate the attention of different channels
    outputs = se(outputs, channels=filter_array[1])
    # the shortcut of identity mapping
    identity = tf.keras.layers.Conv2D(filters=filter_array[1], kernel_size=(1, 1), strides=1, padding='same', activation='linear')(inputs)
    # the non-linear activation function
    outputs = tf.keras.layers.ReLU()(outputs + identity)

    return outputs


@gin.configurable
def output_block(inputs, model_type, dense_units, dropout_rate):
    """A output block consisting of several dense layers and dropout layers

    Parameters:
        inputs (Tensor): input of the output block
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        dense_units (int): number of dense units for the output block
        dropout_rate (float): dropout rate of dropout layers in the output block

    Returns:
        (Tensor): output of the output block
    """

    # convert feature maps to feature tensor
    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # process the feature tensor through dense layers
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
    # for different model output types, choose different number of output layer units and different activation function
    if model_type == 'regression':
        outputs = tf.keras.layers.Dense(units=1, activation='relu', name='output')(outputs)
    elif model_type == 'binary_classification':
        outputs = tf.keras.layers.Dense(units=2, activation='softmax', name='output')(outputs)
    elif model_type == 'multi_classification':
        outputs = tf.keras.layers.Dense(units=5, activation='softmax', name='output')(outputs)
    else:
        outputs = tf.keras.layers.Dense(units=1, activation='relu', name='output')(outputs)

    return outputs
