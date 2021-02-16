import gin
import os
import cv2
import numpy as np
import tensorflow as tf


@gin.configurable
def deep_visualize(model, images, labels, step, model_type, run_paths, train=True):
    """perform the deep visualization

    Parameters:
        model (keras.Model): keras model object
        images (Tensor): image batch uesd for depth visualization
        labels (Tensor): label batch of the corresponding image batch
        step (int): number of corresponding model training steps when performing deep visualization
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        run_paths (dictionary): storage path of model information
        train (bool): whether the model is in the process of training or not (Default is True)
    """

    # set the save path of deep visualization results
    if train:
        deep_visualization_path = run_paths['path_deep_visualization']
        original_image_path = os.path.join(deep_visualization_path, 'original')
        grad_cam_path = os.path.join(deep_visualization_path, 'grad_cam')
        guided_backprop_path = os.path.join(deep_visualization_path, 'guided_backpropagation')
        guided_grad_cam_path = os.path.join(deep_visualization_path, 'guided_grad_cam')
        integrated_gradients_path = os.path.join(deep_visualization_path, 'integrated_gradients')
    else:
        deep_visualization_path = run_paths['path_deep_visualization']
        original_image_path = os.path.join(deep_visualization_path, 'original', 'eval')
        grad_cam_path = os.path.join(deep_visualization_path, 'grad_cam', 'eval')
        guided_backprop_path = os.path.join(deep_visualization_path, 'guided_backpropagation', 'eval')
        guided_grad_cam_path = os.path.join(deep_visualization_path, 'guided_grad_cam', 'eval')
        integrated_gradients_path = os.path.join(deep_visualization_path, 'integrated_gradients', 'eval')

    # generate original images from the dataset
    original_image = tf.cast(images[0], tf.uint8)
    original_image_label = labels[0].numpy()
    original_image = cv2.cvtColor(original_image.numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
    if not os.path.exists(original_image_path):
        os.makedirs(original_image_path)
    original_image_path = original_image_path + '/original_image_' + str(step) + '(label_' + str(original_image_label) + ').png'

    # generate the Grad-CAM images
    grad_cam_image, cam, prediction, grad_cam_path = grad_cam(model, images, step, model_type, grad_cam_path)

    # generate the guided backpropagation images
    guided_backprop_image, guided_backprop_path = guided_backpropagation(model, images, step, prediction, guided_backprop_path)

    # generate the guided Grad-CAM images
    guided_grad_cam_image, guided_grad_cam_path = guided_grad_cam(guided_backprop_image, cam, step, prediction, guided_grad_cam_path)

    # generate the integrated gradients images
    integrated_gradients_image, integrated_gradients_path = integrated_gradients(model, images, step, prediction, model_type, integrated_gradients_path)

    # save deep visualization images to files
    image_output(original_image, original_image_path, step)
    image_output(grad_cam_image, grad_cam_path, step, name='grad_cam')
    image_output(guided_backprop_image, guided_backprop_path, step, name='guided_backpropagation')
    image_output(guided_grad_cam_image, guided_grad_cam_path, step, name='guided_grad_cam')
    image_output(integrated_gradients_image, integrated_gradients_path, step, name='integrated_gradients')


@gin.configurable
def grad_cam(model, images, step, model_type, grad_cam_path):
    """generate the Grad-CAM images

    Parameters:
        model (keras.Model): keras model object
        images (Tensor): image batch uesd for depth visualization
        step (int): number of corresponding model training steps when performing deep visualization
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        grad_cam_path (string): directory where Grad-CAM images are saved

    Returns:
        grad_cam_image (numpy array): Grad-CAM images
        cam (numpy array): CAM mask images
        idx (int): predicted labels of the corresponding images
        save_path (string): the path and file names of the Grad-CAM images
    """

    if not os.path.exists(grad_cam_path):
        os.makedirs(grad_cam_path)

    # extract the corresponding layer result from the model
    grad_cam_model = tf.keras.models.Model([model.inputs], [model.get_layer('last_conv').output, model.get_layer('output').output])

    # calculate the gradients of the predictions to the feature maps in the last convolutional layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_model(images, training=False)
        tape.watch(conv_outputs)
        if model_type == 'regression':
            idx = tf.squeeze(tf.cast(tf.clip_by_value(predictions[0], clip_value_min=0, clip_value_max=4), tf.int32))
            idx = idx.numpy()
            top_class = predictions[:, 0]
        elif model_type == 'binary_classification' or model_type == 'multi_classification':
            idx = np.argmax(predictions[0])
            top_class = predictions[:, idx]
    grads = tape.gradient(top_class, conv_outputs)

    # calculate the CAM
    weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    output = conv_outputs.numpy()[0]
    for i, w in enumerate(weights):
        output[:, :, i] *= w

    cam = np.mean(output, axis=-1)
    cam = cv2.resize(cam, (images.shape[1], images.shape[2]))

    # normalize the CAM
    cam = np.maximum(cam, 0) / (cam.max() + 1e-16)
    cam_image = cam * 255
    cam_image = np.clip(cam_image, 0, 255).astype('uint8')

    # generate the CAM and Grad-CAM images
    cam_image = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
    original_image = tf.cast(images[0], tf.uint8)
    grad_cam_image = cv2.addWeighted(cv2.cvtColor(original_image.numpy().astype('uint8'), cv2.COLOR_RGB2BGR), 1, cam_image, 0.3, 0)
    save_path = grad_cam_path + '/grad_cam_image_' + str(step) + '(prediction_' + str(idx) + ').png'

    return grad_cam_image, cam, idx, save_path


@gin.configurable
def guided_backpropagation(model, images, step, prediction, guided_backprop_path):
    """generate the guided backpropagation images

    Parameters:
        model (keras.Model): keras model object
        images (Tensor): image batch uesd for depth visualization
        step (int): number of corresponding model training steps when performing deep visualization
        prediction (int): predicted labels of the corresponding images
        guided_backprop_path (string): directory where guided backpropagation images are saved

    Returns:
        guided_backprop_image (numpy array): guided backpropagation images
        save_path (string): the path and file names of the guided backpropagation images
    """

    if not os.path.exists(guided_backprop_path):
        os.makedirs(guided_backprop_path)

    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy

        return tf.nn.relu(x), grad

    # extract the corresponding layer result from the model
    guided_backprop_model = tf.keras.models.Model([model.inputs], [model.get_layer('last_conv').output])
    layer_dict = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer, 'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

    # calculate the gradients
    with tf.GradientTape() as tape:
        inputs = images
        tape.watch(inputs)
        outputs = guided_backprop_model(images, training=False)
    grads = tape.gradient(outputs, inputs)[0]
    guided_backprop = grads.numpy()

    # normalize the guided backpropagation images
    guided_backprop = (guided_backprop - guided_backprop.mean()) / (guided_backprop.std() + 1e-16)
    guided_backprop = (guided_backprop * 0.25 + 0.5) * 255

    # generate the guided backpropagation images
    guided_backprop_image = np.clip(guided_backprop, 0, 255).astype('uint8')
    save_path = guided_backprop_path + '/guided_backprop_image_' + str(step) + '(prediction_' + str(prediction) + ').png'

    return guided_backprop_image, save_path


@gin.configurable
def guided_grad_cam(guided_backprop_image, cam, step, prediction, guided_grad_cam_path):
    """generate the guided Grad-CAM images

    Parameters:
        guided_backprop_image (numpy array): guided backpropagation images
        cam (numpy array): CAM mask images
        step (int): number of corresponding model training steps when performing deep visualization
        prediction (int): predicted labels of the corresponding images
        guided_grad_cam_path (string): directory where guided Grad-CAM images are saved

    Returns:
        guided_grad_cam_image (numpy array): guided Grad-CAM images
        save_path (string): the path and file names of the guided Grad-CAM images
    """

    if not os.path.exists(guided_grad_cam_path):
        os.makedirs(guided_grad_cam_path)

    # combine the Grad-CAM and guided backpropagation, then generate the guided Grad-CAM images
    cam = np.expand_dims(cam, axis=-1)
    guided_grad_cam_image = (guided_backprop_image * cam).astype('uint8')
    save_path = guided_grad_cam_path + '/guided_grad_cam_image_' + str(step) + '(prediction_' + str(prediction) + ').png'

    return guided_grad_cam_image, save_path


@gin.configurable
def integrated_gradients(model, images, step, prediction, model_type, integrated_gradients_path, m_steps=50, batch_size=32):
    """generate the integrated gradients images

    Parameters:
        model (keras.Model): keras model object
        images (Tensor): image batch uesd for depth visualization
        step (int): number of corresponding model training steps when performing deep visualization
        prediction (int): predicted labels of the corresponding images
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        integrated_gradients_path (string): directory where integrated gradients images are saved
        m_steps (int): degree of integral discretization
        batch_size (int): batch size when calculating gradients

    Returns:
        integrated_gradients_image (numpy array): integrated gradients images
        save_path (string): the path and file names of the integrated gradients images
    """

    if not os.path.exists(integrated_gradients_path):
        os.makedirs(integrated_gradients_path)

    # creat the baseline image and the gradually varied image group
    baseline = tf.zeros(shape=(images.shape[1], images.shape[2], images.shape[3]))

    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    def interpolate_images(base_image, image, alpha_set):
        alpha_set_x = alpha_set[:, tf.newaxis, tf.newaxis, tf.newaxis]
        base_image_x = tf.expand_dims(base_image, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - base_image_x
        image_set = base_image_x + alpha_set_x * delta
        return image_set

    def compute_gradients(image_set, target_class_idx):
        with tf.GradientTape() as tape:
            tape.watch(image_set)
            predictions = model(image_set)
            if model_type == 'regression':
                probs = predictions[:, 0]
            elif model_type == 'binary_classification' or model_type == 'multi_classification':
                probs = predictions[:, target_class_idx]
        return tape.gradient(probs, image_set)

    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # generate images of different intensities through linear interpolation
        interpolated_path_input_batch = interpolate_images(base_image=baseline, image=images[0], alpha_set=alpha_batch)

        # calculate the gradients
        gradient_batch = compute_gradients(image_set=interpolated_path_input_batch, target_class_idx=prediction)

        # batch the gradients of different images together
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    total_gradients = gradient_batches.stack()

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_grads = tf.math.reduce_mean(grads, axis=0)
        return integrated_grads

    def normalization(gradients):
        gradients = (gradients - tf.reduce_mean(gradients)) / (tf.math.reduce_std(gradients) + 1e-16)
        return gradients

    avg_gradients = integral_approximation(gradients=total_gradients)

    # normalize the average gradients
    avg_gradients = normalization(avg_gradients)

    # generate the attribution mask
    attributions = (images[0] - baseline) * avg_gradients
    attribution_mask = tf.reduce_mean(tf.math.abs(attributions), axis=-1).numpy()
    attribution_mask = np.clip(attribution_mask, 0, 255).astype('uint8')

    # generate the integrated gradients images
    attribution_mask = cv2.applyColorMap(attribution_mask, cv2.COLORMAP_INFERNO)
    original_image = tf.cast(images[0], tf.uint8)
    integrated_gradients_image = cv2.addWeighted(cv2.cvtColor(original_image.numpy().astype('uint8'), cv2.COLOR_RGB2BGR), 0.8, attribution_mask, 0.6, 0)
    save_path = integrated_gradients_path + '/integrated_gradients_' + str(step) + '(prediction_' + str(prediction) + ').png'

    return integrated_gradients_image, save_path


def image_output(image, path, step, name='original'):
    """save deep visualization images to files

    Parameters:
        image (Tensor): deep visualization image
        path (string): the path and file name of the deep visualization image
        step (int): number of corresponding model training steps when performing deep visualization
        name (string): name of deep visualization method
    """

    # scale the picture to its original height-width-ratio
    image = cv2.resize(image, (308, 256))
    if cv2.imwrite(path, image):
        tf.print('saved ' + name + '_image for step {}: {}'.format(int(step), path))
    else:
        tf.print(name + '_image for step {} was not successfully saved'.format(int(step)))
