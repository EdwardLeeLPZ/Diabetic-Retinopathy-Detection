import cv2
import gin
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """online dataset preprocessing: Resizing"""
    # change the type of image data: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32)
    # resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    return image, label


@gin.configurable
def graham_preprocessing(image, scale=144):
    """offline dataset preprocessing: graham preprocessing"""
    # filter (high-pass) the image
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
    # remove boundary shadows
    image_mask = np.zeros(image.shape, dtype=np.int32)
    cv2.circle(image_mask, (int(image.shape[0] * 0.5), int(image.shape[1] * 0.5)), int(scale * 0.975), (1, 1, 1), -1, 8, 0)
    image = image * image_mask + 128 * (1 - image_mask)
    return image


@gin.configurable
def augment(image, label, random_rotate_rate, shear_level, random_contrast_boundary, random_saturation_boundary, max_delta_hue):
    """online data augmentation"""
    # randomly rotate the image
    image = tfa.image.rotate(image, tf.random.uniform([], minval=-(np.pi * random_rotate_rate), maxval=(np.pi * random_rotate_rate), dtype=tf.float32))
    # randomly shear the image
    shear_x = tf.random.uniform([], minval=-shear_level[0], maxval=shear_level[0], dtype=tf.float32)
    shear_y = tf.random.uniform([], minval=-shear_level[1], maxval=shear_level[1], dtype=tf.float32)
    image = tfa.image.transform(image, [1.0, shear_x, 0.0, shear_y, 1.0, 0.0, 0.0, 0.0])
    # randomly crop the image
    image = tf.image.random_crop(image, [256, 256, 3])
    # randomly flip the image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # randomly change contrast of the image
    image = tf.image.random_contrast(image, lower=random_contrast_boundary[0], upper=random_contrast_boundary[1])
    # randomly change saturation of the image
    image = tf.image.random_saturation(image, lower=random_saturation_boundary[0], upper=random_saturation_boundary[1])
    # randomly change hue of the image
    image = tf.image.random_hue(image, max_delta=max_delta_hue)
    return image, label


@gin.configurable
def crop_black_border(image, reduction_rate, threshold, graham):
    """offline dataset preprocessing: black border cropping"""
    # locate the image border
    img = cv2.resize(image, (int(image.shape[1] / reduction_rate), int(image.shape[0] / reduction_rate)))
    img = cv2.medianBlur(img, 5)

    # crop the image border
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image = image[(8 * y):(8 * (y + h)), (8 * x):(8 * (x + w))]
    cropped_image = cv2.resize(cropped_image, (288, 288))

    # graham preprocess the image
    if graham:
        cropped_image = graham_preprocessing(cropped_image, 144)

    # encode the image as a string
    success, encoded_image = cv2.imencode('.jpg', cropped_image)
    return encoded_image


@gin.configurable
def over_sampling(label, distribution, over_sample_rate):
    """online dataset balancing: oversampling"""
    if label == 0:
        return tf.cast(distribution[0] * over_sample_rate, tf.int64)
    elif label == 1:
        return tf.cast(distribution[1] * over_sample_rate, tf.int64)
    elif label == 2:
        return tf.cast(distribution[2] * over_sample_rate, tf.int64)
    elif label == 3:
        return tf.cast(distribution[3] * over_sample_rate, tf.int64)
    else:
        return tf.cast(distribution[4] * over_sample_rate, tf.int64)
