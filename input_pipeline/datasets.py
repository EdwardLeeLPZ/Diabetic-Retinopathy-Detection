import os
import logging
import pandas as pd

from input_pipeline.preprocessing import *


@gin.configurable
def load(name, data_dir, validation_rate, model_type, tfrecord_exist=False, graham=False, sample_pairing=False):
    """Loads data from files or TFRecord

    Parameters:
        name (string): name of the dataset (name list: 'idrid', 'eyepacs')
        data_dir (string): original path directory where the data is stored
        validation_rate (float): proportion of validation set to original data set
        model_type (string): output type of the model (type list: 'regression', 'binary_classification', 'multi_classification')
        tfrecord_exist (bool): whether TFRecord exists or not (Default is False)
        graham (bool): whether graham preprocessing is used or not (Default is False)
        sample_pairing (bool): whether sample pairing is used or not (Default is False)

    Returns:
        ds_train (tf.data.Dataset): training set
        ds_val (tf.data.Dataset): validation set
        ds_test (tf.data.Dataset): test set
        ds_info (dictionary): information and structure of dataset
    """

    # choose the dataset (IDRID or Kaggle Dataset provided by EyePACS)
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # check if there are already some existed TFRecord files. If not, then build the new TFRecord file
        if not tfrecord_exist:
            build_tfrecord(data_dir, graham, sample_pairing)

        # explain the structure of the dataset
        ds_info = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'img_height': tf.io.FixedLenFeature([], tf.int64),
            'img_width': tf.io.FixedLenFeature([], tf.int64),
            'img_depth': tf.io.FixedLenFeature([], tf.int64),
        }

        # parse data from bytes format into original image format
        def _preprocess(img_label_dict):
            """parse data"""
            temp = tf.io.parse_single_example(img_label_dict, ds_info)
            img_raw = tf.io.decode_jpeg(temp['image'], channels=3)
            image = tf.reshape(img_raw, [temp['img_height'], temp['img_width'], temp['img_depth']])

            label = temp['label']
            if model_type == 'binary_classification':
                label = tf.cast(label >= 2, dtype=tf.int32)

            return image, label

        # import the original training set from TFRecord files
        ds_before_split = tf.data.TFRecordDataset('./TFRecord/IDRID_train.tfrecord')
        ds_before_split = ds_before_split.map(_preprocess)

        # shuffle the original training set and divide it into training set and validation set
        num_before_split = sum(1 for _ in ds_before_split)
        num_train = int(num_before_split * (1 - validation_rate))
        ds_before_split = ds_before_split.shuffle(int(num_before_split * 0.5))

        # split dataset into train and validation
        ds_train = ds_before_split.take(num_train)
        ds_val = ds_before_split.skip(num_train)

        # oversample to balance the training data set
        ds_train = ds_train.flat_map(
            lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(over_sampling(label))
        )

        # import the test set from TFRecord files
        ds_test = tf.data.TFRecordDataset('./TFRecord/IDRID_test.tfrecord')
        ds_test = ds_test.map(_preprocess)

        # preprocess (and augment) the datasets before training
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")

        # explain the structure of the dataset
        ds_info = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'name': tf.io.FixedLenFeature([], tf.string),
        }

        # import training set, validation set and test set from TFRecord files
        train_files, val_files, test_files = get_eyepacs_tfrecord(data_dir)
        ds_train = tf.data.TFRecordDataset(train_files)
        ds_val = tf.data.TFRecordDataset(val_files)
        ds_test = tf.data.TFRecordDataset(test_files)

        # parse data from bytes format into original image format
        def _preprocess(img_label_dict):
            temp = tf.io.parse_single_example(img_label_dict, ds_info)
            img_raw = tf.io.decode_jpeg(temp['image'], channels=3)
            image = tf.image.resize(img_raw, (300, 300))
            image = tf.image.resize(image, size=(288, 288))
            label = temp['label']
            return image, label

        ds_train = ds_train.map(_preprocess)
        ds_val = ds_val.map(_preprocess)
        ds_test = ds_test.map(_preprocess)

        # oversample to balance the training data set
        ds_train = ds_train.flat_map(
            lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(over_sampling(label))
        )

        # preprocess (and augment) the datasets before training
        return prepare(ds_train, ds_val, ds_test, ds_info)
    else:
        raise ValueError


@gin.configurable
def build_tfrecord(data_dir, graham, sample_pairing):
    """build TFRecord file of IDRID dataset

    Parameters:
        data_dir (string): original path directory where the data is stored
        graham (bool): whether graham preprocessing is used or not (Default is False)
        sample_pairing (bool): whether sample pairing is used or not (Default is False)
    """

    def _bytes_feature(value):
        """returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def image_example(image_string, label):
        """change the image string into an example"""
        image_shape = tf.image.decode_jpeg(image_string).shape
        feature = {
            'image': _bytes_feature(image_string),
            'label': _int64_feature(label),
            'img_height': _int64_feature(image_shape[0]),
            'img_width': _int64_feature(image_shape[1]),
            'img_depth': _int64_feature(image_shape[2]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def read(image_dir, reduction_rate=8, threshold=20, graham=False, sample_pairing=False):
        """convert the image into a string file after processing"""
        if not sample_pairing:
            # load the original image
            image = cv2.imread(image_dir)
            # save the original image and resize it
            image = cv2.resize(image, (int(image.shape[1] / reduction_rate), int(image.shape[0] / reduction_rate)))
        else:
            # load the original image
            image1 = cv2.imread(image_dir[0])
            image2 = cv2.imread(image_dir[1])
            # save the original image and resize it
            image1 = cv2.resize(image1, (int(image1.shape[1] / reduction_rate), int(image1.shape[0] / reduction_rate)))
            image2 = cv2.resize(image2, (int(image2.shape[1] / reduction_rate), int(image2.shape[0] / reduction_rate)))
            # use sample pairing to mix these two images
            image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

        # crop the black border of the image
        encoded_image = crop_black_border(image, reduction_rate=reduction_rate, threshold=threshold, graham=graham)

        # change the image to a string
        image_string = encoded_image.tostring()

        return image_string

    # read  names and labels of images
    train_dict = pd.read_csv(data_dir + '/IDRID_dataset/labels/train.csv')
    ds_train_num = len(train_dict['Image name'])
    train_dict = {train_dict['Image name'][i]: train_dict['Retinopathy grade'][i] for i in range(ds_train_num)}
    train_list = sorted(train_dict.items(), key=lambda x: x[1], reverse=False)

    test_dict = pd.read_csv(data_dir + '/IDRID_dataset/labels/test.csv')
    ds_test_num = len(test_dict['Image name'])

    # determine whether the path used to store TFRecord exists
    if not os.path.exists('./TFRecord/'):
        os.makedirs('./TFRecord/')

    # build a TFRecord file of training set
    with tf.io.TFRecordWriter('./TFRecord/IDRID_train.tfrecord') as writer:
        # read in the original unmixed image
        for i in range(ds_train_num):
            img_name = train_list[i][0]
            img_label = train_list[i][1]
            img_string = read(data_dir + '/IDRID_dataset/images/train/' + img_name + '.jpg', graham=graham)
            tf_example = image_example(img_string, img_label)
            writer.write(tf_example.SerializeToString())

        # use sample pairing to mix images of training set to augment data
        if sample_pairing:
            count = 0
            for i in range(ds_train_num):
                if count > 3000:  # if the number of augmented images exceeds the threshold (3000), then stop augmentation
                    if train_list[i][1] != train_list[min(i + 1, ds_train_num - 1)][1]:
                        count = 0
                    continue
                # select two images with the same label to mix
                for j in range(i + 1, ds_train_num):
                    if train_list[i][1] != train_list[j][1]:
                        break
                    img1_name = train_list[i][0]
                    img2_name = train_list[j][0]
                    img_label = train_list[i][1]
                    img_string = read([data_dir + '/IDRID_dataset/images/train/' + img1_name + '.jpg', data_dir + '/IDRID_dataset/images/train/' + img2_name + '.jpg'],
                                      graham=graham, sample_pairing=sample_pairing)
                    tf_example = image_example(img_string, img_label)
                    writer.write(tf_example.SerializeToString())
                    count += 1

    # build a TFRecord file of test set
    with tf.io.TFRecordWriter('./TFRecord/IDRID_test.tfrecord') as writer:
        for i in range(ds_test_num):
            img_name = test_dict['Image name'][i]
            img_label = test_dict['Retinopathy grade'][i]
            img_string = read(data_dir + '/IDRID_dataset/images/test/' + img_name + '.jpg', graham=graham)
            tf_example = image_example(img_string, img_label)
            writer.write(tf_example.SerializeToString())


@gin.configurable
def get_eyepacs_tfrecord(data_dir):
    """Read in the name lists of TFRecord file of EyePACS dataset

    Parameters:
        data_dir (string): original path directory where the data is stored

    Returns:
        train_files (list): name list of TFRecord file of training set
        val_files (list): name list of TFRecord file of validation set
        test_files (list): name list of TFRecord file of test set
    """

    train_files = []
    val_files = []
    test_files = []
    for file in os.listdir(data_dir + '/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0/'):
        if file.find('train') != -1:
            train_files.append([data_dir + '/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        elif file.find('validation') != -1:
            val_files.append([data_dir + '/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        elif file.find('test') != -1:
            test_files.append([data_dir + '/tensorflow_datasets/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        else:
            continue
    return train_files, val_files, test_files


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, buffer_size, batch_size, caching):
    """prepares dataset: preprocessing and augmenting

    Parameters:
        ds_train (tf.data.Dataset): original training set
        ds_val (tf.data.Dataset): original validation set
        ds_test (tf.data.Dataset): original test set
        ds_info (dictionary): information and structure of dataset
        buffer_size (int):  size of the shuffle buffer
        batch_size (int): size of dataset batch
        caching (bool): whether cache is used or not

    Returns:
        ds_train (tf.data.Dataset): training set after preparation
        ds_val (tf.data.Dataset): validation set after preparation
        ds_test (tf.data.Dataset): test set after preparation
        ds_info (dictionary): information and structure of dataset
    """

    # prepare training dataset
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # use traditional online data augmentation method
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)  # AUTOTUNE will dynamically set the number of parallel calls based on the available CPU

    # prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)  # AUTOTUNE will dynamically set the number of parallel calls based on the available CPU

    # prepare test dataset
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)  # AUTOTUNE will dynamically set the number of parallel calls based on the available CPU

    return ds_train, ds_val, ds_test, ds_info
