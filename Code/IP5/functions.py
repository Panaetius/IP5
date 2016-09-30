import tensorflow as tf

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example = tf.image.convert_image_dtype(example, dtype=tf.float32)

    bbox_begin = (0, 0, 0)
    bbox_size = tf.constant((224, 224, 3), dtype=tf.int32)

    example = tf.slice(example, bbox_begin, bbox_size)

    return example, label

"""Distorts and changes images to produce more data"""
def preprocess_image(image):
    distorted_image = tf.image.random_flip_left_right(image)

    distorted_image = tf.image.random_flip_up_down(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=0.5)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.9, upper=1.1)
    distorted_image = tf.image.random_hue(distorted_image,
                                          max_delta=0.1)

    distorted_image = tf.image.random_saturation(distorted_image,
                                                 lower=0.9,
                                                 upper=1.1)
    return distorted_image