from functions import *
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt

filename = "prepared_images/files.txt"
batch_size = 22

# Reads pfathes of images together with their labels
image_list, label_list = read_labeled_image_list(filename)

images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],shuffle=True)

image, label = read_images_from_disk(input_queue)

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
image = preprocess_image(image)
#label = preprocess_label(label)
label = tf.one_hot(label, 6)
image = tf.reshape(image, [-1])

# Optional Image and Label Batching
image_batch, label_batch = tf.train.shuffle_batch([image, label],batch_size=batch_size, capacity=100000, min_after_dequeue=2)

tensor_name = image.op.name
image_summary_t = tf.image_summary(tensor_name + 'images', image_batch,max_images=22)
print(tensor_name + 'images')

#x = tf.placeholder(tf.float32, [None, 50176])

x = image_batch
y_ = label_batch

W = tf.Variable(tf.zeros([150528, 6]))
b = tf.Variable(tf.zeros([6]))
#W = tf.Print(W, [W], "W: ")
#x = tf.Print(x, [x], "x: ")
#b = tf.Print(b, [b], "b: ")
y = tf.nn.softmax(tf.matmul(x, W) + b)


#y_ = tf.Print(y_, [y_], "y_: ", summarize=20)
#y = tf.Print(y, [y], "y: ", summarize=20)
log_calc = tf.log(tf.clip_by_value(y,1e-10,1.0));
#log_calc = tf.Print(log_calc, [log_calc], "log_calc: ", summarize=20)
cross_sum = tf.reduce_sum(y_ * log_calc, reduction_indices=[1])
#cross_sum = tf.Print(cross_sum, [cross_sum], "cross_sum: ", summarize=20)
cross_entropy = tf.reduce_mean(-cross_sum)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init_op = tf.initialize_all_variables()
sess = tf.Session()
with sess.as_default():
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print(label_batch.eval())

    #summary_op = tf.merge_all_summaries()
    #writer = tf.train.SummaryWriter('/tmp/basic')
    #summary_str = sess.run(summary_op)
    #writer.add_summary(summary_str, 0)

    for i in range(500):
        sess.run(train_step)
        print(i)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))

    coord.request_stop()
    coord.join(threads)

    print("Finish Test")

    sess.close()