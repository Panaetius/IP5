"""A binary to train ip5wke using a single GPU.

Accuracy:
ip5wke_train.py achieves ~94% accuracy after 100K steps (256 epochs of
data) as judged by ip5wke_eval.py.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.models.image.ip5wke import ip5wke
import ip5wke

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/ip5wke_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('dropout_keep_probability', 0.5,
                          """How many nodes to keep during dropout""")

tf.app.flags.DEFINE_integer('batch_size', 22,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('is_training', True,
                            """Is training or not for batch norm""")


def train():
    """Train ip5wke for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        conf_matrix = tf.Variable(tf.zeros([ip5wke.NUM_CLASSES,
                                            ip5wke.NUM_CLASSES],
                                           tf.float32),
                                  name='conf_matrix',
                                  trainable=False)

        # Get images and labels for ip5wke.
        images, labels = ip5wke.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = ip5wke.inference(images)

        # Calculate loss.
        loss = ip5wke.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = ip5wke.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),
                               write_version=tf.train.SaverDef.V2)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        # Start running operations on the Graph.
        sess = tf.Session(config=config)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 25 == 0:
                # save summaries and print current loss/accuracy
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                correct_prediction = tf.equal(tf.argmax(logits, 1),
                                              tf.cast(labels, tf.int64))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))
                train_acc = sess.run(accuracy)
                tf.summary.scalar('accuracy', accuracy)

                format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, train_acc,
                                    examples_per_sec, sec_per_batch))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
