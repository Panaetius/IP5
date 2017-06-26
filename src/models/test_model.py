"""Evaluation for ip5wke.

Accuracy:
ip5wke_train.py achieves 94% accuracy after 100K steps (256 epochs
of data) as judged by ip5wke_eval.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import ip5wke

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/ip5wke_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'validation',
                           """Either 'test' or 'validation'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/ip5wke_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 600,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 8000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_float('dropout_keep_probability', 1.0,
                          "How many nodes to keep during dropout")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('is_training', False,
                            """Is training or not for batch norm""")


def eval_once(saver, summary_writer, top_k_op, top_k_op2, conf_matrix_op,
              num_classes, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/ip5wke_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            
            # calculate accuracy, precision, recall and f1 score
            true_count = 0  # Counts the number of correct predictions.
            true_count2 = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            precisions = np.zeros(shape=(num_classes))
            recalls = np.zeros(shape=(num_classes))
            tp = np.zeros(shape=(num_classes))

            while step < num_iter and not coord.should_stop():
                predictions, predictions2, conf_matrix = sess.run(
                    [top_k_op, top_k_op2, conf_matrix_op])
                true_count += np.sum(predictions)
                true_count2 += np.sum(predictions2)
                precisions += conf_matrix.sum(axis=0)
                recalls += conf_matrix.sum(axis=1)
                tp += np.diagonal(conf_matrix)

                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            precision2 = true_count2 / total_sample_count
            print('%s: precision @ 1 = %.3f, @ 3 = %.3f' % (datetime.now(),
                                                            precision,
                                                            precision2))

            precs = np.divide(tp, precisions)
            recs = np.divide(tp, recalls)
            f1_scores = np.multiply(2.0, np.divide(np.multiply(precs, recs),
                                                   np.add(precs, recs)))

            print('precisions: ' + str(precs))
            print('recalls: ' + str(recs))
            print('f1: ' + str(f1_scores))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval ip5wke for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for ip5wke.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = ip5wke.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = ip5wke.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        top_k_op2 = tf.nn.in_top_k(logits, labels, 3)
        conf_matrix_op = tf.contrib.metrics.confusion_matrix(
            tf.argmax(logits, 1), labels,
            num_classes=ip5wke.NUM_CLASSES)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            ip5wke.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore,
                               write_version=tf.train.SaverDef.V2)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, top_k_op2,
                      conf_matrix_op, ip5wke.NUM_CLASSES, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    # ip5wke.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
