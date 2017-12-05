from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import net_train
from biwi_1_data import Biwi_1_Data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/data/huozengwei/tfrecord/aflw_1/',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_string('train_dir', '/data/huozengwei/train_dir/aflw_2016/aflw_1/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('fold', 1,
                            """cross valdation.""")
def main(_):
  dataset = Biwi_1_Data(subset='train')
  assert dataset.data_files()
  net_train.train(dataset)

if __name__ == '__main__':
  tf.app.run()
