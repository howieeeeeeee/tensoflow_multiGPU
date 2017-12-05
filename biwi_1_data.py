from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dataset import Dataset


class Biwi_1_Data(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(Biwi_1_Data, self).__init__('Biwi_1', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 5

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      if FLAGS.fold == 1:
        return 19508
      if FLAGS.fold == 2:
        return 19507
      if FLAGS.fold == 3:
        return 19507
      if FLAGS.fold == 4:
        return 19507
      if FLAGS.fold == 5:
        return 19507
    if self.subset == 'validation':
      if FLAGS.fold == 1:
        return 4876
      if FLAGS.fold == 2:
        return 4877
      if FLAGS.fold == 3:    
        return 4877
      if FLAGS.fold == 4:
        return 4877
      if FLAGS.fold == 5:
        return 4877

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any Flowers %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('Please see README.md for instructions on how to build '
          'the flowers dataset using download_and_preprocess_flowers.\n')
