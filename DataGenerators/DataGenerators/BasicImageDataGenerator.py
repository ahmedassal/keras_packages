import warnings
from keras import backend as K
# from keras.preprocessing.image import DirectoryIterator
from keras.preprocessing.image import NumpyArrayIterator
from .BasicDirectoryIterator import BasicDirectoryIterator
import numpy as np
import scipy as sp

# REFACTOR into generic data generator based on task: class,regress, recog, in addition to data and annos shape
class BasicImageDataGenerator(object):
  """Generate minibatches of image data

    # Arguments
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

  def __init__(self,
               labels=None,
               labels_shape = None,
               color_mode='rgb',
               data_format=None):

    self.labels = labels
    self.labels_shape = labels_shape

    if data_format is None:
      data_format = K.image_data_format()

    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format should be "channels_last" (channel after row and '
                       'column) or "channels_first" (channel before row and column). '
                       'Received arg: ', data_format)
    self.data_format = data_format
    if data_format == 'channels_first':
      self.channel_axis = 1
      self.row_axis = 2
      self.col_axis = 3
    if data_format == 'channels_last':
      self.channel_axis = 3
      self.row_axis = 1
      self.col_axis = 2

    if color_mode not in {'rgb', 'grayscale'}:
      raise ValueError('Invalid color mode:', color_mode,
                       '; expected "rgb" or "grayscale".')
    self.color_mode = color_mode


  def flow_from_directory(self, directory,
                          target_size=(256, 256), color_mode=None,
                          # classes=None, class_mode='categorical',
                          labels = None,
                          labels_shape = None,
                          batch_size=32, shuffle=True, seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='jpeg',
                          follow_links=False):
    # if self.preprocessing_function is not None:
    #   self.preprocessing_function(self.new_size)
    if color_mode == None:
      color_mode = self.color_mode

    if labels is None:
      labels = self.labels

    if labels_shape is None:
      labels_shape = self.labels_shape

    return BasicDirectoryIterator(
      directory, self,
      target_size=target_size, color_mode=color_mode,
      # classes=classes, class_mode=class_mode,
      labels= labels,
      labels_shape=labels_shape,
      data_format=self.data_format,
      batch_size=batch_size, shuffle=shuffle, seed=seed,
      save_to_dir=save_to_dir,
      save_prefix=save_prefix,
      save_format=save_format,
      follow_links=follow_links)
