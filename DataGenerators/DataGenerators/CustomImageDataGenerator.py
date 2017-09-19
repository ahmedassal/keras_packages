import warnings
from keras import backend as K
from keras.preprocessing.image import DirectoryIterator
from keras.preprocessing.image import NumpyArrayIterator
import numpy as np
import scipy as sp

class CustomImageDataGenerator(object):
  """Generate minibatches of image data with real-time data augmentation.

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
               new_size=None,
               download_function = None,
               preprocessing_function=None,
               data_format=None):

    if new_size is not None:
      self.new_size = new_size
      # raise ValueError('Invalid color mode:', new_size,
      #                  '; expected "rgb" or "grayscale".')

    if data_format is None:
      data_format = K.image_data_format()

    if download_function is not None:
      self.download_function = download_function

    if preprocessing_function is not None:
      self.preprocessing_function = preprocessing_function

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

    if download_function is not None:
      self.train_path, self.test_path = self.download_function()


  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
           save_to_dir=None, save_prefix='', save_format='jpeg'):
    return NumpyArrayIterator(
      x, y, self,
      batch_size=batch_size,
      shuffle=shuffle,
      seed=seed,
      data_format=self.data_format,
      save_to_dir=save_to_dir,
      save_prefix=save_prefix,
      save_format=save_format)

  def flow_from_directory(self, directory,
                          target_size=(256, 256), color_mode='rgb',
                          classes=None, class_mode='categorical',
                          batch_size=32, shuffle=True, seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='jpeg',
                          follow_links=False):
    # if self.preprocessing_function is not None:
    #   self.preprocessing_function(self.new_size)
    return DirectoryIterator(
      directory, self,
      target_size=target_size, color_mode=color_mode,
      classes=classes, class_mode=class_mode,
      data_format=self.data_format,
      batch_size=batch_size, shuffle=shuffle, seed=seed,
      save_to_dir=save_to_dir,
      save_prefix=save_prefix,
      save_format=save_format,
      follow_links=follow_links)

  def standardize(self, x):
    """Apply the normalization configuration to a batch of inputs.

    # Arguments
        x: batch of inputs to be normalized.

    # Returns
        The inputs, normalized.
    """
    if self.preprocessing_function:
      x = self.preprocessing_function(x)
    if self.rescale:
      x *= self.rescale
    # x is a single image, so it doesn't have image number at index 0
    img_channel_axis = self.channel_axis - 1
    if self.samplewise_center:
      x -= np.mean(x, axis=img_channel_axis, keepdims=True)
    if self.samplewise_std_normalization:
      x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

    if self.featurewise_center:
      if self.mean is not None:
        x -= self.mean
      else:
        warnings.warn('This ImageDataGenerator specifies '
                      '`featurewise_center`, but it hasn\'t'
                      'been fit on any training data. Fit it '
                      'first by calling `.fit(numpy_data)`.')
    if self.featurewise_std_normalization:
      if self.std is not None:
        x /= (self.std + 1e-7)
      else:
        warnings.warn('This ImageDataGenerator specifies '
                      '`featurewise_std_normalization`, but it hasn\'t'
                      'been fit on any training data. Fit it '
                      'first by calling `.fit(numpy_data)`.')
    if self.zca_whitening:
      if self.principal_components is not None:
        flatx = np.reshape(x, (x.size))
        whitex = np.dot(flatx, self.principal_components)
        x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
      else:
        warnings.warn('This ImageDataGenerator specifies '
                      '`zca_whitening`, but it hasn\'t'
                      'been fit on any training data. Fit it '
                      'first by calling `.fit(numpy_data)`.')
    return x

  def random_transform(self, x):
    """Randomly augment a single image tensor.

    # Arguments
        x: 3D tensor, single image.

    # Returns
        A randomly transformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = self.row_axis - 1
    img_col_axis = self.col_axis - 1
    img_channel_axis = self.channel_axis - 1

    # use composition of homographies
    # to generate final transform that needs to be applied
    if self.rotation_range:
      theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
    else:
      theta = 0

    if self.height_shift_range:
      tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
    else:
      tx = 0

    if self.width_shift_range:
      ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
    else:
      ty = 0

    if self.shear_range:
      shear = np.random.uniform(-self.shear_range, self.shear_range)
    else:
      shear = 0

    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
      zx, zy = 1, 1
    else:
      zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

    transform_matrix = None
    if theta != 0:
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
      transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
      shift_matrix = np.array([[1, 0, tx],
                               [0, 1, ty],
                               [0, 0, 1]])
      transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
      shear_matrix = np.array([[1, -np.sin(shear), 0],
                               [0, np.cos(shear), 0],
                               [0, 0, 1]])
      transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
      zoom_matrix = np.array([[zx, 0, 0],
                              [0, zy, 0],
                              [0, 0, 1]])
      transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
      h, w = x.shape[img_row_axis], x.shape[img_col_axis]
      transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
      x = apply_transform(x, transform_matrix, img_channel_axis,
                          fill_mode=self.fill_mode, cval=self.cval)

    if self.channel_shift_range != 0:
      x = random_channel_shift(x,
                               self.channel_shift_range,
                               img_channel_axis)
    if self.horizontal_flip:
      if np.random.random() < 0.5:
        x = flip_axis(x, img_col_axis)

    if self.vertical_flip:
      if np.random.random() < 0.5:
        x = flip_axis(x, img_row_axis)

    return x

  def fit(self, x,
          augment=False,
          rounds=1,
          seed=None):
    """Fits internal statistics to some sample data.

    Required for featurewise_center, featurewise_std_normalization
    and zca_whitening.

    # Arguments
        x: Numpy array, the data to fit on. Should have rank 4.
            In case of grayscale data,
            the channels axis should have value 1, and in case
            of RGB data, it should have value 3.
        augment: Whether to fit on randomly augmented samples
        rounds: If `augment`,
            how many augmentation passes to do over the data
        seed: random seed.

    # Raises
        ValueError: in case of invalid input `x`.
    """
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 4:
      raise ValueError('Input to `.fit()` should have rank 4. '
                       'Got array with shape: ' + str(x.shape))
    if x.shape[self.channel_axis] not in {1, 3, 4}:
      raise ValueError(
        'Expected input to be images (as Numpy array) '
        'following the data format convention "' + self.data_format + '" '
                                                                      '(channels on axis ' + str(
          self.channel_axis) + '), i.e. expected '
                               'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                                                                                               'However, it was passed an array with shape ' + str(
          x.shape) +
        ' (' + str(x.shape[self.channel_axis]) + ' channels).')

    if seed is not None:
      np.random.seed(seed)

    x = np.copy(x)
    if augment:
      ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
      for r in range(rounds):
        for i in range(x.shape[0]):
          ax[i + r * x.shape[0]] = self.random_transform(x[i])
      x = ax

    if self.featurewise_center:
      self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
      broadcast_shape = [1, 1, 1]
      broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
      self.mean = np.reshape(self.mean, broadcast_shape)
      x -= self.mean

    if self.featurewise_std_normalization:
      self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
      broadcast_shape = [1, 1, 1]
      broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
      self.std = np.reshape(self.std, broadcast_shape)
      x /= (self.std + K.epsilon())

    if self.zca_whitening:
      flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
      sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
      u, s, _ = sp.linalg.svd(sigma)
      self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)