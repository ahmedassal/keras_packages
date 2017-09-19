import os
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator, img_to_array, array_to_img, load_img
from .Encoders import TEXT_OHE_COL_NAME

class BasicDirectoryIterator(Iterator):
  """Iterator capable of reading images from a directory on disk.

  # Arguments
      directory: Path to the directory to read images from.
      image_data_generator: Instance of `ImageDataGenerator`
          to use for random transformations and normalization.
      target_size: tuple of integers, dimensions to resize input images to.
      color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
      classes: Optional list of strings, names of sudirectories
          containing images from each class (e.g. `["dogs", "cats"]`).
          It will be computed automatically if not set.
      class_mode: Mode for yielding the targets:
          `"binary"`: binary targets (if there are only two classes),
          `"categorical"`: categorical targets,
          `"sparse"`: integer targets,
          `None`: no targets get yielded (only input images are yielded).
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seed for data shuffling.
      data_format: String, one of `channels_first`, `channels_last`.
      save_to_dir: Optional directory where to save the pictures
          being yielded, in a viewable format. This is useful
          for visualizing the random transformations being
          applied, for debugging purposes.
      save_prefix: String prefix to use for saving sample
          images (if `save_to_dir` is set).
      save_format: Format to use for saving sample images
          (if `save_to_dir` is set).
  """

  def __init__(self, directory, image_data_generator,
               target_size=(256, 256), color_mode='rgb',
               # classes=None, class_mode='categorical',
               labels=None,
               labels_shape = None,
               batch_size=32, shuffle=True, seed=None,
               data_format=None,
               save_to_dir=None, save_prefix='', save_format='jpeg',
               follow_links=False):
    if data_format is None:
      data_format = K.image_data_format()
    self.directory = directory
    self.image_data_generator = image_data_generator
    self.target_size = tuple(target_size)
    if color_mode not in {'rgb', 'grayscale'}:
      raise ValueError('Invalid color mode:', color_mode,
                       '; expected "rgb" or "grayscale".')
    self.color_mode = color_mode
    self.data_format = data_format
    if self.color_mode == 'rgb':
      if self.data_format == 'channels_last':
        self.image_shape = self.target_size + (3,)
      else:
        self.image_shape = (3,) + self.target_size
    else:
      if self.data_format == 'channels_last':
        self.image_shape = self.target_size + (1,)
      else:
        self.image_shape = (1,) + self.target_size
    # self.classes = classes
    # if class_mode not in {'categorical', 'binary', 'sparse', None}:
    #   raise ValueError('Invalid class_mode:', class_mode,
    #                    '; expected one of "categorical", '
    #                    '"binary", "sparse", or None.')
    # self.class_mode = class_mode
    self.labels = labels
    self.labels_shape = labels_shape
    self.save_to_dir = save_to_dir
    self.save_prefix = save_prefix
    self.save_format = save_format

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    # first, count the number of samples and classes
    self.samples = 0

    # if not classes:
    #   classes = []
    #   for subdir in sorted(os.listdir(directory)):
    #     if os.path.isdir(os.path.join(directory, subdir)):
    #       classes.append(subdir)
    # self.num_class = len(classes)
    # self.class_indices = dict(zip(classes, range(len(classes))))

    def _recursive_list(subpath):
      return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    i = 0
    self.filenames = []
    for root, _, files in _recursive_list(directory):
      for fname in files:
        is_valid = False
        for extension in white_list_formats:
          if fname.lower().endswith('.' + extension):
            is_valid = True
            break
        if is_valid:
          self.samples += 1
          # add filename relative to directory
          absolute_path = os.path.join(root, fname)
          self.filenames.append(os.path.relpath(absolute_path, directory))
    print('Found %d images in %s directory.' % (self.samples, self.directory))
    super(BasicDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

  def next(self):
    """For python 2.x.

    # Returns
        The next batch.
    """
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock
    # so it can be done in parallel
    self.labels_new_shape = (self.labels_shape[0] * self.labels_shape[1],)
    batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
    batch_y = np.zeros((current_batch_size,) + self.labels_shape, )
    # batch_y = np.zeros((current_batch_size,) + self.labels_new_shape, )
    # batch_y = list()
    grayscale = self.color_mode == 'grayscale'
    # build batch of image data
    for i, j in enumerate(index_array):
      fname = self.filenames[j]
      img = load_img(os.path.join(self.directory, fname),
                     grayscale=grayscale,
                     target_size=self.target_size)
      x = img_to_array(img, data_format=self.data_format)
      # x = self.image_data_generator.random_transform(x)
      # x = self.image_data_generator.standardize(x)
      batch_x[i] = x
      idx = self.labels.index[self.labels['image'] == os.path.basename(fname)].tolist()
      # print(idx)
      y = self.labels.loc[idx, TEXT_OHE_COL_NAME]
      if len(y)==0:
        print("Image {} not found!!! in Directory {}".format(fname, self.directory))
      else:
        y = y.values[0]
        # y = np.reshape(y, self.labels_new_shape)
        batch_y[i] = y
        # batch_y.append(y)

      # y = self.labels.ix[self.labels['image'] == os.path.basename(fname), TEXT_OHE_COL_NAME]
      #y =  row[ TEXT_OHE_COL_NAME].values
      # y = y.reshape((y.shape[0], y.shape[1], 1))


    # optionally save augmented images to disk for debugging purposes
    if self.save_to_dir:
      for i in range(current_batch_size):
        img = array_to_img(batch_x[i], self.data_format, scale=True)
        fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                          index=current_index + i,
                                                          hash=np.random.randint(1e4),
                                                          format=self.save_format)
        img.save(os.path.join(self.save_to_dir, fname))
    # build batch of labels
    # if self.class_mode == 'sparse':
    #   batch_y = self.classes[index_array]
    # elif self.class_mode == 'binary':
    #   batch_y = self.classes[index_array].astype(K.floatx())
    # elif self.class_mode == 'categorical':
    #   batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
    #   for i, label in enumerate(self.classes[index_array]):
    #     batch_y[i, label] = 1.
    else:
      # return batch_x
      # batch_y = list(batch_y)
      # batch_y = batch_y.tolist()
      # print(batch_y[0])
      # print("batch_y shape {} {}".format(np.shape(batch_y), type(batch_y) ))
      batch_y = list(np.transpose(batch_y, (1, 0, 2)))
      print(np.shape(batch_y))
      return batch_x, batch_y
    # batch_y = list(batch_y)
    # batch_y = batch_y.tolist()
    batch_y = list(np.transpose(batch_y, (1,0,2)))
    print(np.shape(batch_y))
    return batch_x, batch_y
