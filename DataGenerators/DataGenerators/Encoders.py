import os
import itertools
import string
import numpy as np
import pandas as pd

import csv
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

TEXT_OHE_COL_NAME = 'text_ohe'




# TODO add sow and eow symbols to vocabs

def encode_labels_to_ohe(dataset_train_path, train_anno_path, train_anno_fn,
                         dataset_test_path, test_anno_path, test_anno_fn):
  '''
  enocode annotations' labels into ohe encoding
  :param dataset_path:
  :param train_anno_path:
  :param train_anno_fn:
  :return:
  '''

  # load annotations
  annos_train = load_annos(train_anno_fn, train_anno_path, dataset_train_path)
  annos_test = load_annos(test_anno_fn, test_anno_path, dataset_test_path)


  # extract vocab/vocabs sets based on annotations' labels
  vocab, labels_train, labels_test, labels_maxlen, n_vocab = extract_vocab(annos_train, annos_test)

  # create encoding maps for annotations' labels
  char_to_int, int_to_char = create_enc_maps(vocab)

  # encode train labels into padded int code
  train_label_enc_padded_int = encode_str_to_padded_int(char_to_int, labels_train, labels_maxlen)

  # encode test labels into padded int code
  test_label_enc_padded_int = encode_str_to_padded_int(char_to_int, labels_test, labels_maxlen)

  # one hot encode train labels
  train_labels_enc_ohe, train_label_enc_ohe_shape = encode_padded_int_to_ohe(train_label_enc_padded_int, n_vocab)
  print("train labels shape ", train_label_enc_ohe_shape)
  # one hot encode test labels
  test_labels_enc_ohe, test_label_enc_ohe_shape = encode_padded_int_to_ohe(test_label_enc_padded_int, n_vocab)
  print("test labels shape ", test_label_enc_ohe_shape)

  if train_label_enc_ohe_shape == test_label_enc_ohe_shape:
    label_enc_ohe_shape = train_label_enc_ohe_shape

  annos_train[TEXT_OHE_COL_NAME] = pd.Series(train_labels_enc_ohe).values
  annos_test[TEXT_OHE_COL_NAME] = pd.Series(test_labels_enc_ohe).values

  return annos_train, annos_test, vocab, n_vocab, label_enc_ohe_shape

  # invert encoding
  # inverted = np.argmax(np.array(label_enc_ohe), axis=1)
  # print(inverted)

def encode_str_to_ohe():
  # load annotations
  annos_train = load_annos(train_anno_fn, train_anno_path, dataset_train_path)
  annos_test = load_annos(test_anno_fn, test_anno_path, dataset_test_path)

  # extract vocab/vocabs sets based on annotations' labels
  vocab, labels_train, labels_test, labels_maxlen, n_vocab = extract_vocab(annos_train, annos_test)

  # create encoding maps for annotations' labels
  char_to_int, int_to_char = create_enc_maps(vocab)

  # encode train labels into padded int code
  train_label_enc_padded_int = encode_str_to_padded_int(char_to_int, labels_train, labels_maxlen)

  # encode test labels into padded int code
  test_label_enc_padded_int = encode_str_to_padded_int(char_to_int, labels_test, labels_maxlen)

  # one hot encode train labels
  train_labels_enc_ohe, train_label_enc_ohe_shape = encode_int_to_ohe(train_label_enc_padded_int, n_vocab)
  print("train labels shape ", train_label_enc_ohe_shape)
  # one hot encode test labels
  test_labels_enc_ohe, test_label_enc_ohe_shape = encode_int_to_ohe(test_label_enc_padded_int, n_vocab)
  print("test labels shape ", test_label_enc_ohe_shape)

  if train_label_enc_ohe_shape == test_label_enc_ohe_shape:
    label_enc_ohe_shape = train_label_enc_ohe_shape

  annos_train[TEXT_OHE_COL_NAME] = pd.Series(train_labels_enc_ohe).values
  annos_test[TEXT_OHE_COL_NAME] = pd.Series(test_labels_enc_ohe).values

  return annos_train, annos_test, vocab, n_vocab, label_enc_ohe_shape

def encode_padded_int_to_ohe(data_wrd_enc_int_pad, n_vocab):
  data_wrd_enc_he = [to_categorical(word_encoded, num_classes=n_vocab) for word_encoded in data_wrd_enc_int_pad]
  # data_wrd_enc_he = [[to_categorical(word_char_encoded, num_classes=n_vocab) for word_char_encoded in list(word_encoded)]
  #                                                                             for word_encoded in data_wrd_enc_int_pad]
  # print(data_wrd_enc_he)
  # print(np.shape(data_wrd_enc_he))
  # print(data_words[0])
  # print(data_wrd_enc_int_pad[0])
  # print(data_wrd_enc_he[0][5])
  # print(np.shape(data_wrd_enc_he[0]))
  # print(np.shape(data_wrd_enc_he[1]))
  # print(data_wrd_enc_he[0])
  return data_wrd_enc_he, np.shape(data_wrd_enc_he)[1:]

# TODO use scikit-learn LabelEncoder()
def encode_str_to_padded_int(char_to_int, data_words, data_words_len_max):
  data_wrd_enc_int = encode_str_to_int(char_to_int, data_words)
  data_wrd_enc_int_pad = pad_sequences(data_wrd_enc_int, maxlen=data_words_len_max, padding='post', truncating='post',
                                       value=0.)
  data_wrd_enc_int_pad_len = [len(word) for word in data_wrd_enc_int_pad]
  data_int_enc_pad_len_max = max(data_wrd_enc_int_pad_len)
  # print(data_wrd_enc_int_pad[:3])
  # print(data_int_enc_pad_len)
  # print(data_int_enc_pad_len_max)

  return data_wrd_enc_int_pad


def encode_str_to_int(char_to_int, data_words):
  data_wrd_enc_int = [[char_to_int[char] for char in word] for word in data_words]
  # print(data_wrd_enc_int)
  # print(data_words[0:3])
  # print(data_wrd_enc_int[0:3])
  return data_wrd_enc_int


def create_enc_maps(chars):
  # create mapping of unique chars to integers
  char_to_int = dict((c, i + 1) for i, c in enumerate(chars))
  int_to_char = dict((i + 1, c) for i, c in enumerate(chars))
  # char_to_ix = {ch: i for i, ch in enumerate(chars)}
  # ix_to_char = {i: ch for i, ch in enumerate(chars)}
  return char_to_int, int_to_char


def extract_vocab(train_annos, test_annos):
  annos = pd.concat([train_annos, test_annos])

  train_annos_labels = train_annos['text'].str.replace('"', '').tolist()
  test_annos_labels = test_annos['text'].str.replace('"', '').tolist()
  annos_labels = annos['text'].str.replace('"', '').tolist()

  labels_words = annos_labels
  labels_words_len = [len(word) for word in labels_words]
  labels_words_len_max = max(labels_words_len)
  # print(labels_words_len_max)
  labels_chars = [[char for char in list(word)] for word in annos_labels]
  labels_chars = itertools.chain(*labels_chars)
  labels_chars = list(labels_chars)
  # print(labels_chars)
  # print(data_text)
  extra_chars = ['<sow>', '<eow>', '<spc>']
  raw_chars = list(string.printable)
  all_chars = labels_chars + raw_chars + extra_chars
  vocab = sorted(list(set(all_chars)))
  print("".join(vocab), len(vocab))
  n_labels_chars, n_vocab = len(labels_chars), len(vocab)
  # print('data has %d alphabet, %d unique.' % (n_labels_chars, n_vocab))
  return vocab, train_annos_labels, test_annos_labels, labels_words_len_max, n_vocab

# FIXME use text reader instead of csv reader as there are some text strings that have a comma which can fool the csv parser
def load_annos(train_anno_fn, train_anno_path, dataset_path):
  annos_fname = os.path.join(dataset_path, train_anno_path, train_anno_fn)
  # annos_df = pd.read_csv(annos_fname)
  annos = pd.read_csv(annos_fname,
                            names=['image', 'text'],
                            header=None, quoting=csv.QUOTE_ALL)
  # annos['text'] = pd.Series(annos['text']).str.replace('"','')
  annos['text'] = annos['text'].str.lstrip()
  # annos.head()
  return annos