from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

tf.executing_eagerly()

# load data
examples, metadata = tfds.load(name="ted_hrlr_translate/pt_to_en",
                               with_info=True,
                               as_supervised=True,
                               data_dir="./dataset/")
train_examples, val_examples = examples["train"], examples["validation"]
print("train example size: %d"%len(list(train_examples)))
print("evalution examples size: %d"%len(list(val_examples)))

# using bpe to tokenize the sentence into subwords
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_generator=(en.numpy() for pt, en in train_examples),   # generator yielding `str`
    target_vocab_size=2 ** 13,
    max_subword_length=20,
    max_corpus_chars=None,
    reserved_tokens=None
)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

def get_dataset(MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE):
    # remove the length longer than 40
    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    # add start and end tokens
    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size + 1]  # add start and end token

        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]))
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = get_dataset(20, 2, 1)
    for batch, (inp, tgt) in enumerate(train_dataset):
        if batch > 1:
            break
        print(inp.shape)
