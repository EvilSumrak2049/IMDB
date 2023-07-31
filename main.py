import os
import numpy as np
import shutil
import re
import string
import tensorflow as tf
from keras import layers
from keras import losses
from get_data import raw_train_ds, raw_val_ds, raw_test_ds





def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  stripped_html = tf.strings.regex_replace(stripped_html, r"http",  ' ')
  stripped_html = tf.strings.regex_replace(stripped_html, r"@\S+",  ' ')
  stripped_html = tf.strings.regex_replace(stripped_html, r"@\S+",  ' ')
  stripped_html = tf.strings.regex_replace(stripped_html, r"[^A-Za-z0-9(),!?@\'\`\"\_\n]",  ' ')
  stripped_html = tf.strings.regex_replace(stripped_html, r"@",'at')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


embedding_dim = 16


if __name__=="__main__":
  

  model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])
  
  model.summary()
  
  
  
  model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
  
  
  epochs = 10
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs)


