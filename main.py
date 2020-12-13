import os
import random
import sys

import math
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from generator import Generator


# https://stackoverflow.com/a/25512869/6073927
def create_grayscale_masks(image_count, masks_path):
  for image_number in range(0, image_count + 1):
    rgb_mask_image = Image.open(f'{masks_path}/rgb/image{image_number}-mask-trimap.png', 'r').convert('L')  # makes it grayscale
    rgb_mask_pixels = np.asarray(rgb_mask_image.getdata(), dtype=np.float64).reshape((rgb_mask_image.size[1], rgb_mask_image.size[0]))

    black_mask_pixels = np.zeros_like(rgb_mask_pixels)

    # 76 - 29 - 150 - 29 - 76
    red_encountered = False
    green_encountered = False
    blue_encountered = False
    for row in range(len(rgb_mask_pixels)):
      for column in range(len(rgb_mask_pixels[row])):
        value = rgb_mask_pixels[row][column]
        if value < 50:
          blue_encountered = True
          result = 3
        elif value < 125:
          red_encountered = True
          result = 2
        else:
          green_encountered = True
          result = 1
        black_mask_pixels[row][column] = result
    if not red_encountered or not green_encountered or not blue_encountered:
      print("ERROR: trimap does not include all three colors")

    black_mask_pixels = np.asarray(black_mask_pixels, dtype=np.uint8)  # if values still in range 0-255!
    Image.fromarray(black_mask_pixels, mode='L').save(f'{masks_path}/grayscale/image{image_number}-mask-trimap-grayscale.png')


def get_paths(X_path, y_path):
  X_paths = sorted(
    [
      os.path.join(X_path, fname)
      for fname in os.listdir(X_path)
      if fname.endswith(".jpg")
    ]
  )
  y_paths = sorted(
    [
      os.path.join(y_path, fname)
      for fname in os.listdir(y_path)
      if fname.endswith(".png") and not fname.startswith(".")
    ]
  )
  return X_paths, y_paths


def get_model(img_size, num_classes):
  inputs = keras.Input(shape=img_size + (3,))

  # [First half of the network: downsampling inputs]

  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
      previous_block_activation
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  # [Second half of the network: upsampling inputs]

  for filters in [256, 128, 64, 32]:
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)

    # Project residual
    residual = layers.UpSampling2D(2)(previous_block_activation)
    residual = layers.Conv2D(filters, 1, padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  # Add a per-pixel classification layer
  outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

  # Define the model
  return keras.Model(inputs, outputs)


def get_validation_dataset(X_paths, y_paths, img_size, batch_size):
  # Split our img paths into a training and a validation set

  # val_samples = 1000
  # TODO: I changed this because val_samples must be a multiple of batch_size; for some reason, only a multiple of batch_size predictions will be produced
  val_samples = math.floor(0.1 * len(X_paths))
  while val_samples % batch_size != 0:
    val_samples += 1
  print("val_samples: " + str(val_samples))

  random.Random(1337).shuffle(X_paths)
  random.Random(1337).shuffle(y_paths)
  train_input_img_paths = X_paths[:-val_samples]
  train_target_img_paths = y_paths[:-val_samples]
  val_input_img_paths = X_paths[-val_samples:]
  val_target_img_paths = y_paths[-val_samples:]

  # Instantiate data Sequences for each split
  generator_training = Generator(batch_size, img_size, train_input_img_paths, train_target_img_paths)
  generator_validation = Generator(batch_size, img_size, val_input_img_paths, val_target_img_paths)
  return generator_training, generator_validation, val_input_img_paths


def train(model, generator_training, generator_validation, epochs):
  # Configure the model for training.
  # We use the "sparse" version of categorical_crossentropy
  # because our target data is integers.
  model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

  # original
  callbacks = [
    keras.callbacks.ModelCheckpoint("model-checkpoint.h5", save_best_only=True)
  ]

  # Train the model, doing validation at the end of each epoch.
  return model.fit(generator_training, epochs=epochs, validation_data=generator_validation, callbacks=callbacks)


# Creates images where the predicted mask and the input image for which it was generated are interlaced.
def create_interlaced_images(model, generator, X_paths, y_path_prefix):
  predictions = model.predict(generator)

  for i in range(len(X_paths)):
    bw_mask_file_name = X_paths[i].split("/")[-1][:-4] + "-mask.png"
    bw_mask_path = f'{y_path_prefix}/bw/{bw_mask_file_name}'
    os.makedirs(os.path.dirname(bw_mask_path), exist_ok=True)  # create intermediate directories
    bw_mask_pixels = np.argmax(predictions[i], axis=-1)
    bw_mask_pixels = np.expand_dims(bw_mask_pixels, axis=-1)
    keras.preprocessing.image.array_to_img(bw_mask_pixels).resize(Image.open(X_paths[i]).size).save(bw_mask_path, 'PNG')

    black_mask_file_name = X_paths[i].split("/")[-1][:-4] + "-black-mask.png"
    black_mask_path = f'{y_path_prefix}/black/{black_mask_file_name}'
    os.makedirs(os.path.dirname(black_mask_path), exist_ok=True)  # create intermediate directories

    bw_mask_image = Image.open(bw_mask_path, 'r').convert('L')  # makes it grayscale
    bw_mask_pixels = np.asarray(bw_mask_image.getdata(), dtype=np.float64).reshape((bw_mask_image.size[1], bw_mask_image.size[0]))

    black_mask_pixels = np.zeros_like(bw_mask_pixels)
    # 76 - 29 - 150 - 29 - 76
    for row in range(len(bw_mask_pixels)):
      for column in range(len(bw_mask_pixels[row])):
        value = bw_mask_pixels[row][column]
        if value > 127:
          black_mask_pixels[row][column] = 3
        elif value > 0:
          black_mask_pixels[row][column] = 2
        else:
          black_mask_pixels[row][column] = 1
    black_mask_pixels = np.asarray(black_mask_pixels, dtype=np.uint8)  # if values still in range 0-255!
    Image.fromarray(black_mask_pixels, mode='L').save(black_mask_path)

    y_i_path = f'{y_path_prefix}/final/{X_paths[i].split("/")[-1][:-4] + "-final.png"}'
    os.makedirs(os.path.dirname(y_i_path), exist_ok=True)  # create intermediate directories

    X_i_path = X_paths[i]
    os.system("convert '%s' \\( '%s' -fill white -opaque 'rgb(1,1,1)' -opaque 'rgb(3,3,3)' -fill black -opaque 'rgb(2,2,2)' \\) -compose darken -composite '%s'" % (X_i_path, black_mask_path, y_i_path))


# only need to run once to convert manually-created rgb masks to grayscale
def set_up():
  create_grayscale_masks(481, training_masks_path)
  create_grayscale_masks(129, f'{root_directory}/test/masks')


def run():
  input_img_paths, target_img_paths = get_paths(training_path, grayscale_masks_path)
  # Free up RAM in case the model definition cells were run multiple times
  keras.backend.clear_session()
  img_size = (160, 160)
  num_classes = 4
  model = get_model(img_size, num_classes)
  # TODO: the lower the batch_size, the better the prediction
  batch_size = 3
  train_gen, val_gen, val_input_img_paths = get_validation_dataset(input_img_paths, target_img_paths, img_size, batch_size)
  epochs = 15
  history = train(model, train_gen, val_gen, epochs)

  print(history.history.keys())
  print(list(enumerate(history.history['loss'], start=1)))
  print(list(enumerate(history.history['acc'], start=1)))
  print(list(enumerate(history.history['val_loss'], start=1)))
  print(list(enumerate(history.history['val_acc'], start=1)))

  X_test_paths, y_test_paths = get_paths(f'{root_directory}/test', f'{root_directory}/test/masks/grayscale')
  test_gen = Generator(batch_size, img_size, X_test_paths, y_test_paths)
  test_predictions_path = f'{root_directory}/test/masks/predictions'
  create_interlaced_images(model, test_gen, X_test_paths, test_predictions_path)

  score = model.evaluate_generator(test_gen, verbose=1)
  print(f"score: {score}")
  print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


if __name__ == '__main__':
  root_directory = sys.argv[1]
  training_path = f'{root_directory}/training'
  training_masks_path = f'{training_path}/masks'
  predictions_path = f'{training_masks_path}/predictions'
  grayscale_masks_path = f'{training_masks_path}/grayscale'

  # set_up()
  run()
