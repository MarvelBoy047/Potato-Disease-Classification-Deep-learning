import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os

# Set environment variable to suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Rest of your code
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r"PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(len(dataset))

# plt.figure(figsize=(10,10))
# for image_batch, label_batch in dataset.take(1):import tensorflow as tf
# from tensorflow.keras import models, layers
# import matplotlib.pyplot as plt
# import os
#
# # Set environment variable to suppress TensorFlow messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Rest of your code
# IMAGE_SIZE = 256
# BATCH_SIZE = 32
# CHANNELS = 3
# EPOCHS = 50
#
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     r"PlantVillage",
#     shuffle=True,
#     image_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE
# )
#
# class_names = dataset.class_names
# print(len(dataset))
#
# # plt.figure(figsize=(10,10))
# # for image_batch, label_batch in dataset.take(1):
# #     for i in range(12):
# #         ax = plt.subplot(3,4,i+1)
# #         plt.imshow(image_batch[i].numpy().astype("uint8"))
# #         plt.title(class_names[label_batch[i]])
# #         plt.axis("off")
# #         plt.show()
#
# train_size = 0.8
# len(dataset)*train_size
#
# train_ds = dataset.take(54)
# len(train_ds)
#
# test_ds = dataset.skip(54)
# len(test_ds)
#
# val_size = 0.1
# len(dataset)*val_size
#
# val_ds = test_ds.take(6)
# len(val_ds)
#
# test_ds = test_ds.skip(6)
# len(test_ds)
#
# def get_dataset_partitions_tf(ds, train_split = 0.8, val_split = 0.1, test_size = 0.1, shuffle=True, shuffle_size = 10000):
#     ds_size = len(ds)
#
#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)
#
#     train_ds = ds.take(train_size)
#
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)
#
#     return test_ds,val_ds,test_ds
#
#
# train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
#
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
#
# resize_and_rescale = tf.keras.Sequential([
#     layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
#     layers.experimental.preprocessing.Rescaling(1.0/255)
# ])
#
# data_augmentation = tf.keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#     layers.experimental.preprocessing.RandomRotation(0.2),
# ])
#
# input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
# n_classes = 3
#
# model = models.Sequential([
#     resize_and_rescale,
#     data_augmentation,
#     layers.Conv2D(32, (3,3), activation='relu', input_shape= input_shape),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation = 'relu'),
#     layers.Dense(n_classes, activation='softmax')
# ])
#
# model.build(input_shape=input_shape)
#
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )
#
# history = model.fit(
#     train_ds,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     verbose=1,
#     validation_data=val_ds
# )
#
# scores = model.evaluate(test_ds)
#
# import numpy as np
# for images_batch, labels_batch in test_ds.take(1):
#
#   first_image = images_batch[0].numpy().astype("uint8")
#   first_label = labels_batch[0]
#
#   print("first image to predict")
#   plt.imshow(first_image)
#   print("actual label :", class_names[first_label])
#
#   batch_prediction = model.predict(images_batch)
#   print("predicted label:",class_names[np.argmax(batch_prediction[0])])
#
#
#   def predict(model, img):
#       img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#       img_array = tf.expand_dims(img_array, 0)
#
#       predictions = model.predict(img_array)
#
#       predicted_class = class_names[np.argmax(predictions[0])]
#       confidence = round(100 * (np.max(predictions[0])), 2)
#       return predicted_class, confidence
#
# for images, labels in test_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#
#     predicted_class, confidence = predict(model, images[i].numpy())
#     actual_class = class_names[labels[i]]
#
#     plt.title(f"\nActual: {actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}%")
#     plt.axis("off")
#     for i in range(12):
#         ax = plt.subplot(3,4,i+1)
#         plt.imshow(image_batch[i].numpy().astype("uint8"))
#         plt.title(class_names[label_batch[i]])
#         plt.axis("off")
#         plt.show()

train_size = 0.8
len(dataset)*train_size

train_ds = dataset.take(54)
len(train_ds)

test_ds = dataset.skip(54)
len(test_ds)

val_size = 0.1
len(dataset)*val_size

val_ds = test_ds.take(6)
len(val_ds)

test_ds = test_ds.skip(6)
len(test_ds)

def get_dataset_partitions_tf(ds, train_split = 0.8, val_split = 0.1, test_size = 0.1, shuffle=True, shuffle_size = 10000):
    ds_size = len(ds)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)

    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return test_ds,val_ds,test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape= input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

scores = model.evaluate(test_ds)

import numpy as np
for images_batch, labels_batch in test_ds.take(1):

  first_image = images_batch[0].numpy().astype("uint8")
  first_label = labels_batch[0]

  print("first image to predict")
  plt.imshow(first_image)
  print("actual label :", class_names[first_label])

  batch_prediction = model.predict(images_batch)
  print("predicted label:",class_names[np.argmax(batch_prediction[0])])


  def predict(model, img):
      img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
      img_array = tf.expand_dims(img_array, 0)

      predictions = model.predict(img_array)

      predicted_class = class_names[np.argmax(predictions[0])]
      confidence = round(100 * (np.max(predictions[0])), 2)
      return predicted_class, confidence

for images, labels in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))

    predicted_class, confidence = predict(model, images[i].numpy())
    actual_class = class_names[labels[i]]

    plt.title(f"\nActual: {actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}%")
    plt.axis("off")