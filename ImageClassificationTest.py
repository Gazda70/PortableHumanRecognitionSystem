import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# split the dataset into training and testing data
from sklearn.model_selection import train_test_split

# import the Resnet50 and others libraries
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten

# load dataset
dataset = tfds.load('horses_or_humans', split=['train'], as_supervised=True)

# take the images and the target
array = np.vstack(tfds.as_numpy(dataset[0]))
X = np.array(list(map(lambda x: x[0], array)))
y = np.array(list(map(lambda x: x[1], array)))

# visualize
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))
ax1.imshow(X[0])
ax2.imshow(X[50])
ax3.imshow(X[100])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=133, shuffle=True)

y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

# define the ResNet50
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(300,300,3))

# set all nodes on our pre-trained model to be untrainable node
# so that it will not change when we perform our own training
for layer in restnet.layers:
    layer.trainable = False

# create the model
model = Sequential()
model.add(restnet)
model.add(Flatten())
model.add(Dense(16, activation='relu', input_dim=(300,300,3)))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# train our model
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=5,
    verbose=1,
    validation_data=(X_test, y_test),
)

# visualize our training
plt.figure()
plt.plot(range(5), history.history['loss'])
plt.plot(range(5), history.history['val_loss'])
plt.ylabel('Loss')
plt.legend(['training','validation'])
plt.show()

# predict our data
predict = model.predict(X_test)