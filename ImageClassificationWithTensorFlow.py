#%%
import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
# %%
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
x_train.shape
y_train.shape
x_test.shape
y_test.shape
# %%
plt.imshow(x_train[0],cmap='binary')
plt.show()
y_train[0]
y_train[:20]
# %%
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
y_test_encoded[0]
# %%
x_train_reshaped = np.reshape(x_train,(60000,784))
x_test_reshaped = np.reshape(x_test,(10000,784))
print(x_train_reshaped.shape)
print(x_test_reshaped.shape)
print(set(x_train_reshaped[0]))
# %%
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
print(x_mean)
print(x_std)
# %%
epsilon = 1e-10
x_train_norm = (x_train_reshaped-x_mean)/(x_std+epsilon)
x_test_norm = (x_test_reshaped-x_mean)/(x_std+epsilon)
print(set(x_train_norm[0]))
# %%
model = keras.models.Sequential()
model.add(keras.layers.Dense(128,activation='sigmoid',input_shape=(784,)))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
# %%
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()
# %%
h = model.fit(x_train_norm,y_train_encoded,epochs = 3)
# %%
loss,accuracy = model.evaluate(x_test_norm,y_test_encoded)
print(accuracy*100)
# %%
predictions = model.predict(x_test_norm)
print(predictions.shape)
# %%
plt.figure(figsize=(12,12))
start_index = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred = np.argmax(predictions[start_index+i])
    actual = np.argmax(y_test_encoded[start_index+i])
    col = 'g'
    if pred != actual:
        col = 'r'
    plt.xlabel('i={} | pred={} | true={}'.format(start_index + i, pred, actual), color = col)
    plt.imshow(x_test[start_index+i],cmap='binary')
plt.show()
# %%
index = 10
plt.plot(predictions[index])
plt.show()
# %%
