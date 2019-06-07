import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout
import numpy as np
import math
import sys

np.set_printoptions(threshold=sys.maxsize)


# Input representation is 8 x 8 grid of integers between 0 and 16
input_dim = 64

# Output is probability distribution over 0 to 9
output_dim = 10

# Train and test datasets
# Taken from https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
train = open("optdigits.tra", "r")
test = open("optdigits.tes", "r")
x_train = None
y_train = None
x_test = None
y_test = None

for point in train:
	data_line = np.array(point.split(',')).reshape(1, input_dim+1)
	if x_train is not None:
		x_train = np.vstack((x_train, data_line[:,:-1]))
		y_train = np.vstack((y_train, np.array([[data_line[0, -1][0]]])))
	else:
		x_train = data_line[:,:-1]
		y_train = np.array([[data_line[0, -1][0]]])

for point in test:
	data_line = np.array(point.split(',')).reshape(1, input_dim+1)
	if x_test is not None:
		x_test = np.vstack((x_test, data_line[:,:-1]))
		y_test = np.vstack((y_test, np.array([[data_line[0, -1][0]]])))
	else:
		x_test = data_line[:,:-1]
		y_test = np.array([[data_line[0, -1][0]]])

x_train = x_train.reshape(-1, 8, 8, 1).astype(int)
x_test = x_test.reshape(-1, 8, 8, 1).astype(int)
y_train = keras.utils.to_categorical(y_train.astype(int))
y_test = keras.utils.to_categorical(y_test.astype(int))

# Original train and test datasets

# Original dataset is 32 x 32
input_dim_orig = 32*32

train_orig = open("optdigits-orig.tra", "r")
test_orig = open("optdigits-orig.cv", "r")
x_train_orig = None
y_train_orig = None
x_test_orig = None
y_test_orig = None

ctr = 0
pixels = ''

for point in train_orig:
	ctr += 1
	pixels = pixels + point.strip()
	if ctr == 33:
		pixels = list(pixels)

		data_line = np.array(pixels).reshape(1, input_dim_orig+1)

		if x_train_orig is not None:
			x_train_orig = np.vstack((x_train_orig, data_line[:,:-1]))
			y_train_orig = np.vstack((y_train_orig, np.array([[data_line[0, -1][0]]])))
		else:
			x_train_orig = data_line[:,:-1]
			y_train_orig = np.array([[data_line[0, -1][0]]])

		ctr = 0
		pixels = ''

ctr = 0
pixels = ''

for point in test_orig:
	ctr += 1
	pixels = pixels + point.strip()
	if ctr == 33:
		pixels = list(pixels)

		data_line = np.array(pixels).reshape(1, input_dim_orig+1)
		if x_test_orig is not None:
			x_test_orig = np.vstack((x_test_orig, data_line[:,:-1]))
			y_test_orig = np.vstack((y_test_orig, np.array([[data_line[0, -1][0]]])))
		else:
			x_test_orig = data_line[:,:-1]
			y_test_orig = np.array([[data_line[0, -1][0]]])

		ctr = 0
		pixels = ''


x_train = x_train.reshape(-1, 8, 8, 1).astype(int)
x_test = x_test.reshape(-1, 8, 8, 1).astype(int)
y_train = keras.utils.to_categorical(y_train.astype(int))
y_test = keras.utils.to_categorical(y_test.astype(int))

def scale_down(dp, new_size):
	# Scale down 
	cur_size = len(dp)
	factor = cur_size // new_size
	new_dp = np.zeros(new_size)
	for i in range(0, cur_size, factor):
		if sum(dp[i:i+factor]) > 0:
			new_dp[i//factor] = 1
		else:
			new_dp[i//factor] = 0
	return new_dp

def dump(dp, grid_size):
	for i in range(grid_size):
		for j in range(grid_size):
			print(str(int(dp[grid_size*i+j])), end ="")
		print()
	print()



def pad_data_left(x):
	# h = number of data points, w = size of grid
	h, w = x.shape 
	grid_size = int(math.sqrt(w))
	x_new = None
	for i in range(h):
		dp = x[i,:]
		pad = np.zeros((grid_size, grid_size))
		mod_dp = np.hstack((pad, dp.reshape(grid_size, grid_size)))
		mod_dp = scale_down(mod_dp.flatten(), w)

		#dump(mod_dp, grid_size)

		if x_new is not None:
			x_new = np.vstack((x_new, mod_dp))
		else:
			x_new = mod_dp
	return x_new


def pad_data_right(x, y):
	h, w = x.shape
	pad = np.zeros((h, w))
	x_new = np.hstack((x, pad))
	return (x_new, y)

def pad_data_up(x, y):
	h, w = x.shape
	pad = np.zeros((h, w))
	x_new = np.vstack((pad, x))
	return (x_new, y)

def pad_data_down(x, y):
	h, w = x.shape
	pad = np.zeros((h, w))
	x_new = np.vstack((x, pad))
	return (x_new, y)


x_train_orig_pad_left = pad_data_left(x_train_orig.astype(int)).reshape(-1, 32, 32, 1)

x_train_orig = x_train_orig.reshape(-1, 32, 32, 1).astype(int)
x_train_orig = np.vstack((x_train_orig, x_train_orig_pad_left))
x_test_orig = x_test_orig.reshape(-1, 32, 32, 1).astype(int)

y_train_orig = keras.utils.to_categorical(y_train_orig.astype(int))
y_train_orig = np.vstack((y_train_orig, y_train_orig))
y_test_orig = keras.utils.to_categorical(y_test_orig.astype(int))


model = Sequential()
model.add(Conv2D(filters=100, kernel_size=4, strides=1, input_shape=(32, 32, 1), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=output_dim, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train_orig, y_train_orig, epochs=30, batch_size=32)
model.save("model")

test_loss, test_acc = model.evaluate(x_test_orig, y_test_orig)
print("Test loss: " + str(test_loss))
print("Test accuracy: " + str(test_acc))

