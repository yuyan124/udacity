#! Assignment 3
#! Previously in 2_fullyconnected.ipynb, you trained a logitsic regression and a neural network model.
#! The goal of this assignment is to explore regularization techniques.

#! These are all the modules we'll be using later. Make sure you can import them
#! before proceeding further.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
#! First reload the data we generated in 1_notmnist.ipynb.
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)
	
#! Reformat into a shape that's more adapted to the models we're going to train:
#! * data as a flat matrix,
#! * labels as float 1-hot encodings.
image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
					/ predictions.shape[0])
		


#! Problem 1
#! Introduce and tune L2 regularization for both logitsic and neural network models. 
#! Remember that L2 amounts to adding a penalty on the norm of the weights to the loss.
#! In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t).
#! The right amount of regularization should improve your validation / test accuracy.

#! logitsic Regression
batch_size = 128
graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	beta_regular = tf.placeholder(tf.float32)
	# Variables.
	weights = tf.Variable(
		tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
	
	# Training computation.
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	loss += (beta_regular * tf.nn.l2_loss(weights))
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

#!Let's run it:
num_steps = 3001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Problem 1 : using Logistic Regression")
	print("Initialized")
	for step in range(num_steps):
		#! Pick an offset within the training data, which has been randomized.
		#! Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		#! Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		#! Prepare a dictionary telling the session where to feed the minibatch.
		#! The key of the dictionary is the placeholder node of the graph to be fed,
		#! and the value is the numpy array to feed to it.
		
		#! @brief test 0.1 0.01 0.001,0.0001
		#! 0.001 seem best
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regular : 0.01 }
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#! Neural Network
batch_size = 128
hidden_layer_nodes = 1024
graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	beta_regular = tf.placeholder(tf.float32)
	
	# Variables.
	weights_1 = tf.Variable(
		tf.truncated_normal([image_size * image_size, hidden_layer_nodes]))
	biases_1 = tf.Variable(tf.zeros([hidden_layer_nodes]))
	weights_2 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes, num_labels]))
	biases_2 = tf.Variable(tf.zeros([num_labels]))
	
	# Training computation.
	logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	hidden_1 = tf.nn.relu(logits_1)
	logits_2 = tf.matmul(hidden_1,weights_2)+biases_2

	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
	loss += beta_regular*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits_2)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights_1)+biases_1),weights_2)+biases_2)
	test_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,weights_1)+biases_1),weights_2)+biases_2)


num_steps = 3001
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Problem 1 : using Nueral Network")
	print("Initialized")
	for step in range(num_steps):
		#! Pick an offset within the training data, which has been randomized.
		#! Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		#! Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		#! Prepare a dictionary telling the session where to feed the minibatch.
		#! The key of the dictionary is the placeholder node of the graph to be fed,
		#! and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels ,beta_regular:0.001}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#! Problem 2
#! Let's demonstrate an extreme case of overfitting.
#! Restrict your training data to just a few batches. What happens?
#! Generalization capability is much poor

#! Neural Network
batch_size = 128
hidden_layer_nodes = 1024
graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	beta_regular = tf.placeholder(tf.float32)
	
	# Variables.
	weights_1 = tf.Variable(
		tf.truncated_normal([image_size * image_size, hidden_layer_nodes]))
	biases_1 = tf.Variable(tf.zeros([hidden_layer_nodes]))
	weights_2 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes, num_labels]))
	biases_2 = tf.Variable(tf.zeros([num_labels]))
	
	# Training computation.
	logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	hidden_1 = tf.nn.relu(logits_1)
	logits_2 = tf.matmul(hidden_1,weights_2)+biases_2

	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
	loss += beta_regular*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits_2)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights_1)+biases_1),weights_2)+biases_2)
	test_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,weights_1)+biases_1),weights_2)+biases_2)


num_steps = 3001
#! @brief test 1-10
num_bacthes = 3
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Problem 2 : using Nueral Network")
	print("Initialized")
	for step in range(num_steps):
		#! Pick an offset within the training data, which has been randomized.
		#! Note: we could use better randomization across epochs.
		#offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		offset = step % num_bacthes
		#! Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		#! Prepare a dictionary telling the session where to feed the minibatch.
		#! The key of the dictionary is the placeholder node of the graph to be fed,
		#! and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels ,beta_regular:0.001}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#! Problem 3
#! Introduce Dropout on the hidden layer of the neural network.
#! Remember: Dropout should only be introduced during training, not evaluation,
#! otherwise your evaluation results would be stochastic as well.
#! TensorFlow provides nn.dropout() for that,
#! but you have to make sure it's only inserted during training.
#! What happens to our extreme overfitting case?

#! Neural Network
batch_size = 128
hidden_layer_nodes = 1024
graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	beta_regular = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	
	# Variables.
	weights_1 = tf.Variable(
		tf.truncated_normal([image_size * image_size, hidden_layer_nodes]))
	biases_1 = tf.Variable(tf.zeros([hidden_layer_nodes]))
	weights_2 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes, num_labels]))
	biases_2 = tf.Variable(tf.zeros([num_labels]))
	
	# Training computation.
	logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	hidden_1 = tf.nn.relu(logits_1)
	drop_1 = tf.nn.dropout(hidden_1,keep_prob)
	logits_2 = tf.matmul(drop_1,weights_2)+biases_2

	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
	loss += beta_regular*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits_2)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights_1)+biases_1),weights_2)+biases_2)
	test_prediction = tf.nn.softmax(
		tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,weights_1)+biases_1),weights_2)+biases_2)


num_steps = 3001
#! @brief test 1-10
num_bacthes = 3
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Problem 3 : using Nueral Network")
	print("Initialized")
	for step in range(num_steps):
		#! Pick an offset within the training data, which has been randomized.
		#! Note: we could use better randomization across epochs.
		#offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		offset = step % num_bacthes
		#! Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		#! Prepare a dictionary telling the session where to feed the minibatch.
		#! The key of the dictionary is the placeholder node of the graph to be fed,
		#! and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels ,beta_regular:0.001,keep_prob:0.5}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#! Problem 4
#! Try to get the best performance you can using a multi-layer model!
#! The best reported test accuracy using a deep network is 97.1%.
#! One avenue you can explore is to add multiple layers.
#! Another one is to use learning rate decay:
#! 		global_step = tf.Variable(0)  # count the number of steps taken.
#! 		learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#! 		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#! Neural Network
batch_size = 128
hidden_layer_nodes_1 = 1024
hidden_layer_nodes_2 = 512
hidden_layer_nodes_3 = 256
graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	beta_regular = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	global_step = tf.Variable(0)
	
	# Variables.
	weights_1 = tf.Variable(
		tf.truncated_normal([image_size * image_size, hidden_layer_nodes_1],stddev=np.sqrt(2.0/(image_size**2))))
	biases_1 = tf.Variable(tf.zeros([hidden_layer_nodes_1]))
	weights_2 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes_1, hidden_layer_nodes_2],stddev=np.sqrt(2.0/(hidden_layer_nodes_1))))
	biases_2 = tf.Variable(tf.zeros([hidden_layer_nodes_2]))
	weights_3 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes_2, hidden_layer_nodes_3],stddev=np.sqrt(2.0/(hidden_layer_nodes_2))))
	biases_3 = tf.Variable(tf.zeros([hidden_layer_nodes_3]))
	weights_4 = tf.Variable(
		tf.truncated_normal([hidden_layer_nodes_3,num_labels],stddev=np.sqrt(2.0/(hidden_layer_nodes_3))))
	biases_4 = tf.Variable(tf.zeros([num_labels]))

	
	# Training computation.
	logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	hidden_1 = tf.nn.relu(logits_1)
	#drop_1 = tf.nn.dropout(hidden_1,keep_prob)
	#logits_2 = tf.matmul(drop_1,weights_2)+biases_2
	logits_2 = tf.matmul(hidden_1,weights_2)+biases_2
	hidden_2 = tf.nn.relu(logits_2)
	#drop_2 = tf.nn.dropout(hidden_2,keep_prob)
	#logits_3 = tf.matmul(drop_2,weights_3)+biases_3
	logits_3 = tf.matmul(hidden_2,weights_3)+biases_3
	hidden_3 = tf.nn.relu(logits_3)
	drop_3 = tf.nn.dropout(hidden_3,keep_prob)
	logits_4 = tf.matmul(drop_3,weights_4)+biases_4

	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_4))
	loss += beta_regular*(tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) +
		 tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4))

	# Optimizer.
	learning_rate = tf.train.exponential_decay(0.5,global_step,10000,0.7,staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits_4)
	valid_hidden_1 = tf.nn.relu(tf.matmul(tf_valid_dataset,weights_1)+biases_1)
	valid_hidden_2 = tf.nn.relu(tf.matmul(valid_hidden_1,weights_2)+biases_2)
	valid_hidden_3 = tf.nn.relu(tf.matmul(valid_hidden_2,weights_3)+biases_3)
	valid_prediction = tf.nn.softmax(
		tf.matmul(valid_hidden_3,weights_4)+biases_4)
	test_hidden_1 = tf.nn.relu(tf.matmul(tf_test_dataset,weights_1)+biases_1)
	test_hidden_2 = tf.nn.relu(tf.matmul(test_hidden_1,weights_2)+biases_2)
	test_hidden_3 = tf.nn.relu(tf.matmul(test_hidden_2,weights_3)+biases_3)
	test_prediction = tf.nn.softmax(
		tf.matmul(test_hidden_3,weights_4)+biases_4)


num_steps = 20001
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Problem 4 : using Nueral Network")
	print("Initialized")
	for step in range(num_steps):
		#! Pick an offset within the training data, which has been randomized.
		#! Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		#offset = step % num_bacthes
		#! Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		#! Prepare a dictionary telling the session where to feed the minibatch.
		#! The key of the dictionary is the placeholder node of the graph to be fed,
		#! and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels ,beta_regular:0.002,keep_prob:0.5}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
