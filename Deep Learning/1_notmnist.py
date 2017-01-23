#! Assignment 1
#! The objective of this assignment is to learn about simple data curation practices, 
#! and familiarize you with some of the data we'll be reusing later.
#! This notebook uses the notMNIST dataset to be used with python experiments.
#! This dataset is designed to look like the classic MNIST dataset,
#! while looking a little more like real data: 
#! it's a harder task, and the data is a lot less 'clean' than MNIST.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


#! First, we'll download the dataset to our local machine. 
#! The data consists of characters rendered in a variety of fonts on a 28x28 image. 
#! The labels are limited to 'A' through 'J' (10 classes). 
#! The training set has about 500k and the testset 19000 labelled examples.
#! Given these sizes, it should be possible to train models quickly on any machine.
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def DownloadProgressHook(count, block_size, total_size):
	"""A hook to report the progress of a download. This is mostly intended for users with
	slow internet connections. Reports every 1% change in download progress.
	"""
	global last_percent_reported
	percent = int(count * block_size * 100 / total_size)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
			
		last_percent_reported = percent
				
def MaybeDownload(filename, expected_bytes, force=False):
	"""Download a file if not present, and make sure it's the right size."""
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url + filename, filename, reporthook=DownloadProgressHook)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		raise Exception(
			'Failed to verify ' + filename + '. Can you get to it with a browser?')
	return filename

train_filename = MaybeDownload('notMNIST_large.tar.gz', 247336696)
test_filename = MaybeDownload('notMNIST_small.tar.gz', 8458043)

#! Extract the dataset from the compressed .tar.gz file.
#! This should give you a set of directories, labelled A through J.
num_classes = 10
np.random.seed(133)
def DataFloder(filename,force=False):
	#! remove .tar.gz
	root = os.path.splitext(os.path.splitext(filename)[0])[0]  
	if os.path.isdir(root) and not force:
		#! You may override by setting force=True.
		print('%s already present - Skipping extraction of %s.' % (root, filename))
	else:
		print('Extracting data for %s. This may take a while. Please wait.' % root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	data_folders = [
		os.path.join(root,d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root,d))
	]
	if len(data_folders) != num_classes:
		raise Exception('Expected %d folders, one per class, Found %d instead.' %
			(num_classes,len(data_folders)))
	print(data_folders)
	return data_folders
train_folders = DataFloder(train_filename)
test_folders = DataFloder(test_filename)


#! Problem 1
#! Let's take a peek at some of the data to make sure it look sensible.
#! Each examplar should be an image of character A though J rendered in a different font.
#! Display a sample of th images that we just downloaded
#! Hint: you can use the package IPython.display

def ShowImage(folder_index = None,file_index = None):
	if not folder_index:
		folder_index = 0
	if not file_index:
		file_index = 0
	file_names = os.listdir(os.path.join(os.getcwd(),train_folders[folder_index]))
	file_path = os.path.join(os.getcwd(),train_folders[folder_index],file_names[file_index])
	return mpimg.imread(file_path)
images = [ShowImage(i,j) for i in range(10) for j in [9]]
for image in images:
	plt.imshow(image,cmap='Greys_r')
	#! @breif show image
#	plt.show()
	
#! Now let's load the data in a more manageable format. 
#! Since, depending on your computer setup you might not be able to fit it all in memory, 
#! we'll load each class into a separate dataset, store them on disk and curate them independently.
#! Later we'll merge them into a single dataset of manageable size.
#! We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, 
#! normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.
#! A few images might not be readable, we'll just skip them.

#! Pixel width and height
image_size = 28
#! Number of levels per pixel
pixl_depth = 255.0

def LoadLetter(folder,min_num_images):
	"""
	Load the data for a single letter label.
	"""
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape(len(image_files),image_size,image_size),dtype=np.float32)
	print(folder)
	num_images = 0
	for image in image_files:
		image_file = os.path.join(folder, image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
		
	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' %
										(num_images, min_num_images))
		
	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset
	
def MaybePickle(data_folders, min_num_images_per_class, force=False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		if os.path.exists(set_filename) and not force:
			#! You may override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)
			dataset = LoadLetter(folder, min_num_images_per_class)
			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)
	return dataset_names
train_datasets = MaybePickle(train_folders, 45000)
test_datasets = MaybePickle(test_folders, 1800)
print(train_datasets)
#! Problem 2
#! Let's verify that the data stills looks good. 
#! Displaying a sample of the labels and images from the ndarray
#! Hint: you can use matplotlib.pyplot
letter_sample = pickle.load(open(train_datasets[0],'rb'))
plt.imshow(letter_sample[9,:,:],cmap='Greys_r')
plt.show()
#! Problem 3
#! Another check: we expect the data to be balanced across classes. Verify that.
for pickle_file in train_datasets:
	data = pickle.load(open(pickle_file,'rb'))
	print(pickle_file," size:",data.shape[0])
	
#! Merge and prune the training data as needed. 
#! Depending on your computer setup, you might not be able to fit it all in memory, 
#! and you can tune train_size as needed.
#! The labels will be stored into a separate array of integers 0 through 9.
#! Also create a validation dataset for hyperparameter tuning.

def MakeArrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels
	
def MergeDatasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = MakeArrays(valid_size, image_size)
	train_dataset, train_labels = MakeArrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes
		
	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):       
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# let's shuffle the letters to have random validation and training set
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class
										
				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
	return valid_dataset, valid_labels, train_dataset, train_labels
	
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = MergeDatasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = MergeDatasets(test_datasets, test_size)
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

#! Next, we'll randomize the data.
#! It's important to have the labels well shuffled for the training and test distributions to match.
def randomize(dataset,labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#! Problem 4
#! Convince yourself that the data is still good after shuffling!
num_rand = np.random.randint(train_dataset.shape[0])
plt.imshow(train_dataset[num_rand,:,:],cmap='Greys_r')
plt.show()
print('Train labels :',train_labels[num_rand])
#! check label balanc
np.bincount(train_labels)

#! Finally, let's save the data for later reuse:
pickle_file = 'notMNIST.pickle'
try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
		}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise
stat_info = os.stat(pickle_file)
print('Compressed pickle size:', stat_info.st_size)

#! Problem 5
#! By construction, this dataset might contain a lot of overlapping samples, 
#! including training data that's also contained in the validation and test set! 
#! Overlap between training and test can skew the results 
#! if you expect to use your model in an environment where there is never an overlap,
#! but are actually ok if you expect to see training samples recur when you use it. 
#! Measure how much overlap there is between training, validation and test samples.
#! 
#! Optinal questions:
#! 	* What about near duplicates between datasets? (images that are almost identical)
#!  * Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
def Duplicates(dataset_1,dateset_2):
	num_duplicates = 0
	for image_1 in dataset_1:
		for image_2 in dataset_2:
			if np.array_equal(image_1,image_2):
				num_duplicates +=1
	return num_duplicates
#print("number of duplicates between train and test: {}".format(Duplicates(train_dataset, test_dataset)))
#print("number of duplicates between validation and test: {}".format(Duplicates(valid_dataset, test_dataset)))
#print("number of duplicates between train and validation: {}".format(Duplicates(valid_dataset, train_dataset)))

#! Thoughts on optional questions: 
#! maybe there are somekind of hashing method can be used to find similar images.
#! The following solution obtained from :
#! https://discussions.udacity.com/t/assignment-1-problem-5/45657/32

def FastOverlapsHashlibAndNumpy():
	import hashlib
	train_hashes = [hashlib.sha1(x).digest() for x in train_dataset]
	valid_hashes = [hashlib.sha1(x).digest() for x in valid_dataset]
	test_hashes  = [hashlib.sha1(x).digest() for x in test_dataset]

	valid_in_train = np.in1d(valid_hashes, train_hashes)
	test_in_train  = np.in1d(test_hashes,  train_hashes)
	test_in_valid  = np.in1d(test_hashes,  valid_hashes)

	valid_keep = ~valid_in_train
	test_keep  = ~(test_in_train | test_in_valid)

	valid_dataset_clean = valid_dataset[valid_keep]
	valid_labels_clean  = valid_labels [valid_keep]

	test_dataset_clean = test_dataset[test_keep]
	test_labels_clean  = test_labels [test_keep]

	print("valid -> train overlap: %d samples" % valid_in_train.sum())
	print("test  -> train overlap: %d samples" % test_in_train.sum())
	print("test  -> valid overlap: %d samples" % test_in_valid.sum())
	
import time
print('Test : Hashlib and Numpy')
start = time.time()
FastOverlapsHashlibAndNumpy()
print("Time: %0.2fs" % (time.time()-start))

#! Problem 6
#! Let's get an idea of what an off-the-shelf classifier can give you on this data. 
#! It's always good to check that there is something to learn, 
#! and that it's a problem that is not so trivial that a canned solution solves it.
#! Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
#!  Hint: you can use the LogisticRegression model from sklearn.linear_model.
#!
#! Optional question: 
#! 	* train an off-the-shelf model on all the data!

num_train = train_dataset.shape[0]
num_validation = valid_dataset.shape[0]
num_test = test_dataset.shape[0]
num_features = image_size**2
print(num_train,num_validation,num_test,num_features)

def MakeDatasetSample(dataset,labels,num_samples=None):
	if num_samples:
		indices = np.random.choice(dataset.shape[0],num_samples,replace=False)
		dataset = dataset[indices].reshape((num_samples,num_features))
		labels = labels[indices]
	return dataset,labels

for num_samples in [50,100,1000,5000,train_dataset.shape[0]]:
	start = time.time()
	print("Training using ",num_samples," smaples from the training set")
	dataset,labels = MakeDatasetSample(train_dataset, train_labels,num_samples)
	model = LogisticRegression()
	model.fit(dataset,labels)
	print("Training model using %0.2fs" % (time.time()-start))
	print("Traning accurary:",model.score(dataset,labels))
	print("Validation accuracy: ",model.score(valid_dataset.reshape(num_validation,num_features),valid_labels))
	print("Test accuracy: ",model.score(test_dataset.reshape(num_test,num_features),test_labels))
	print("")
	
	