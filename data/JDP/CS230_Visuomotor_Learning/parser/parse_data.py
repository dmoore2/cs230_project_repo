import numpy as np
import cv2
import sklearn.model_selection as sk
from analysis import parse_protobufs
from ipdb import set_trace as debug

class struct():
	pass

def parse_data(save=True):
	path = struct()
	path.data_folder = 'TeleOpVRSession_2018-02-05_15-44-11/'
	path.data_name = '_SessionStateData.proto'
	data = parse_protobufs(path)
	print(len(data.states))	
	# example data extraction of x value of object/item 0 in training example 0: data.states[0].items[0].x
	num_examples = len(data.states) # number or screenshots
	num_items = []  # number of items in each example
	labels = []
	X_rgb = np.empty([0,299,299,3])
	X_d = np.empty([0,299,299])
	# format labels into n x 6 array
	for i in range(11):
		print(i)
		num_items.append(len(data.states[i].items))
		img_name = str(data.states[i].snapshot.name)
		depth_name = img_name[:-4] + '-Depth.jpg'

		# read in rgb and depth images and add a new axis to them to indicate which snapshot index for each image
		rgb_img = np.expand_dims(cv2.imread(img_name, 1), axis=0)
		depth_img = np.expand_dims(cv2.imread(depth_name, 0), axis = 0)

		for j in range(num_items[i]):
			# input data (X)
			X_rgb = np.vstack([X_rgb, rgb_img])
			X_d = np.vstack([X_d, depth_img])

			# Output label (Y)
			rlabel = data.states[i].items[j] 
			current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch, rlabel.yaw]
			labels.append(current_label)


	# convert to numpy array
	y = np.array(labels)

	if save:
		save_path = 'data/'
		np.save(save_path + "X_rgb.npy", X_rgb)
		np.save(save_path + "X_d.npy", X_d)
		np.save(save_path + "y.npy", y)

	return X_rgb, X_d, y


if __name__ == '__main__':
	X_rgb, X_d, y = parse_data(save=False)
	X = (np.concatenate((X_rgb,np.expand_dims(X_d, axis=3)), axis=3))
	X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=.3, random_state=42)	# random_state=42 ensure indices are same for train/test set for X_rgb and X_d since they must match

