"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn1 import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger
import pickle
import numpy as np
# import zaddy as patyon
from os.path import basename


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/keras_new',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/JDP/CS230_Visuomotor_Learning/parser/TeleOpVRSession_2018-03-07_14-38-06_Camera1',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")

data_split = [.8, 0.1, .1] 
if __name__ == '__main__':
    path_to_labeling_dict = os.path.join('data/JDP/CS230_Visuomotor_Learning/parser/data', "cam1dict.p")
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    train_data_dir = args.data_dir
    #test_data_dir = os.path.join(data_dir, "test_signs")

    # Get the filenames from the test set
    
    all_filenames = [f for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg') and not f.endswith('Depth.jpg')]
    num_files = len(all_filenames)
    train_filenames = all_filenames[:int(num_files*data_split[0])]
    eval_filenames = all_filenames[int(num_files*data_split[0]) + 1: int(num_files*data_split[0])+int(num_files*data_split[1])]
    test_filenames = all_filenames[int(num_files*data_split[0])+int(num_files*data_split[1]) + 1 :]
#     test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]
    image_to_labels_dict = pickle.load( open(path_to_labeling_dict, "rb") )

    # Labels will be between 0 and 54 included (55 classes in total)
    train_labels = np.zeros((len(train_filenames), params.num_labels))
    eval_labels = np.zeros((len(eval_filenames), params.num_labels))
    test_labels = np.zeros((len(test_filenames), params.num_labels))
    for i in range(len(train_filenames)):
        img_name = train_filenames[i]
        train_labels[i][:] = image_to_labels_dict[basename(img_name)];
    for i in range(len(eval_filenames)):
        img_name = eval_filenames[i]
        eval_labels[i][:] = image_to_labels_dict[basename(img_name)];
    for i in range(len(test_filenames)):
        img_name = test_filenames[i]
        test_labels[i][:] = image_to_labels_dict[basename(img_name)];
        
    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
