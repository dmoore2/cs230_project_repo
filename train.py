"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf
from model.model_fn1 import model_fn
from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
# from model.model_fn import model_fn
from model.training import train_and_evaluate
import pickle
import numpy as np
from os.path import basename


def printExample(label, filename):
    print(basename(filename))
    print("sum " + str(np.sum(label)))
    print('contains objects:')
    for i in range(len(label)):
        if(label[i] == 1):
            print(i)
    print(" ")

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/keras_new',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/JDP/CS230_Visuomotor_Learning/parser/TeleOpVRSession_2018-03-07_14-38-06_Camera1',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


# CHOOSE THE DATA PERCENTAGE SPLIT, must sum to 1: TRAIN / DEV / TEST
data_split = [.8, 0.1, .1] 


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)
    print('starting')
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.test = 0;
    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    #assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"
    
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    
    
    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = data_dir #os.path.join(data_dir, "train_signs")
    dev_data_dir = data_dir #os.path.join(data_dir, "dev_signs")
    path_to_labeling_dict = os.path.join('data/JDP/CS230_Visuomotor_Learning/parser/data', "cam1dict.p")
    
    
    all_filenames = [f for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg') and not f.endswith('Depth.jpg')]
#     TODO: Shuffle array, probably a good idea, uncomment this later.
#     random.shuffle(all_filenames)

    num_files = len(all_filenames)
    train_filenames = all_filenames[:int(num_files*data_split[0])]
    eval_filenames = all_filenames[int(num_files*data_split[0]) + 1: int(num_files*data_split[0])+int(num_files*data_split[1])]
    
#     TODO: Pipe test_filenames to the appropriate place
    test_filenames = all_filenames[int(num_files*data_split[0])+int(num_files*data_split[1]) + 1 :]
    
    
    
    # Get the filenames from the train and dev sets
#     train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
#                        if f.endswith('.jpg')]
#     eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
#                       if f.endswith('.jpg')]
    
    # TODO: get the REAL labels here, an np.array(55, 1).
    image_to_labels_dict = pickle.load( open(path_to_labeling_dict, "rb") )

    # Labels will be between 0 and 54 included (55 classes in total)
    train_labels = np.zeros((len(train_filenames), params.num_labels))
    eval_labels = np.zeros((len(eval_filenames), params.num_labels))
    for i in range(len(train_filenames)):
        img_name = train_filenames[i]
        train_labels[i][:] = image_to_labels_dict[basename(img_name)];
    for i in range(len(eval_filenames)):
        img_name = eval_filenames[i]
        eval_labels[i][:] = image_to_labels_dict[basename(img_name)];
     

    pos_weights = 0;
    neg_weights = 0;
#     print('finished labeling khot')
#     train_labels = [image_to_labels_dict[basename(img_name)] for img_name in train_filenames]
#     eval_labels = [image_to_labels_dict[basename(img_name)] for img_name in eval_filenames]
    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)
    # Create the two iterators over the two datasets
#     print('yea boi')
    train_inputs = input_fn(True, train_filenames, train_labels, params)
#     print('yea boi')
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)
#     print('didn"t fail on input_fn creation')
    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
