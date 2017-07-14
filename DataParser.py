__author__ = 'charlie modified by Lilly Thomas'
import numpy as np
import os
import random
#from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import argparse

import TensorflowUtils as utils

#data_dir = "./data/"

def read_dataset(data_dir):

    result = create_image_lists(data_dir)
    print ("Creating Image Lists ...")
    
    training_records = result['training']
    validation_records = result['validation']

    return training_records, validation_records

def read_dataset_else(data_dir):

    result = create_image_list_else(data_dir)
    print ("Creating Image List ...")

    image_records = result['images']

    return image_records

def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

def create_image_list_else(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['images']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                if os.path.exists(f):
                    record = {'image': f, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("File not found for %s - Skipping" % filename)

        random.shuffle(image_list)
        no_of_images = len(image_list)
        print ('No. of files: %d' % (no_of_images))

    return image_list
