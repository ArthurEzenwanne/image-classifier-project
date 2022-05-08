# Imports here
import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets

from time import time
from math import ceil

from utility import *

def get_train_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the training program from a terminal window. This function uses Python's 
    argparse module to create and define these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Directory to load datasets from as --data_dir with default value 'flowers'
      2. Checkpoint directory as --save_dir with default value 'checkpoint_dir'
      3. Model architecture as --arch with default value 'vgg11'
      4. Training learning rate as --learning_rate with default value 0.001
      5. Number of hidden units in the model as --hidden_units with default value 512
      6. Number of training epochs as --epochs with default 2
      7. Device to use for training as --gpu with default True
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type=str, default='flowers', help='directory to load dataset from')
    parser.add_argument('--save_dir', type=str, default='checkpoint_dir', help='directory path to save checkpoint files')
    parser.add_argument('--arch', type=str, default='vgg11', help='CNN model to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training model')
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU for training model')
        
    return parser.parse_args()

def get_predict_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when
    they run the prediction program from a terminal window. This function uses Python's 
    argparse module to create and define these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Path to load image from as --input with default value ''
      2. Path to checkpoint file as --checkpoint with default value 'checkpoint_dir/checkpoint.pth'
      3. Top probabilities values to return as --top_k with default value 3
      4. Labels to real names mapping file as --category_names with default value 'cat_to_name.json'
      5. Device to use for training as --gpu with default True
      6. Model architecture as --arch with default value 'vgg11'

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 6 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--input', type=str, default='flowers/valid/100/image_07904.jpg', help='image file to predict')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_dir/checkpoint.pth', help='file path to checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Number of probabilities to display')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='label mappings file')
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU for training model')
    parser.add_argument('--arch', type=str, default='vgg11', help='CNN model to use')
        
    return parser.parse_args()

def save_checkpoint(model, idx_data, save_dir):
    """
    Saves a PyTorch checkpoint model object to directory
    Parameters:
     model - PyTorch trained model to save on disk
     save_dir - Directory path to save model checkpoint
    Returns:
     The directory path to the model checkpoint
    """
    # add checkpoint information    
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': idx_data.class_to_idx
                 }
    # make the directory
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
        
    # save the checkpoint
    torch.save(checkpoint, save_dir +'/'+'checkpoint.pth')
    checkpoint_file = save_dir +'/checkpoint.pth'
    return checkpoint_file
