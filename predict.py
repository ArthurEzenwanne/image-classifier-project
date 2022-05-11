# Imports here
import json
import torch
from torch import nn, optim
from torchvision import models

from functions import *
from utility import *

def main():
    print('### Starting Prediction Script Execution ###')
    # get the input arguments
    in_arg_predict = get_predict_args()
    print('### Obtained Script Arguments ###')
    
    # label mappings
    with open(in_arg_predict.category_names, 'r') as f:
        cat_to_name = json.load(f)
    print('### Mapped Labels ###')
    
    # select a device
    device = torch.device('cuda' if torch.cuda.is_available() and in_arg_predict.gpu is True else 'cpu')
    print('### Set Device to {} ###'.format(device))
    
    model = load_checkpoint(in_arg_predict.checkpoint, in_arg_predict.arch)
    print('### Loaded Saved Checkpoint file ###')
    
    img = process_image(in_arg_predict.input)
    # add processed_image to device
    img = img.to(device)
    print('### Processed Image File is added to device  ###')    
    
    proba, labels, flowers = predict(img, model, in_arg_predict.top_k, device, cat_to_name)
    print('### Prediction on File Completed ###') 
    
    print('############# Prediction Results Output Printing #############') 
    get_results(proba, flowers)
    print('############# Prediction Results Output Printing #############')
    print('### Exiting Prediction Script Execution ###')
    
if __name__ == '__main__': main()
