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
def load_checkpoint(path, arch):
    """
    Loads a PyTorch checkpoint file from directory
    Parameters:
     path - Directory path to checkpoint file
    Returns:
     PyTorch model
    """
    # ensure the model is loaded by GPU even if the GPU is off (SO)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    checkpoint = torch.load(path, map_location=map_location)
    
    # load the arch of the model used in the training phase
    model = eval('models.{}(pretrained=True)'.format(arch))
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        # disable use of gradients since we are predicting at this point
        param.requires_grad = False
    
    return model

def train_model(epochs, train_loader, valid_loader, device, model, optimizer, criterion):
    """
    Trains a PyTorch network model object
    Parameters:
     epochs - number ot training epochs
     train_loader - iterator object containing the training dataset
     valid_loader - iterator object containing the validation dataset
     device - device for training can be GPU or CPU
     model - PyTorch pretrained model to use in training
     optimizer - optimizer object to use in training eg Adam
     criterion - error loss criterion metric eg NLLLoss
    Returns:
     The trained network Pytorch model
    """
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    train_steps = 0
    train_loss = 0
    iteration = 5
    start_time = time()
    print('###################### Start Model Training ######################')
    for epoch in range(epochs):
        # Training loop
        for images, labels in train_loader:
            train_steps += 1

            # Add images and labels to device
            images, labels = images.to(device), labels.to(device)

            # zero gradients before each loop pass, move forward, loss, backwards, step
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Testing accurracy on validation data
            if train_steps % iteration == 0:
                valid_loss = 0
                valid_accuracy = 0
                # Place model in evaluation mode
                model.eval()

                # speed up by removing gradients calculations
                with torch.no_grad():                
                    for images, labels in valid_loader:                
                        # Add images and labels to device
                        images, labels = images.to(device), labels.to(device)

                        # Calculating loss on validation set
                        logps_valid = model.forward(images)
                        loss = criterion(logps_valid, labels)
                        valid_loss += loss.item()

                        # Calculating accuracy on validation set
                        proba_valid = torch.exp(logps_valid)
                        top_proba, top_class = proba_valid.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                # Display results
                print('Epoch {}/{} ...'.format(epoch+1, epochs),
                      'Training Loss: {:3f} ...'.format(train_loss/len(train_loader)),
                      'Validation Loss: {:3f} ...'.format(valid_loss/len(valid_loader)),
                      'Validation Accuracy: {:3f} '.format(valid_accuracy/len(valid_loader))
                     )
                running_loss = 0
                model.train()  
                
    print('###################### End Model Training ######################')
    print('Elapsed time: {}'.format((time() - start_time)//60))
    
    # return trained model
    return model
      
# test the above model on test images
# using the function from https://github.com/chauhan-nitin/Udacity-ImageClassifier/blob/master/train.py
# this is strictly not needed - just a nice to have
def test_model(model, test_loader, device):
    """
    Test the trained model on some never before seen test data
    Parameters:
     model - PyTorch trained model 
     test_loader - iterator object containing the testing dataset
     device - device for training can be GPU or CPU
    Returns:
     None - Only prints results to screen
    """
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct/total))
    
def predict(image, model, topk, device, category_names):
    """
    Given an input image predicts the results
    Parameters:
     image - processed image file path to input image
     model - PyTorch trained model
     topk - number of top probalilities to return
     device - device for training can be GPU or CPU
     category_names - path to flower classes to names file
    Returns:
     proba - list of top probabilities
     labels - list of  top labels
     flowers - list of top flower names
    """
    img = image.unsqueeze(0).float()    
    model.to(device)    
    model.eval()                # evaluation mode
    
    with torch.no_grad():
        ps = model(img)
    proba = F.softmax(ps.data, dim=1)      # use Functional and softmax to do the hard lifting
    
    # unpack proba topk into probabilities and labels
    proba, labels = proba.topk(topk)
    
    # convert from tensor objects to numpy arrays
    proba = proba.to('cpu').numpy().squeeze()
    labels = labels.to('cpu').numpy().squeeze()
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    labels = [idx_to_class[i] for i in labels]
    flowers = [category_names[i] for i in labels]

    return proba, labels, flowers
    
# using the function from https://github.com/chauhan-nitin/Udacity-ImageClassifier/blob/master/predict.py    
def get_results(ps, images):
    """
    Prints the output of the prediction and the top probabilities
    Parameters:
     ps - list of the top probabilities
     images - list of the top flower names
    Returns:
     None - Only displays the results
    """
    for i, j in enumerate(zip(ps, images)):
        print('Rank {}:'.format(i+1),
              'Flower: {}, Likelihood: {}%'.format(j[1], ceil(j[0]*100)))
    