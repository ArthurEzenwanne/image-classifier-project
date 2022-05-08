import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets
from PIL import Image

from functions import *

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    Parameters:
     image_path - Name or full directory path to image
    Returns:
     A Tensor object
    """
    # process a PIL image for use in a PyTorch model
    # could also have used transforms here. 
    # feel it would be a better approach even
    # get the image using the image_path argument, get image size
    img = Image.open(image_path)
    width, height = img.size
    
    # resize the image to 256 
    if width < height:
        n_height = int(256*(height/width))
        img = img.resize((256, n_height))
    else:
        n_width = int(256*(width/height))
        img = img.resize((n_width, 256))
        
    # center crop image
    # don't really understand these implementation
    # should have used transforms that I understand
    c_width, c_height = 224, 224
    c_left = int((img.width - c_width)/2) 
    c_top = int((img.height - c_height)/2) 
    c_right = int((img.width - c_width)/2 + c_width) 
    c_bottom = int((img.height - c_height)/2 + c_height)
    img = img.crop((c_left, c_top, c_right, c_bottom))
    
    # convert image to array    
    np_img = np.array(img)
    
    # normalize RGB values in range 0:1
    # divide all values by 255 because imshow() expects integers (0:1)
    np_img = np_img/255
    mu = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # normalize by subtracting mu from each color channel & divide by std
    np_img = (np_img - mu)/std
    # transpose the order so that the color channel is now at the 1st index
    np_img = np_img.transpose((2, 0, 1)) 
    
    # convert processed image into a tensor
    tensor_img = torch.from_numpy(np_img)
        
    return tensor_img


def load_data(data_dir):
    """
    Loads the training, validation, and test data in the required format.
    Parameters:
     None
    Returns:
     The iterator train_loader, valid_loader, and test_loader objects
    """
    # define transforms for the training, validation, and testing sets    
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # data transforms for the training phase
    data_transforms_train = transforms.Compose([transforms.RandomRotation(15),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    
    # data transforms for the validation and test phases
    data_transforms_valid = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms_valid)

    # using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=96, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=96, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=96, shuffle=True)
    
    return train_data, train_loader, valid_loader, test_loader

# no longer needed 
def data_loader(data, phase='training'):
    """
    Iterator object to load training, validation, and test datasets
    Parameters:
     data - dataset to use for training
     phase - network phase - training, validation, or testing
    Returns:
     dataLoader iterator object
    """
    if phase == 'training':
        train_loader = torch.utils.data.DataLoader(data, 96, True)
        return train_loader
    elif phase == 'validation':
        valid_loader = torch.utils.data.DataLoader(data, 96, True)
        return valid_loader
    elif phase == 'testing':
        test_loader = torch.utils.data.DataLoader(data, 96, True)
        return test_loader

def imshow(image, ax=None, title=None):
    """
    Accepts an image and displays the image on a chart.
    Parameters:
     Image - Full directory path to the image 
     ax - The axis on which the plot the image. Default - None
     Title - Title of the chart. Default - None
    Returns:
     The transformed image in a matplotlib chart
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

