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