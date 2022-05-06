# Imports here
import torch
from torch import nn, optim
from torchvision import models

from functions import *
from utility import *

def main():
    print('### Starting Training Script Execution ###')
    # get the input arguments
    in_arg = get_train_args()
    print('### Obtained Script Arguments ###')
    
    # TODO: Make TESTS here to ensure that the arguments supplied by the user matches 
    # what I want; e.g: hidden units value is equal or greater than 512
    print('### Running Argument Tests ###')
    if in_arg.hidden_units < 512:
        print('Exiting training. Network hidden units is less than {}'.format(in_arg.hidden_units))
        quit()
    print('### Argument Tests Passed ###')

    # load the datasets
    train_data, train_loader, valid_loader, test_loader = load_data(in_arg.data_dir)
    print('### Loaded datasets ###')
  
    # select a device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() and in_arg.gpu is True else 'cpu')
    print('### Set Device to {} ###'.format(device))
    
    # choose and load a pre-trained network model architecture
    # need to find a way to make this work
    model = eval('models.{}(pretrained=True)'.format(in_arg.arch))
    print('### Downloaded Pretrained Model {} ###'.format(model))

    # define a new, untrained feed-forward network as a classifier
    # using ReLU activations and dropout
    # turn off gradients
    for param in model.parameters():
        param.requires_grad = False
        
    # get model input layers
    input_features = model.classifier[0].in_features
        
    model.classifier = nn.Sequential(nn.Linear(input_features, in_arg.hidden_units, bias=True),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(in_arg.hidden_units, 512, bias=True),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512, 102, bias=True),
                               nn.LogSoftmax(dim=1)
                              )
    print('### Updated Model Classifier ###')
    # 8192
    # set criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    # Add model to device
    model.to(device)
    print('### Set Criterion, Optimizer, and added Model to Device ###')
    
    # train the model
    trained_model = train_model(in_arg.epochs, train_loader, valid_loader, device, model, optimizer, criterion)
    print('### Trained Model Successfully ###')
    # test the model accuracy
    test_model(trained_model, test_loader, device)
    print('### Tested Trained Model Succesfully ###')
    # save the model checkpoint
    checkpoint = save_checkpoint(trained_model, train_data, in_arg.save_dir)   
    print('### Saved Trained Model Checkpoint ###')
    print('### Exiting Training Script Execution ###')

if __name__ == '__main__': main()
    
# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 --save_dir save_directory --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu