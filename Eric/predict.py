import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
import torch
import time
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
import json
import argparse



#data_dir  = 'flowers/'             # You dont want data here, you will not used datasets during the prediction,
                                    # You will use your checkpoint file which included trained weights and biases of the model
                                    # then you will use an input image which needs to be predicted all set with the arparser
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir  = data_dir + '/test'


with open('cat_to_name.json', 'r') as f: # This is hardcoded, but must be an argument from the argparser again
    flower = json.load(f)

#print(json.dumps(flower, sort_keys=True, indent=2, separators=(',', ': ')))

"""

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
   
    model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
         param.requires_grad = False
 
    model.class_to_idx = checkpoint[8]
    
    classifier = nn.Sequential(OrderedDict([('hidden1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('hidden2', nn.Linear(4096, len(flower))),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.load_state_dict(checkpoint[7])
    
    return model
"""    


def load_checkpoint(path):

    print('Loading model from checkpoint...')

    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(path)

    if 'hidden_units' in checkpoint:
        hidden_units = checkpoint['hidden_units']
    else:
        hidden_units = 1000

    if checkpoint['model'] == "Densenet":
        model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
    else:
        model = models.vgg19(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096, hidden_units)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier


    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model




model = load_checkpoint('checkpoint.pth')
model
 

    

    
   