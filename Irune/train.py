import torch
import time
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import argparse
import json


from utility import load_data


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', help='directory containing images')
    parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory' )
    parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg19')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
    parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=4)
    parser.add_argument('--learning_rate', action='store', help='which learning rate to start with', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=500)
    parser.add_argument('--output_size', action='store', help='# of classes to output', type=int, default=102)
    
    return parser.parse_args()



def main():
  
    in_arg = get_input_args()
    
    start_time = time.time()
    
    image_datasets, dataloaders = load_data(in_arg.data_dir)
    
    model = get_model(in_arg.arch)
        
    model = load_model(model, in_arg.arch, in_arg.hidden_units, in_arg.lr) # HOLDING OPTIONAL ARGUMENTS

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.lr)
    
    
    train(model, in_arg.epochs, in_arg.lr, criterion, optimizer, dataloaders['training'], dataloaders['validation'],in_arg.gpu, start_time)
   
    print(f"Time to train and validate model: {(time.time() - start_time):.3f} seconds")

    save_checkpoint(in_arg.save_dir, model, optimizer, in_arg.epochs, in_arg.arch, image_datasets, in_arg.lr)

def get_model(arch):
    if arch == 'densenet121': #
        model = models.densenet121(pretrained = True)
    return model



def load_model(arch, hidden_units, lr):

    if arch == 'densenet121':    
        model=models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        classifier= nn.Sequential(nn.Linear(1024,hidden_units), # VARIABLE EVENTUALLY WHEN FUNCTION CALLED HOLDS OPTIONAL ARGUMENT
                              nn.ReLU(),
                              nn.Linear(hidden_units, 102),
                              nn.LogSoftmax(dim=1))
        model.classifier = classifier
    

    elif arch == 'VGG': # INSTRUCTIONS ASK FOR LETTING THE USER CHOOSE BETWEEN AT LEAST 2 ARCHITECTURES, THIS IS A SUGGESTION
            model = models.vgg16(pretrained=True)
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


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) # LEARNING RATE WAS HARDCODED
    model.to(device)
    return model

def train(model, epochs, criterion, optimizer, trainloader, vloader, gpu):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu') # THIS LINE WAS CORRECTED AS GPU MUST HAVE BEEN A REQUIREMENT,TOO, SINCE THE USER MIGHT ALSO OPT TO TRAIN ON CPU WHICH MUST BE POSSIBLE
    model.train() # ALSO THE DEVICE LINE ABOVE POSSIBLY MUST BE PLACED DIFFERENT LOCATIONS IF NOT SET GLOBALLY
    epochs=epochs # EPOCHS WHERE HARDCODED, BUST MUST ALSO BE ACCESSIBLE FROM THE COMMAND LINE
    steps=0
    running_loss=0
    print_every=20
    print(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels =inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if steps % print_every==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in  vloader:
                        inputs, labels=inputs.to(device), labels.to(device)
                        logps =model.forward(inputs)
                        batch_loss=criterion(logps, labels)
                        test_loss+=batch_loss.item()
                        ps=torch.exp(logps)
                        top_p, top_class=ps.topk(1, dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Validation loss:{test_loss/len(vloader):.3f}.."
                      f"Validation accuracy: {accuracy/len(vloader):.3f}")  

def validate(model, criterion, data_loader):
    correct = 0
    total = 0
    model.to(device) # DEVICE MUST BE SET EVERYWHERE, CUDA CANNOT BE HARDCODED
    model.eval()
    with torch.no_grad():
        for data in dataloaders[2]: 
            image, label = data
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
        return ('Test Accuracy: %d %%' % (100 * correct / total))
    

def save_checkpoint(save_dir, model, optimizer, epochs, arch, image_datasets, lr):
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint = {'output_size' : 102,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'arch': arch, # ARGUMENT CHOSEN FROM FUNCTION CALL - PREDICT.PY LOAD FUNCTION MUST BE ADAPTED TO SAVE FUNCTION
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': classifier.state_dict()}
    torch.save(checkpoint, save_dir)
    
       
   


if __name__ == "__main__":
    main()































