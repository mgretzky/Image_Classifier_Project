# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

# Define parser (gets inputs prom Command Line)
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'directory with the images')
parser.add_argument('--arch', type = str, default = 'densenet121', help = 'choose pre-trained model - default densenet121')
parser.add_argument('--learning_rate', type = float, default = 0.001,  help = 'learning rate of the model - default 0.001')
parser.add_argument('--hidden_layers', type = int, default  = 512,  help = 'size of hidden layers in the classifier')
parser.add_argument('--dropout', type = float, default = 0.2, help = 'number as a float - default 0.2')
parser.add_argument('--epochs', type = int, default = 2, help = 'number of classifier training cycles')
parser.add_argument('--gpu', type=bool, default='True', help="True: gpu, False: cpu - default True")

# Parse arguments
args = parser.parse_args()
data_dir = args.data_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_layers = args.hidden_layers
dout = args.dropout
epochs = args.epochs
gpu = args.gpu

def get_data(data_dir):
    # Define transforms for training, validation and testing data sets
    # Transforms for all data sets: resize, crop, transform into tensor, normalise color channels
    # Additional augmentation transforms for training data set: random rotation, random flipping
    data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    
    # Load data sets with ImageFolder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
    
    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data

trainloader, validloader, testloader, train_data, valid_data, test_data = get_data(data_dir)

def set_model(arch):
    # Load the pre-trained model
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        print('Using densenet121')
    else:
        model = models.densenet121(pretrained=True)
        print('Use densenet or vgg16 only. Defaulting to densenet121')
        
    return model

def set_classifier(model, hidden_units):
    # Define classifier to work with the pre-trained model
    input_layer = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_layer, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=dout)),
                            ('fc2', nn.Linear(hidden_units, 256)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=dout)),
                            ('fc3', nn.Linear(256, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    return model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

def train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu):
    # Define model training loop
    if type(epochs) == type(None):
        epochs=10
        print('Set epochs to 10')
    
    steps=0
    
    # Move model to gpu if available
    if gpu == True:
        model.to('cuda')
        
    for ii in range(epochs):
        steps += 1
        train_loss = 0
        
        # Move images and labels to the gpu if available
        for images, labels in trainloader:
            if gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            logprob = model.forward(images)
            loss = criterion(logprob, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            
            # Put model in evaluation mode
            model.eval()
            
            with torch.no_grad():
            
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logprob = model.forward(images)
                
                    loss = criterion(logprob, labels)
                    valid_loss += loss.item()
                
                    #Calculate accuracy
                    prob = torch.exp(logprob)
                    top_prob, top_class = prob.topk(1, dim=1)
                    comparison = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(comparison.type(torch.FloatTensor)).item()
            
            print(f"Epoch {ii+1}/{epochs}.. "
                f"Train loss: {train_loss/len(trainloader):.3f}.. "
                f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                f"Validation accuracy: {accuracy/len(validloader):.3f}")
            
            # Put model back in training mode
            model.train()
        
train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu)

def test_model(model, testloader, gpu):
    # Define model testing function
    if gpu == True:
        model.to('cuda')
        
    test_loss = 0
    accuracy = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for images, labels in testloader:
            
            if gpu == True:
                images, labels = images.to('cuda'), labels.to('cuda')
                
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
                
            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")

test_model(model, testloader, gpu)

# Save checkpoint
model.class_to_idx = train_data.class_to_idx

state = {
    'arch': 'densenet121',
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'optimizer': optimizer,
    'optimizer_dict' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
    }

torch.save(state, 'checkpoint.pth')
    
