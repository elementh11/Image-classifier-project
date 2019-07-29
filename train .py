#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Adrian P.
# DATE CREATED: 4/4/2019                                 
# REVISED DATE: 
# PURPOSE:  train a new network on a dataset and save the model as a checkpoint


# Imports modules
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from workspace_utils import get_input_args, save_checkpoint

# Main program function defined below
def main():
    in_arg = get_input_args()

    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Data batching with torchvision's DataLoader --> changed to single-line
    dataloader = [torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
                  torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True),
                  torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)]

    # Load pretrained model and retrieve its # of feature output units
    model = getattr(models, in_arg.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Define model classifier
    if in_arg.arch[:3] == 'vgg': # --> use this classifier for vgg networks only
        in_units = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(in_units, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(in_arg.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    elif:
        classifier = nn.Sequential(nn.Linear(1024, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_arg.hidden_units, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(in_arg.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(in_arg.learning_rate))    
    
    if in_arg.gpu == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device);

    # Train the model
    #with active_session():
    epochs = int(in_arg.epochs)
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloader[0]:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader[1]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch/step {epoch+1}/{steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloader[1]):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloader[1]):.3f}")
                running_loss = 0
                model.train()

    # Save the trained model
    model.class_to_idx = datasets.ImageFolder(train_dir).class_to_idx
    save_checkpoint(model, optimizer, in_arg)

                
                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
