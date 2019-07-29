#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Adrian P.
# DATE CREATED: 4/5/2019                                 
# REVISED DATE: 
# PURPOSE:  take the path to an image and a checkpoint, then return the top K most probably classes for that image


# Imports modules
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from workspace_utils import get_input_args, load_checkpoint, get_labels

# Process a PIL image for use in a PyTorch model
def process_image(image):
    im = Image.open(image)
    width, height = im.size
    size = 256, 256

    # Resize keeping the aspect-ratio
    if width > height:
        ratio = float(width) / float(height)
        newwidth = int(ratio * size[0])
        size = newwidth, 256

    else: 
        ratio = float(height) / float(width)
        newheight = int(ratio * size[0])
        size = 256, newheight
    im.thumbnail(size)

    # Center-crop to 224 x 224
    im = im.crop(((size[0] - 224)/2,(size[1] - 224)/2,(size[0] + 224)/2,(size[1] + 224)/2))
    
    # Normalize colors
    im = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])   
    im = ((im - mean) / std)
    im = np.transpose(im, (2, 0, 1))                                                           
    
    return im

# Predict the class of an image using a trained model
def predict(in_arg):
    img = process_image(in_arg.image_path)
    img = torch.from_numpy(img)
    img = img.float()
    img.unsqueeze_(0)

    model = load_checkpoint(in_arg.checkpoint_path)
    
    if in_arg.gpu == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device);
    img = img.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img)
    ps = torch.exp(output)
    probs, classes = ps.topk(int(in_arg.topk), dim=1)
    
    return probs, classes



# Main program function defined below
def main():
    in_arg = get_input_args()
    
    probs, classes = predict(in_arg)

    # convert from cuda tensor to numpy
    probs = probs.cpu()
    probs = probs.numpy()[0]
    classes = classes.cpu()
    classes = classes.numpy()[0]
    cat_to_name = get_labels(in_arg.labels_path)
    labels=[]
    for idx in classes:
            labels.append(cat_to_name[str(idx)])

    for i in range(int(in_arg.topk)):
        print('{:<20}  {:>.3f}'.format(labels[i], probs[i]))

    
    
    # Call to main function to run the program
if __name__ == "__main__":
    main()


