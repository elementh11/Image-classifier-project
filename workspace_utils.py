import os
import copy
import json
import signal
from contextlib import contextmanager

import requests
import argparse
import torch
from torchvision import datasets, transforms, models


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'flowers', 
                    help = 'path to the folder of flowers images')
    parser.add_argument('--image_path', type = str, default = 'flowers/train/10/image_07086.jpg', 
                    help = 'path to image for inference')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                    help = 'directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint_vgg16.pth', 
                    help = 'path to checkpoint')
    parser.add_argument('--labels_path', type = str, default = 'cat_to_name.json', 
                    help = 'path to labels file')
    parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'vgg19'],
                    help = 'NN model architecture')
    parser.add_argument('--gpu', type = str, default = 'gpu', 
                    help = 'use GPU / CPU')
    parser.add_argument('--topk', default='5')
    parser.add_argument('--learning_rate', default='0.001')
    parser.add_argument('--hidden_units', default='256')
    parser.add_argument('--epochs', type=int, default='1')
    
    return parser.parse_args()

def save_checkpoint(model, optimizer, in_arg):
checkpoint = {'arch': 'vgg16',
              'classifier': model.classifier,
              'epochs': 1,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, in_arg.save_dir)    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def get_labels(filepath):
    with open(filepath) as f:
        cat_to_name = json.load(f)
    return cat_to_name

def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable

        
        
        
        
        
        