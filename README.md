# Image-classifier-project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, an image classifier built with PyTorch is added to a pretraind VGG16 net, then code is developed to convert it into a command line application.

* The classifier, composed of one hidden linear layer, with Relu activation and dropout, is trained on a flower dataset to a 0.75 testing accuracy.

* The class prediction uses top-k prediction, here the top 5 most probable classes.

* The training and inference are coded into `train.py` and `predict.py` files, which are called using command line with arguments built with the `argparse` module.
