import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import argparse
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store')

parser.add_argument('checkpoint', action='store')

parser.add_argument('--top_k ', action='store',
                    dest='top_k',
                    default=5,
                    type=int)

parser.add_argument('--category_names ', action='store',
                    dest='category_names',
                    default='cat_to_name.json')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu')

results = parser.parse_args()

def get_args():
    return results

args=get_args()

def gpu_check():
    print("PyTorch version is {}".format(torch.__version__))
    gpu_check = torch.cuda.is_available()

    if gpu_check:
        print("GPU Device available.")
    else:
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')

    return gpu_check

	
def loadcheckpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    if checkpoint['arch'] == 'vgg16' :
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19' :
        model=models.vgg19(pretrained=True)
    classifier = nn.Sequential(nn.Linear(checkpoint['input_features'], checkpoint['hidden_units']),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(checkpoint['hidden_units'] ,1000),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(1000, checkpoint['output']),
                           nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['opt_state'])
    
    return model, optimizer

# load model
model, optimizer = loadcheckpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #size = 256, 256
    im = Image.open (image) #loading image
    width, height = im.size #original size
    #proportion = width/ float (height) #to keep aspect ratio
    
    if width > height: 
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else: 
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)
        
    
    width, height = im.size #new size of im
    #crop 224x224 in the center
    reduce = 224
    left = (width - reduce)/2 
    top = (height - reduce)/2
    right = left + 224 
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))
    
    #preparing numpy array
    np_image = np.array (im)/255 #to make values from 0 to 1
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    
    np_image= np_image.transpose ((2,0,1))
    #np_image.transpose (1,2,0)
    return np_image



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    np_img = process_image(image_path)
    model.eval()
    img = torch.zeros(1, 3, 224, 224)
    img[0] = torch.from_numpy(np_img)
    if args.gpu and gpu_check :
        model.to('cuda')
        img = img.to('cuda')

    with torch.no_grad():
        output = model(img)

    output = output.to('cpu')
    ps = torch.exp(output).topk(topk)
    p, c = ps
    idx_to_class = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
    c = [idx_to_class[i] for i in c[0].numpy()]
    
    
    return p[0].numpy(), c

probs, classes = predict(args.image_path, model, topk=args.top_k)

names = classes
if args.category_names:
    if os.path.isfile(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = np.array([cat_to_name[i] for i in classes])

for result in zip(names, probs):
    name, prob = result
    print(name, prob)

