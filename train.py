import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    default='./checkpoint.pth',
                    help='Set checkpoint dir')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    default='vgg16',
                    help='Choose architecture')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.0001,
                    type=float,
                    help='Set learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    default=4096,
                    type=int,
                    help='Set hidden units number')

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    default=4,
                    type=int,
                    help='Set epochs')

parser.add_argument('--gpu', action='store_true',
                    default = False,
                    dest='gpu',
                    help='Use GPU for training')

results = parser.parse_args()

def get_args():
    return results

	
def gpu_check():
    print("PyTorch version is {}".format(torch.__version__))
    gpu_check = torch.cuda.is_available()

    if gpu_check:
        print("GPU Device available.")
    else:
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')

    return gpu_check


args = get_args()

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.RandomRotation(225),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.RandomRotation(225),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder 
training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
training_loader=torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
validation_loader=torch.utils.data.DataLoader(validation_data, batch_size=64)
testing_loader=torch.utils.data.DataLoader(testing_data, batch_size=64)




if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'vgg19':
    model = models.vgg19(pretrained=True)

	

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(args.hidden_units, 1000),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(1000, 102),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

criterion = nn.CrossEntropyLoss()
# Define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
# Define deep learning method

steps = 0
print_every = 30


def validation(model, testloader, criterion):
    if args.gpu and gpu_check:
        model.to('cuda')
    accuracy = 0
    test_loss = 0
    for inputs, labels in testloader:
       
        if args.gpu and gpu_check:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        return test_loss, accuracy

if args.gpu and gpu_check:
    model.to('cuda')

for epoch in range(args.epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(training_loader):
        steps += 1
        # Move input and label tensors to the default device
        if args.gpu and gpu_check:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            #validation_loss = 0
            #accuracy = 0
            model.eval()
        
            with torch.no_grad():
                validation_loss, accuracy = validation(model, validation_loader, criterion)
                    
            print("Epoch: {}/{}.. ".format(epoch+1, args.epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
            
            running_loss = 0
            model.train()

# TODO: Do validation on the test set
model.eval()
    
with torch.no_grad():
    _, accuracy = validation(model, testing_loader, criterion)
                
print("Testing Accuracy: {:.2f}%".format(accuracy*100/len(testing_loader)))

# TODO: Save the checkpoint 
model.class_to_idx = training_data.class_to_idx
torch.save({
            'class_to_idx': model.class_to_idx,
            'arch' : args.arch,
            'input_features' : 25088 ,
            'state_dict': model.state_dict(),
            'hidden_units': args.hidden_units,
            'output' :102,
			'epochs' :4,
			'learning_rate':0.001,
            'opt_state': optimizer.state_dict()
        }, args.save_dir )