import argparse
import functions_train as f_train
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
import torch
import json

# Arguments
parser = argparse.ArgumentParser(description='Build and Train the Model.')
parser.add_argument("--data_dir", default='flowers', help='Main directory')
parser.add_argument("--save_dir", default='./checkpoint.pth')
parser.add_argument("--arch", default='vgg16', help=' Model architecture')
parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate')
parser.add_argument("--hidden_units", type=int, default=5000, help='Hidden units')
parser.add_argument("--epochs", type=int, default=18, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Choose this argument if you want to use GPU')
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Data files
train_transforms, valid_transforms, test_transforms = f_train.data_transformer()
train_dataset, valid_dataset, test_dataset = f_train.load_datasets(train_transforms, train_dir, valid_transforms,
                                                                   valid_dir,
                                                                   test_transforms, test_dir)
train_loader, valid_loader, test_loader = f_train.data_loader(train_dataset, valid_dataset, test_dataset)

# Architecture
if args.arch == 'vgg16':
    print("training using vgg")
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'densenet121':
    print("training using densenet")
    model = models.densenet121(pretrained=True)
    input_size = 1024
else:
    print("training using alexnet")
    model = models.alexnet(pretrained=True)
    input_size = 9216

for parameter in model.parameters():
    parameter.requires_grad = False

# hidden_units
if args.hidden_units == 5000:
    hidden_units = 5000
else:
    hidden_units = args.hidden_units

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

# Initial_Classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(hidden_units, cat_to_name)),
    ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

# Classifier Training
f_train.train_classifier(model, optimizer, criterion, train_loader, valid_loader, args.epochs, device)

# Testing My Network
f_train.testing(model, test_loader, criterion)

# Save Checkpoint
f_train.save_checkpoint(model, train_dataset, args.arch, args.hidden_units, args.learning_rate, args.epochs, input_size)
