import argparse
import functions_train as func
from torchvision import models
from torch import nn, optim
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Build and Train the Model.')
parser.add_argument("--data_dir", default='flowers', help='Main directory')
parser.add_argument("--save_dir", default='./checkpoint.pth')
parser.add_argument("--arch", default='vgg16', help=' Model architecture')
parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate')
parser.add_argument("--hidden_units")
parser.add_argument("--epochs", type=int, default=18, help='Number of epochs')
parser.add_argument("--gpu", type=bool, default=False)
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Data files
train_transforms, valid_transforms, test_transforms = func.data_transformer()
train_dataset, valid_dataset, test_dataset = func.load_datasets(train_transforms, train_dir, valid_transforms,
                                                                valid_dir,
                                                                test_transforms, test_dir)
train_loader, valid_loader, test_loader = func.data_loader(train_dataset, valid_dataset, test_dataset)

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

# Initial_Classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, 5000)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(5000, 120)),
    ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

# Classifier Training
func.train_classifier(model, optimizer, criterion, train_loader, valid_loader, args.epochs, args.gpu)

# Testing My Network
func.testing(model, test_loader, criterion)

# Save Checkpoint
func.save_checkpoint(model, train_dataset)
