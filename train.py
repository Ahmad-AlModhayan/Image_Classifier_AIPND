import argparse
import functions_train as f_t
from torchvision import transforms, datasets, models
import torch
from torch import nn, optim
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Build and Train the Model.')
parser.add_argument("data_dir", default='./flowers', help='Main directory')
parser.add_argument("--save_dir", default='./checkpoint.pth')
parser.add_argument("--arch", default='vgg16', help=' Model architecture')
parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate')
parser.add_argument("--hidden_units")
parser.add_argument("--epochs", type=int, default=18, help='Number of epochs')
parser.add_argument("--gpu")
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_loader = f_t.train_data_load(train_dir)
valid_loader = f_t.valid_test_data_load(valid_dir)
test_loader = f_t.valid_test_data_load(test_dir)

# Architecture
if args.arch == 'vgg16':
    input_size = 25088
    model = models.vgg16(pretrained=True)
else:
    print("This Model just supports vgg16 training")

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
f_t.train_classifier(model, optimizer, criterion, train_loader, valid_loader)

