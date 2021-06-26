# Imports
import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import torchvision.models as models
import json
from collections import OrderedDict

# the command line input
parser = argparse.ArgumentParser(description="Train.py")
parser.add_argument("--save_dir", action="store", help="Save_Directory")
parser.add_argument("--arch", action="store", help="Architecture")
parser.add_argument("--learning_rate", type=float, action="store", help="learning_rate")
parser.add_argument("--hidden_units", type=int, action="store", help="hidden_units")
parser.add_argument("--epochs", type=int, action="store", help="Epochs")
parser.add_argument("--gpu", action="store", help="GPU")
args = parser.parse_args()

save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# data_dir
data_dir = './flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

valid_test_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

# Image_Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms),
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms),
test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

# Data_loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Label_mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load pretrained vgg16 Model
model = models.vgg16(pretrained=True)
for parameter in model.parameters():
    parameter.requires_grad = False

# Build and train the network
classifier = nn.Sequential(OrderedDict([
    ('inputs', nn.Linear(25088, 120)),  # hidden layer 1 sets output to 120
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(0.5)),  # could use a different dropout probability,but 0.5 usually works well
    ('hidden_layer1', nn.Linear(120, 90)),  # hidden layer 2 output to 90
    ('relu2', nn.ReLU()),
    ('hidden_layer2', nn.Linear(90, 70)),  # hidden layer 3 output to 70
    ('relu3', nn.ReLU()),
    ('hidden_layer3', nn.Linear(70, 102)),  # output size = 102
    ('output', nn.LogSoftmax(dim=1))]))  # For using NLLLoss()

model.classifier = classifier

if torch.cuda.is_available():
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Test The Network Model

epochs = 18
for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch + 1, epochs))
    model.train()
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0

    valid_loss = 0.0
    valid_acc = 0.0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        # Clean existing gradients
        optimizer.zero_grad()

        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation the gradients
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)

        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
        print("Batch no: {:03d}, Loss on training: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

                                                                                     acc.item()))
