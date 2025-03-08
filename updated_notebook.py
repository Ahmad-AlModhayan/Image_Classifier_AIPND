# Imports here
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from PIL import Image
import json

# Define data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
valid_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load category names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    output_size = len(cat_to_name)

# Build and train your network
# Load pretrained model
model = models.vgg16(pretrained=True)

# Freeze parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Define a new untrained feed-forward network as a classifier using ReLU and Dropout
input_size = 25088  # VGG16 input size
hidden_units = 5000  # Number of hidden units
learning_rate = 0.001  # Learning rate

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(hidden_units, output_size)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Validation Function
def validation(model, valid_loader, criterion, device):
    valid_loss = 0
    accuracy = 0
    
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        
        props = torch.exp(output)
        equality = (labels.data == props.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return valid_loss, accuracy

# Train the network
def train_classifier(model, optimizer, criterion, train_loader, valid_loader, epochs, device):
    running_loss = 0
    steps = 0
    prints_every = 40
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        
        for inputs, labels in iter(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % prints_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs} (steps: {steps}).. "
                      f"Train loss: {running_loss/len(train_loader):.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}.. ")
                
                running_loss = 0
                model.train()
    
    print("!!Train of the model is finished!!")

# Uncomment to train the model
# epochs = 18
# train_classifier(model, optimizer, criterion, train_loader, valid_loader, epochs, device)

# Test the network
def testing(model, test_loader, criterion, device):
    test_loss, accuracy = validation(model, test_loader, criterion, device)
    print("Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

# Uncomment to test the model
# testing(model, test_loader, criterion, device)

# Save the checkpoint
def save_checkpoint(model, train_dataset, arch, hidden_units, learning_rate, epochs, input_size, save_dir='checkpoint.pth'):
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'input_size': input_size,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, save_dir)
    print(f"Model checkpoint saved to {save_dir}")

# Uncomment to save the model
# save_checkpoint(model, train_dataset, "vgg16", hidden_units, learning_rate, 18, input_size)

# Load the checkpoint
def load_checkpoint(filepath):
    # Handle different device types gracefully
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint["arch"] == 'vgg16':
        print("Loading model: vgg16")
        model = models.vgg16(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 25088
    elif checkpoint["arch"] == 'densenet121':
        print("Loading model: densenet121")
        model = models.densenet121(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 1024
    else:
        print("Loading model: alexnet")
        model = models.alexnet(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 9216
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Get hidden units from checkpoint
    hidden_units = checkpoint.get('hidden_units', 5000)  # Default to 5000 if not found
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Open the image
    img = Image.open(image_path)
    
    # Resize the image where the shortest side is 256 pixels
    if img.width > img.height:
        img.thumbnail((img.width * 256 // img.height, 256))
    else:
        img.thumbnail((256, img.height * 256 // img.width))
    
    # Calculate center crop dimensions
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    
    # Crop image
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    np_img = np.array(img) / 255
    
    # Normalize based on means and std devs
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    
    # Transpose to match PyTorch's expected format
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img

# Display image
def imshow(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Predict the class for an image
def predict(image_path, model, cat_to_name, device, top_k=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    cat_to_name: dict mapping categories to names
    device: torch.device to run the model on
    top_k: integer. The top K classes to be calculated

    returns top_probabilities(k), top_labels, top_flowers
    """
    # Move model to the device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Process the image
    image = process_image(image_path)
    
    # Convert image from numpy to torch tensor 
    torch_image = torch.from_numpy(image).type(torch.FloatTensor).to(device)
    
    # Add batch dimension
    torch_image = torch_image.unsqueeze(0)
    
    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass through the model
        log_probs = model(torch_image)
        
        # Convert to linear scale
        linear_probs = torch.exp(log_probs)
        
        # Find the top k results
        top_probs, top_indices = linear_probs.topk(top_k)
        
        # Move to CPU and convert to numpy arrays
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Convert indices to class labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[idx] for idx in top_indices]
        
        # Map labels to flower names
        top_flowers = [cat_to_name[label] for label in top_labels]
    
    return top_probs, top_labels, top_flowers

# Display image along with top predictions
def display_img(image_path, model, cat_to_name, device=None):
    """
    Display an image along with the top 5 predicted classes
    image_path: Path to the image file
    model: Trained model to make predictions
    cat_to_name: Dictionary mapping categories to real names
    device: Device to run prediction on (defaults to available device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    
    # Display image
    img = process_image(image_path)
    imshow(img, ax)
    ax.set_title(image_path.split('/')[-1])
    
    # Get top 5 predictions
    probs, labels, flowers = predict(image_path, model, cat_to_name, device, top_k=5)
    
    # Create bar chart
    plt.subplot(2, 1, 2)
    sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0])
    plt.xlabel('Probability')
    plt.ylabel('Flower Type')
    plt.tight_layout()
    plt.show()

# Example of how to use the functions:
# 1. Load a trained model
# model = load_checkpoint('checkpoint.pth')
# 
# 2. Choose an image to predict
# image_path = 'flowers/test/1/image_06743.jpg'
# 
# 3. Display the image with predictions
# display_img(image_path, model, cat_to_name, device)
