import PIL
from PIL import Image
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


####################################################
def load_checkpoint(filepath):
    # Handle different device types gracefully
    # Set weights_only=False to handle PyTorch 2.6+ compatibility
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath, weights_only=False)
    else:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

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
        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


####################################################
def json_loader(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)

        return cat_to_name


####################################################
def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
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


####################################################
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


####################################################
def predict(image_path, model, cat_to_name, device, top_k=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
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


####################################################
def display_img(image_path, model, cat_to_name, device=None):
    """Display an image along with the top 5 predicted classes
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
