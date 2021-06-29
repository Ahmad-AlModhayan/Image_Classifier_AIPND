import PIL
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import json
import numpy as np
import matplotlib as plt
import seaborn as sb


####################################################
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint["arch"] == 'vgg16':
        print("training using vgg")
        model = models.vgg16(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 25088
    elif checkpoint["arch"] == 'densenet121':
        print("training using densenet")
        model = models.densenet121(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 1024
    else:
        print("training using alexnet")
        model = models.alexnet(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        input_size = 9216

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, cat_to_name)),
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
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)

    # Get original dimensions
    original_width, original_height = img.size

    # Find shorter size and create settings to crop shortest side to 256
    if original_width < original_height:
        size = [256, 256 ** 600]
    else:
        size = [256 ** 600, 256]

    img.thumbnail(size)

    center = original_width / 4, original_height / 4
    left, top, right, bottom = center[0] - (224 / 2), center[1] - (224 / 2), center[0] + (224 / 2), center[1] + (
            224 / 2)
    img = img.crop((left, top, right, bottom))

    np_img = np.array(img) / 255

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_img = (np_img - mean) / std

    # Set the color to the first channel
    np_img = np_img.transpose(2, 0, 1)

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
def predict(image_path, model, jsonfile, device, top_k=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
        image_path: string. Path to image, directly to image and not to folder.
        model: pytorch neural network.
        top_k: integer. The top K classes to be calculated

        returns top_probabilities(k), top_labels
    """

    # TODO: Implement the code to predict the class from an image file

    model.to(device)

    # Set model to evaluate
    model.eval()

    cat_to_name = jsonfile
    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)

    # Detach all of the details
    top_probs = np.array(top_probs.detach())[
        0]
    # This is not the correct way to do it but the correct way isn't working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


####################################################
def display_img(image_dir, jsonfile, model):
    cat_to_name = jsonfile
    # Define image path
    image_path = image_dir

    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    # Set up title
    title = cat_to_name[image_path]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax)

    # Make prediction
    probs, labs, flowers = predict(image_path, model, cat_to_name)

    # Plot bar chart
    plt.subplot(2, 1, 2)
    sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0])
    plt.show()
