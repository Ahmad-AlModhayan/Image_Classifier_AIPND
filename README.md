# Image Classifier Project

This project is part of the AI Programming with Python Nanodegree by Udacity. It's a deep learning application that can train on a dataset of images and then predict the class for new images.

## Project Overview

The project involves building an image classification application that can recognize different species of flowers. The application uses a pre-trained neural network (VGG16, DenseNet121, or AlexNet) that's been retrained to identify 102 different flower categories.

## Key Features

- Train a deep learning model on image data
- Use transfer learning with pre-trained networks (VGG16, DenseNet121, AlexNet)
- Make predictions with the trained model
- Command line application for both training and prediction

## Requirements

The project requires the following dependencies:
- Python 3
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- seaborn

You can install all dependencies using:
```
pip install -r requirements.txt
```

## Usage

### Training a model

To train a new model on the flower dataset:

```
python train.py --data_dir flowers --save_dir checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 5000 --epochs 18 --gpu
```

Arguments:
- `--data_dir`: Directory containing the images (default: 'flowers')
- `--save_dir`: Path to save the checkpoint file (default: './checkpoint.pth')
- `--arch`: Model architecture to use - vgg16, densenet121, or alexnet (default: 'vgg16')
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--hidden_units`: Number of hidden units in the classifier (default: 5000)
- `--epochs`: Number of epochs for training (default: 18)
- `--gpu`: Flag to use GPU for training, if available

### Predicting with a trained model

To predict the class of an image using a trained model:

```
python predict.py --image_dir flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

Arguments:
- `--image_dir`: Path to the image to classify (default: 'flowers/test/1/image_06743.jpg')
- `checkpoint`: Path to the checkpoint file (required)
- `--top_k`: Return top K most likely classes (default: 5)
- `--category_names`: Path to JSON file mapping categories to names (default: 'cat_to_name.json')
- `--gpu`: Flag to use GPU for inference, if available

## Project Structure

- `train.py`: Script for training the neural network
- `predict.py`: Script for making predictions using a trained model
- `functions_train.py`: Helper functions for training
- `functions_predict.py`: Helper functions for prediction
- `cat_to_name.json`: Mapping of category labels to flower names
- `requirements.txt`: List of required packages

## License

This project is part of the Udacity AI Programming with Python Nanodegree.
