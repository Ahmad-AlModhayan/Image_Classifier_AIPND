import argparse

parser = argparse.ArgumentParser(description='Build and Train the Model.')
parser.add_argument("data_dir", )
parser.add_argument("--save_dir")
parser.add_argument("--arch")
parser.add_argument("--learning_rate")
parser.add_argument("--hidden_units")
parser.add_argument("--epochs")
parser.add_argument("--gpu")
args = parser.parse_args()



