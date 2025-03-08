import argparse
import torch
import functions_predict as f_predict

# Arguments
parser = argparse.ArgumentParser(description='Predict Flowers.')
parser.add_argument("--image_dir", default='flowers/test/1/image_06743.jpg', help='Image directory')
parser.add_argument("checkpoint", default='checkpoint.pth', help='Checkpoint file')
parser.add_argument("--top_k", type=int, default=5, help='Top k classes and probabilities')
parser.add_argument("--category_names", default='cat_to_name.json', help='mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Choose this argument if you want to use GPU')
args = parser.parse_args()

# Load category names
cat_to_name = f_predict.json_loader(args.category_names)

# Set device (GPU or CPU)
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model from checkpoint
model = f_predict.load_checkpoint(args.checkpoint)
print(f"Model architecture: {model.__class__.__name__}")

# Process and display the image
processed_image = f_predict.process_image(args.image_dir)
f_predict.imshow(processed_image)

# Make prediction
top_probs, top_labels, top_flowers = f_predict.predict(args.image_dir, model, cat_to_name, device, args.top_k)

# Print results
print("\nTop predictions:")
for i in range(len(top_probs)):
    print(f"{top_flowers[i]}: {top_probs[i]*100:.2f}%")

# Display image with predictions
f_predict.display_img(args.image_dir, model, cat_to_name, device)
