import argparse
import functions_predict as f_predict

# Arguments
parser = argparse.ArgumentParser(description='Predict Flowers.')
parser.add_argument("--image_dir", default='flowers/test/1/image_06743.jpg', help='Image directory')
parser.add_argument("checkpoint", default='checkpoint.pth', help='Checkpoint file')
parser.add_argument("--top_k", default=5, help='Top k classes and probabilities')
parser.add_argument("--category_names", default='cat_to_name.json', help='mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Choose this argument if you want to use GPU')
args = parser.parse_args()

cat_to_name = f_predict.json_loader(args.category_names)

model = f_predict.load_checkpoint(args.checkpoint)
print(model)

process = f_predict.process_image(args.image_dir)

f_predict.imshow(process)

top_probs, top_labels, top_flowers = f_predict.predict(args.image_dir, model, cat_to_name, device, args.top_k)

f_predict.display_img(args.image_dir, model, cat_to_name)
