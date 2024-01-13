from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import sys

# Read in the file from the command line
filename = sys.argv[1]
image = Image.open(filename).convert("RGB")

# Load the image processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Process the input PIL.Image into a tensor(pt = pytorch tensor)
inputs = processor(images=image, return_tensors="pt")

# Run the model on the image
outputs = model(**inputs)

# Get the logits (proxy for probability)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
probs = logits.softmax(dim=1)

class_label = model.config.id2label[predicted_class_idx]
probability = probs[0][predicted_class_idx].item()

# Print the predicted class
print("Predicted class:", class_label)
print("Predicted probability:", probability)

# Display the input image
image.show()

# Print the top 5 predicted classes
probs = logits.softmax(dim=1)
top5_probs, top5_indices = probs.topk(5)
for i in range(top5_probs.shape[0]):
   for j in range(top5_probs.shape[1]):
       print(f"Class: {model.config.id2label[top5_indices[i][j].item()]}, Probability: {top5_probs[i][j].item()}")
