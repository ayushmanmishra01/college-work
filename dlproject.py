mport torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# 1. Download official image class names (1,000 categories like 'Beagle', 'Persian cat', etc.)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")
with open("imagenet_classes.txt", "r") as f:
    categories = [line.strip() for line in f.readlines()]

# 2. Define Image Preprocessor (Resizes image to standard size)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load Pre-trained Neural Network
model = models.resnet18(pretrained=True)
model.eval()  # Set to inference mode

# 4. Create a dummy image (RGB pixel tensor) or load your own image file
# To test with a real file, use: input_image = Image.open("dog.jpg")
input_image = Image.new('RGB', (300, 300), color = (100, 150, 200))

# 5. Process image and run model
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create batch dimension

with torch.no_grad():
    output = model(input_batch)

# 6. Get Top Prediction
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class_id = torch.argmax(probabilities).item()

print(f"Predicted Animal / Object: {categories[predicted_class_id]}")
print(f"Confidence: {probabilities[predicted_class_id].item() * 100:.2f}%")