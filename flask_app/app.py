from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
import base64
import numpy as np

app = Flask(__name__)

# Load your trained model
model_path = 'A:\CODES\MultiClass Classification ResNet50\Classification\your_model.pth' 
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()


# Define transformation
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB if the image is grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Update the prediction function
def predict_image(image_path):
    img = Image.open(image_path)
    img_data = preprocess(img)
    img_data = img_data.unsqueeze(0)

    with torch.no_grad():
        output = model(img_data)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prediction_index = torch.argmax(probabilities).item()
        class_label = model.idx_to_class[prediction_index]

    return class_label, probabilities, img_data





@app.route('/', methods=['GET', 'POST'])
def index():
    file_path = None

    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # Check if the file has a name
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Check if the file is allowed
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template('index.html', message='Invalid file extension')

        # Save the file
        upload_folder = 'A:/CODES/MultiClass Classification ResNet50/Classification/upload_folder'

        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(file_path)

    if file_path:
        # Get prediction
        class_label, probabilities, img_data = predict_image(file_path)

        # Render the result
        return render_template('result.html', class_label=class_label, probabilities=probabilities, img_data=img_data)

    return render_template('index.html', message='Upload an image')


if __name__ == '__main__':
    app.run(debug=True)
