import os
from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import AutoImageProcessor

app = Flask(__name__)
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = torch.load('resnet_transfer_learning.pt').to('cpu')

def Predictor(img):
    out = model(**img)[0].argmax()
    return out

def Process(file):
    img = Image.open(file)
    img = processor(img, return_tensors="pt")
    return img

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   with torch.no_grad():
      model.eval()
      if request.method == 'POST':
         file = request.files['file']
         img = Process(file)
         pred = Predictor(img)
      if pred == 0:
         return render_template('result.html', result = 'Ant')
      else:
         return render_template('result.html', result = 'Bee')

if __name__ == '__main__':
   app.run()