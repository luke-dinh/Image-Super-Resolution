from __future__ import division, print_function
import os 
import numpy as np
import argparse

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from mSRGAN import model

import flask
from flask import Flask
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define app
app = Flask(__name__)
print('Check http://127.0.0.1:5000')

# Global Variables
parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--directory', default='mSRGAN/epochs_msrgan/', type=str, help='directory to the model')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
DIR = opt.directory

model = model.Generator(UPSCALE_FACTOR).eval()

# def load_model(model, model_dir):
#     if TEST_MODE:
#         model.cuda()
#         model.load_state_dict(torch.load(model_dir + MODEL_NAME))
#     else:
#         model.load_state_dict(torch.load(model_dir + MODEL_NAME, map_location=lambda storage, loc:storage))

def generate_img(model, model_dir, image_name):

    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load(model_dir + MODEL_NAME))
    else:
        model.load_state_dict(torch.load(model_dir + MODEL_NAME, map_location=lambda storage, loc:storage))

    image = Image.open(image_name)
    with torch.no_grad():
        image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()
    
    img_out = model(image)
    out_img = ToPILImage()(img_out[0].data.cpu())

    return out_img

@app.route('/', methods = ['GET'])

def index():
    return flask.render_template("index.html")

@app.route('/predict', methods = ['GET', 'POST'])

def upload():
    if flask.request.method == 'POST':
        # get the file
        f = flask.request.files['file']
        # save to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        if str(file_path).endswith('.png'):
            model_dir = DIR
            result = generate_img(model = model, model_dir=model_dir, image_name=file_path)
        else:
            new_model_dir = 'mSRGAN/epochs_msrgan_celeba/'
            result = generate_img(model=model, model_dir=new_model_dir, image_name=file_path)

        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
