from __future__ import division, print_function
import os
from flask.wrappers import Response 
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

# Global Variables
parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--directory', default='mSRGAN/epochs_msrgan/', type=str, help='directory to the model')
parser.add_argument('--model_name_celeba', default='netG_epoch_4_80.pth', type=str, help='generator model for jpg format')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = opt.test_mode
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
MODEL_NAME_CELEBA = opt.model_name_celeba
DIR = opt.directory

model = model.Generator(UPSCALE_FACTOR).eval()

# Define app
app = Flask(__name__)
print('Check http://127.0.0.1:5000')

# def load_model(model, model_dir):
#     if TEST_MODE:
#         model.cuda()
#         model.load_state_dict(torch.load(model_dir + MODEL_NAME))
#     else:
#         model.load_state_dict(torch.load(model_dir + MODEL_NAME, map_location=lambda storage, loc:storage))

def generate_img(model, model_dir, model_name, image_name):

    if TEST_MODE == "GPU":
        model.cuda()
        model.load_state_dict(torch.load(model_dir + model_name))
    else:
        model.load_state_dict(torch.load(model_dir + model_name, map_location=lambda storage, loc:storage))

    image = Image.open(image_name)
    with torch.no_grad():
        image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE == "GPU":
        image = image.cuda()
    else:
        image=image 
    
    img_out = model(image)
    out_img = ToPILImage()(img_out[0].data.cpu())

    return out_img

@app.route('/', methods = ['GET'])

def index():
    return flask.render_template("index.html")

@app.route('/predict', methods = ['GET','POST'])

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
            result = generate_img(model = model, model_dir=model_dir, model_name= MODEL_NAME, image_name=file_path)
        else:
            new_model_dir = 'mSRGAN/epochs_msrgan_celeba/'
            result = generate_img(model=model, model_dir=new_model_dir, model_name=MODEL_NAME_CELEBA , image_name=file_path)

        try:
            return Response(response=result, status=200, mimetype='image/png')
        except FileNotFoundError:
            os.abort(404)

if __name__ == '__main__':
    app.run(debug=True)
