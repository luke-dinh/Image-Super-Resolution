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

def load_model(model, image_name):
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load(DIR + MODEL_NAME))
    else:
        model.load_state_dict(torch.load(DIR + MODEL_NAME, map_location=lambda storage, loc:storage))
