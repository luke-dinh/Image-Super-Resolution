from __future__ import division, print_function
import os 
import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from mSRGAN import model

import flask
from flask import Flask
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_path = "mSRGAN/epochs_msrgan"
model_name = "netG_epoch_4_100.pth"

