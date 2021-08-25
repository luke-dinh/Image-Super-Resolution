import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator
# torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--model_name_celeba', default='netG_epoch_4_80.pth', type='str', help='generator model for jpg images')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

DIR_1 = 'epochs_msrgan'
DIR_2 = 'epochs_msrgan_celeba'

model = Generator(UPSCALE_FACTOR).eval()

# For images with jpg or jpeg format:
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs_msrgan_celeba/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs_msrgan_celeba/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# # For images with png format:
# if TEST_MODE:
#     model.cuda()
#     model.load_state_dict(torch.load('epochs_msrgan/' + MODEL_NAME))
# else:
#     model.load_state_dict(torch.load('epochs_msrgan/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
with torch.no_grad():
  image = Variable(ToTensor()(image)).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.process_time()
out = model(image)
elapsed = (time.process_time() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)