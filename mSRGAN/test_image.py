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
MODEL_NAME_1 = opt.model_name
MODEL_NAME_2 = opt.model_name_celeba

DIR_1 = 'epochs_msrgan'
DIR_2 = 'epochs_msrgan_celeba'

model = Generator(UPSCALE_FACTOR).eval()

def load_image(model, model_dir, model_name, image_name):

    # For images with jpg or jpeg format:
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load(model_dir + model_name))
    else:
        model.load_state_dict(torch.load(model_dir + model_name, map_location=lambda storage, loc: storage))


    image = Image.open(image_name)
    with torch.no_grad():
        image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    img_out = model(image)
    out_img = ToPILImage()(img_out[0].data.cpu())

    return out_img

start = time.process_time()
if str(IMAGE_NAME).endswith('.png'):
    result = load_image(model=model, model_dir=DIR_1, model_name=MODEL_NAME_1, image_name=IMAGE_NAME)
else:
    resultt = load_image(model=model, model_dir=DIR_2, model_name=MODEL_NAME_2, image_name=IMAGE_NAME)
elapsed = (time.process_time() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(result[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)