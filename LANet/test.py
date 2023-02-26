"""
Inference Code
"""
from skimage import metrics
from loss import TrainLoss
import argparse
import cv2
import numpy as np
import os
import random
import torchvision.transforms as trasforms
import tensorboardX
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time

from skimage.transform import resize

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_mask(img, height, width):
    alpha = 0.90
    img_gray = cv2.cvtColor((img*255).astype('uint8'),
                            cv2.COLOR_RGB2GRAY)/255.0
    mask = np.maximum(0, img_gray-alpha)/(1-alpha)
    mask = np.clip(mask, 0, 1)
    mask = mask.reshape((height, width, 1))
    return mask


def binary_mask(img, height, width):
    img_gray = cv2.cvtColor((img*255).astype('uint8'),
                            cv2.COLOR_RGB2GRAY)/255.0
    mask = img_gray
    mask[mask < 0.85] = 0
    mask[mask > 0.85] = 1
    mask = np.clip(mask, 0, 1)
    mask = mask.reshape((height, width, 1))
    return mask


def smooth_mask(img, height, width):
    img_gray = cv2.cvtColor((img*255).astype('uint8'),
                            cv2.COLOR_RGB2GRAY)/255.0
    mask = img_gray
    mask[mask < 0.8] = 0
    mask[mask > 0.85] = 1
    mask = np.maximum(0, mask-0.8)/(1-0.8)
    mask = np.clip(mask, 0, 1)
    mask = mask.reshape((height, width, 1))
    return mask


def read_img_hdr(ldr_name, hdr_name, height, width, strategy='basic'):
    '''
    Read hdr image and equalize the luminance
    '''


    ldr_img = cv2.imread(os.path.join(args.test_dir, ldr_name))
    ldr_img = resize(ldr_img, (height, width), anti_aliasing=True)

    hdr_img = cv2.imread(os.path.join(args.gt_dir, hdr_name), -1)
    hdr_img = resize(hdr_img, (height, width), anti_aliasing=True)

    ldr_img_gamma = ldr_img ** (2.2)
    LDR_I = np.mean(ldr_img_gamma, axis=2)
    mask = np.where(LDR_I > 0.83, 0, 1).reshape((256, 512, 1))
    mean_LDR = (mask * ldr_img_gamma).reshape(-1, 3).mean(0)
    hdr_img_expose = np.clip(hdr_img, 1e-5, 1e6)
    HDR_I = np.mean(hdr_img_expose, axis=2)
    mean_HDR = (mask * hdr_img_expose).reshape(-1, 3).mean(0)
    hdr_img_expose *= mean_LDR/mean_HDR

    hdr_img_gt = torch.from_numpy(hdr_img_expose)
    hdr_img_gt = hdr_img_gt.permute(2, 0, 1)
    hdr_img_gt = torch.log(hdr_img_gt)

    if strategy == 'basic':
        mask = create_mask(ldr_img, height, width)
    elif strategy == 'binary':
        mask = binary_mask(ldr_img, height, width)
    elif strategy == 'smooth':
        mask = smooth_mask(ldr_img, height, width)
    mask_gt = torch.from_numpy(mask)
    mask_gt = mask_gt.permute(2, 0, 1)

    ldr_img_gt = torch.from_numpy(np.clip(ldr_img, 1e-5, 1))
    ldr_img_gt = ldr_img_gt.permute(2, 0, 1)
    ldr_img_gt = ldr_img_gt.float()

    return ldr_img_gt.unsqueeze(dim=0).cuda(), mask_gt.unsqueeze(dim=0).cuda(), hdr_img_gt.unsqueeze(dim=0).cuda()



def read_img(ldr_name,height, width, strategy='smooth'):
    
    ldr_img = cv2.imread(os.path.join(args.test_dir, ldr_name))
    ldr_img = resize(ldr_img, (height, width), anti_aliasing=True)

    if strategy == 'basic':
        mask = create_mask(ldr_img, height, width)
    elif strategy == 'binary':
        mask = binary_mask(ldr_img, height, width)
    elif strategy == 'smooth':
        mask = smooth_mask(ldr_img, height, width)
    mask_gt = torch.from_numpy(mask)
    mask_gt = mask_gt.permute(2, 0, 1)

    ldr_img_gt = torch.from_numpy(np.clip(ldr_img, 1e-5, 1))
    ldr_img_gt = ldr_img_gt.permute(2, 0, 1)
    ldr_img_gt = ldr_img_gt.float()

    return ldr_img_gt.unsqueeze(dim=0).cuda(), mask_gt.unsqueeze(dim=0).cuda()


def read_exr_image(ldr_name, height, width, strategy='smooth'):
    hdr_img = read_exr(os.path.join(args.test_dir, ldr_name))
    hdr_img = hdr_img[..., ::-1]   # RGB to BGR
    hdr_img = resize(hdr_img, (height, width), anti_aliasing=True)
    ldr_img = hdr_to_ldr(hdr_img, dtype='uint8')

    if strategy == 'basic':
        mask = create_mask(ldr_img, height, width)
    elif strategy == 'binary':
        mask = binary_mask(ldr_img, height, width)
    elif strategy == 'smooth':
        mask = smooth_mask(ldr_img, height, width)
    mask_gt = torch.from_numpy(mask)
    mask_gt = mask_gt.permute(2, 0, 1)

    ldr_img_gt = torch.from_numpy(np.clip(ldr_img, 1e-5, 1))
    ldr_img_gt = ldr_img_gt.permute(2, 0, 1)
    ldr_img_gt = ldr_img_gt.float()

    return ldr_img_gt.unsqueeze(dim=0).cuda(), mask_gt.unsqueeze(dim=0).cuda(), hdr_img # (128, 256, 3)


def make_image(image, gamma_correct=True, exposure=1):
    output = np.clip(np.clip(image, 0, 1)*exposure, 0, 1)
    if gamma_correct:
        output = output ** (1.0/2.2)
    output = output * 255.0
    output = output.astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def tensor2np(batched_image):
    return batched_image[0, :, :, :].detach().cpu().numpy()


def siMSE(gt, pred, alpha=1.0):
    return torch.mean(torch.mean((gt-pred)**2, axis=[1, 2, 3])
                      - alpha*torch.pow(torch.mean(gt-pred, axis=[1, 2, 3]), 2))


import Imath
import numpy as np
from PIL import Image
from OpenEXR import InputFile, OutputFile, Header

def read_exr(filename, channel=3):
	"""Reads an OpenEXR file and returns the data as a numpy array.
	Args:
	    filename: path to exr file
		channel: number of channels in exr file
	Return
		rgb32f: numpy array of shape [height, width, channel]
	"""
	src = InputFile(filename)
	pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
	dw = src.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	rgb32f = None
	if channel == 3:
		for i, c in enumerate('RGB'):
			c32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32).reshape(size[::-1])
			rgb32f = c32f if i == 0 else np.dstack((rgb32f, c32f))
	elif channel == 1:
		rgb32f = np.fromstring(src.channel('A', pixel_type), dtype=np.float32).reshape(size[::-1]).unsqueeze(2)

	return rgb32f


def write_exr(filename, data):
	"""Write an OpenEXR file from a numpy array.
	Args:
		filename: path to exr file
		data: numpy array to write to exr file
	"""
	assert '.exr' in filename, 'extension must be .exr'
	assert data.dtype == np.float32, f'Data type is {type(data)}, should be np.float32'

	out = OutputFile(filename, Header(data.shape[1], data.shape[0]))

	if data.shape[2] == 3:
		r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
	elif data.shape[2] == 1:
		r = g = b = data[:, :, 0]
	out.writePixels({'R': r.tostring(), 'G': g.tostring(), 'B': b.tostring()})

	out.close()


def hdr_to_ldr(color, gamma=2.2, dtype='float32', clamp=True):
	# aces tone mapping
	A = 2.51
	B = 0.03
	C = 2.43
	D = 0.59
	E = 0.14

	# color: [B, 3]
	color = (color * (A * color + B)) / (color * (C * color + D) + E)
	if type(color) is torch.Tensor:
		if clamp:
			color = torch.clamp(color, 0, 1)
		if dtype == 'uint8':
			color = (color * 255.).to(torch.uint8) / 255.
			color = color.to(torch.float32)
	elif type(color) is np.ndarray:
		if clamp:
			color = np.clip(color, 0, 1)
		if dtype == 'uint8':
			color = (color * 255).astype(np.uint8) / 255.
			color = color.astype(np.float32)
	else:
		raise TypeError('color must be torch.Tensor or np.ndarray')

	return color ** (1 / gamma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='checkpoints')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--strategy', type=str, default='smooth')
    parser.add_argument('--transport_matrix', type=str,
                        default='transportMat.BumpySphereMiddle.top.e64.r64.half.mat', help='directory to data')
    args = parser.parse_args()

    device = 'cuda:0'

    if args.gt_dir != '':
        imgs = [f for f in os.listdir(args.gt_dir) if os.path.isfile(
            os.path.join(args.gt_dir, f))]
        os.makedirs(f'{args.output_dir}/gt/', exist_ok=True)    
        os.makedirs(f'{args.output_dir}/mask_gt/', exist_ok=True)
    else:
        imgs = [f for f in os.listdir(args.test_dir) if os.path.isfile(
            os.path.join(args.test_dir, f))]

    os.makedirs(f'{args.output_dir}/gt', exist_ok=True)
    os.makedirs(f'{args.output_dir}/pred', exist_ok=True)

    l2loss = nn.MSELoss()
    l1loss = nn.L1Loss()
    model = torch.load(args.ckpt)
    model = model.cuda()

    model.eval()
    for i in tqdm(imgs):
        name = i.split('.')[0]
        ext = i.split('.')[-1]
        
        '''
        Preprocess input data and save ground truth for evaluation
        '''
        if args.gt_dir != '':
            ldr_gt, mask_gt, hdr_gt = read_img_hdr(
                f'{name}.png', f'{name}.exr', args.height, args.width, args.strategy)
            output_hdr_gt = tensor2np(hdr_gt)
            output_hdr_gt = np.transpose(
                np.exp(output_hdr_gt), (1, 2, 0)).astype('float32')
            ldr_gamma_gt = ldr_gt**(2.2)
            ldr_mask = ldr_gamma_gt*(torch.tensor([1]).cuda()-mask_gt)
            hdr_gt_mask = torch.exp(hdr_gt+epsilon)*mask_gt
            hdr_mask_gt = ldr_mask+hdr_gt_mask
            output_mask_gt = tensor2np(hdr_mask_gt)
            output_mask_gt = np.transpose(
                output_mask_gt, (1, 2, 0)).astype('float32')
            cv2.imwrite(os.path.join(f'{args.output_dir}/gt/',
                    name+'.hdr'), output_hdr_gt)
            cv2.imwrite(os.path.join(f'{args.output_dir}/mask_gt/',
                    name+'.hdr'), output_mask_gt)
        else:
            ldr_gt, mask_gt, hdr_gt = read_exr_image(
                f'{i}', args.height, args.width, args.strategy) 
        
        '''
        Infer model and save results
        '''
        hdr_pred, mask_pred = model(ldr_gt)
        epsilon = torch.tensor([1e-5]).cuda()
        output_hdr_pred = tensor2np(hdr_pred)
        output_hdr_pred = np.transpose(np.exp(output_hdr_pred), (1, 2, 0)).astype('float32')

        hdr_mask = torch.exp(hdr_pred+epsilon)*mask_gt
        ldr_gamma_gt = ldr_gt**(2.2)
        ldr_mask = ldr_gamma_gt*(torch.tensor([1]).cuda()-mask_gt)
        hdr_mask_pred = ldr_mask+hdr_mask

        output_mask_pred = tensor2np(hdr_mask_pred)
        output_mask_pred = np.transpose(output_mask_pred, (1, 2, 0)).astype('float32')
        output_mask_pred = resize(output_mask_pred, (128, 256), anti_aliasing=True)
        

        # ldr_gamma_gt = tensor2np(ldr_gamma_gt)
        # ldr_gamma_gt = np.transpose(ldr_gamma_gt, (1, 2, 0)).astype('float32')

        # save pred and gt
        output_mask_pred = output_mask_pred[..., ::-1]
        hdr_gt = hdr_gt[..., ::-1]  # BGR to RGB
        write_exr(output_mask_pred, f'{args.output_dir}/pred/{name}.exr')
        write_exr(hdr_gt, f'{args.output_dir}/gt/{name}.exr')
    

    
