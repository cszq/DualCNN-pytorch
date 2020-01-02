import torch
import torch.nn as nn
import cv2 as cv
import os
import numpy as np
from PIL import Image
from model import DualCNN
import pdb
import torchvision.transforms as transforms
path = "./Data/test_data/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loader = transforms.Compose([
    transforms.ToTensor()])  
path1 = "./Data/test_data_RTV/"
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    image = image / 255.0
    return image.to(device, torch.float)
    
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.mul(255.0)
    image = unloader(image)
    return image
    
def epf_metric(hr_imgs, sr_imgs, filename):
    
    diff=hr_imgs[:,:,:] - sr_imgs[:,:,:]     
    mse = np.mean(diff*diff) 
    psnr =10*np.log10(1.0/mse)
    print("{} : psnr{}".format(filename, psnr))
    
model = torch.load('./checkpoint/model_epoch_390000.pth')
model = model.to(device)

with torch.no_grad():
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            path2 = path1 + filename
            
            im_RTV = Image.open(path2).convert('RGB')
            im_RTV = np.asarray( im_RTV, dtype=np.float32)
            
            im_RTV = im_RTV.transpose(2, 0, 1)
            im_RTV = im_RTV / 255.0
            #pdb.set_trace()
            im_path = os.path.join(root, filename)
            input = image_loader(im_path)
            output, srcnn1 = model(input)
            im_pre = output.cpu().clone()
            
            im_pre = im_pre.squeeze(0).numpy()
            im_pre = np.multiply(im_pre, 255.0)
            epf_metric(im_pre, im_RTV, filename)
            
            #im = tensor_to_PIL(output)
            #im.save("./picture/{}".format(filename))
    

    



