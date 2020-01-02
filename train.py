import argparse
import torch
import torch.nn as nn
import numpy as npy, os
from torch.autograd import Variable
from model import DualCNN
import torch.optim as optim
from dataset import DatasetFromHdf5
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pdb

import cv2 as cv
from model import DualCNN
from data_proc import DataIterEPF

datadir=r"Data"
img_list1=[os.path.join("BSDS200_RTV",f) for f in os.listdir(os.path.join(datadir,"BSDS200_RTV")) if f.find(".png")!=-1]
img_list2=[os.path.join("BSDS200",f) for f in os.listdir(os.path.join(datadir,"BSDS200")) if f.find(".png")!=-1]
img_list=[[f1,f2] for f1, f2 in zip(img_list1,img_list2)]
parser = argparse.ArgumentParser(description="DualCNN for epf")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=400000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")

loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_iter=DataIterEPF(datadir, img_list, 10, 41, True)

def main():
    global opt
    opt = parser.parse_args()
    print(opt)
    dcnn = DualCNN()
    dcnn = dcnn.to(device)
    criterion = nn.MSELoss(size_average=False)
    criterion = criterion.to(device)
    #optimizer = optim.SGD(dcnn.parameters(), lr=opt.lr, momentum=opt.momentum)
    optimizer = optim.Adam(dcnn.parameters(),lr=opt.lr,betas=(0.9,0.99))
    for epoch in range(0, opt.nEpochs):
        train( optimizer, dcnn, criterion, epoch)
        if epoch % 10000 == 0:
            #test(dcnn, epoch)
            save_checkpoint(dcnn, epoch)



def train(optimizer, model, criterion, epoch):

    model.train()
    train_loss = 0
    
    label, data =data_iter.fetch_next()
    label = torch.from_numpy(label)
    data = torch.from_numpy(data)
    data = data.float()
    label = label.float()
    data = data.to(device)
    label = label.to(device)
    l2_regularization = torch.tensor([0],dtype=torch.float32)
    l2_regularization = l2_regularization.to(device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)
    outputs, srcnn = model(data)
    #loss = criterion(outputs, label) + 0.001 * criterion(srcnn, label)
    loss = criterion(outputs, label) + 0.001 * criterion(srcnn, label)+ l2_regularization
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch:{}  Loss:{}".format(epoch, loss.item()))

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

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(model, model_out_path)
    print("checkpoint saved to {}".format(model_out_path))
def test(model, epoch):
    path1 = "./Data/test_data/12.jpg"
    path2 = "./Data/test_data/11.jpg"
    input1 = image_loader(path1)
    input2 = image_loader(path2)
    #pdb.set_trace()
    save_path1 = "./picture/12-{}.jpg".format(epoch)
    save_path2 = "./picture/11-{}.jpg".format(epoch)
    output1, srcnn1 = model(input1)
    output2, srcnn2 = model(input2)
    im1 = tensor_to_PIL(output1)
    im1.save(save_path1)
    im2 = tensor_to_PIL(output2)
    im2.save(save_path2)
    

    

    
if __name__ == "__main__":
    main()
