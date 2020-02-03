from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="Data/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)
f= open("log.txt","a+")
f.write(str(opt))

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]
val_path = data_config["valid"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
#model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

val_dataloader = torch.utils.data.DataLoader(
    ListDataset(val_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    train_nCorrect = 0
    train_nGT = 0
    train_nProp = 0
    val_nCorrect = 0
    val_nGT = 0
    val_nProp = 0
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        train_nCorrect += model.losses["nCorrect"]
        train_nGT += model.losses["nGT"]
        train_nProp += model.losses["nProp"]

        str = "Training [Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f ,ncorrect: %.5f ,nprop: %.5f ,nGT: %.5f ]\n" % (
                epoch+1,
                opt.epochs,
                batch_i+1,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
                model.losses["nCorrect"],
                model.losses["nProp"],
                model.losses["nGT"]
                
        )
        print(str)
        f= open("log.txt","a+")
        f.write(str)
        model.seen += imgs.size(0)

    #validation loss
    with torch.no_grad():
       for batch_i, (_, imgs, targets) in enumerate(val_dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        #loss.backward()
        #optimizer.step()
        val_nCorrect += model.losses["nCorrect"]
        val_nGT += model.losses["nGT"]
        val_nProp += model.losses["nProp"]

        str = "[Validation Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f ,ncorrect: %.5f ,nprop: %.5f ,nGT: %.5f ]\n" % (
                epoch+1,
                opt.epochs,
                batch_i+1,
                len(val_dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
                model.losses["nCorrect"],
                model.losses["nProp"],
                model.losses["nGT"]
        )

        print(str)
        f= open("log.txt","a+")
        f.write(str)
        model.seen += imgs.size(0)

    train_recall = float(train_nCorrect / train_nGT) if train_nGT else 1
    train_precision = 0
    if train_nProp > 0:
        train_precision = float(train_nCorrect / train_nProp)

    str = "Trainig [Epoch %d/%d] [recall: %.5f, precision: %.5f]\n" % (
                epoch+1,
                opt.epochs,
                train_recall,
                train_precision,
    )
    print(str)
    f= open("log.txt","a+")
    f.write(str)

    val_recall = float(val_nCorrect / val_nGT) if val_nGT else 1
    val_precision = 0
    if val_nProp > 0:
        val_precision = float(val_nCorrect / val_nProp)

    str = "validation [Epoch %d/%d] [recall: %.5f, precision: %.5f]\n" % (
                epoch+1,
                opt.epochs,
                val_recall,
                val_precision,
    )
    print(str)
    f= open("log.txt","a+")
    f.write(str)


    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))


