#!/usr/bin/env python
# coding: utf-8

# In[5]:


# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image

cudnn.benchmark = True
plt.ion()   # interactive mode

import cv2
from tqdm import tqdm

data_dir = '/media/cheer/UI/Project/DarkPattern/code/individual_modules/iconModel/icon_random'
save_root = "model_icon_clean_noisy81"

classname_json = "iconModel_labels_noisy81.json"

included_types_path = "/media/cheer/UI/Project/DarkPattern/code/individual_modules/iconModel/iconTypesOver200.json"
included_types = json.load(open(included_types_path, "r"))
included_types.sort()

epoch = 100

# Data augmentation 
data_transforms = {
    'train': transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#         transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
#         transforms.Resize(256),
        transforms.Resize((224, 224)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
#         transforms.Resize(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
splits = ['train', 'val']

class IconDataset(Dataset):
    def __init__(self, root_dir, phase,  included_types, transforms=None):
        self.root_dir = root_dir
        self.phase = phase
        self.included_types = included_types
        self.included_types.append("others")
        self.class2id = {label:idx for idx, label in enumerate(self.included_types)}
        self.classes = self.included_types
        
        self.all_data = glob(os.path.join(self.root_dir, self.phase, "**/**"))
        self.all_data = [d for d in self.all_data if d.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
        
        self.transforms = transforms
        
        self.all_index = list(range(len(self.all_data)))
        if self.phase == "train":
            random.shuffle(self.all_index)
        
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        tmp_idx = self.all_index[idx]

        img_name = self.all_data[tmp_idx]
        img_type = os.path.basename(os.path.dirname(img_name))
        
        if img_type in self.class2id:
            class_id = self.class2id[img_type]
        else:
            class_id = self.class2id["others"]
        class_id = torch.tensor(class_id)
        
        image = Image.open(img_name).convert('RGB')
        image = self.transforms(image)
    
        return [image, class_id]


image_datasets = {x: IconDataset(data_dir, x, included_types,
                                          data_transforms[x])
                  for x in splits}
dataloaders = {x: DataLoader(image_datasets[x], 
                             batch_size=128,
                             shuffle=True, 
                             num_workers=10)
                  for x in splits}
dataset_sizes = {x: len(image_datasets[x]) for x in splits}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, save_root, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(0,num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) 
                torch.save(model, os.path.join(save_root, "best-{:.02f}.pt".format(epoch_acc)))
            torch.save(model, os.path.join(save_root, "{}-{}-{:.02f}.pt".format(epoch, phase, epoch_acc)))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# model_ft = torch.load("Model/best-0.84.pt")

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if not os.path.exists(save_root):
    os.makedirs(save_root)
model_ft = train_model(model_ft, criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler,
                       save_root = save_root,
                       num_epochs= epoch)

json.dump(class_names, open(classname_json, "w"))

