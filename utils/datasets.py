import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import utils, models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from argparse import ArgumentParser# customized dataset for loaidng pair of images

class ListDataset(Dataset):
    def __init__(self, data_root):
        self.data = []
        f = open('./find_phone/labels.txt', 'r')
        labels = f.read().splitlines()
        Dict = {}
        for line in labels:
            splitline = line.split(' ', 1)
            Dict[splitline[0]] = splitline[-1]
        
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(data_root)
        img_file = open('./train.txt', 'r')
        img_path = f.read().splitlines()
        for img in img_path:
            print(img)
            img_filepath = os.path.join(img_folder, img)
                
            self.data.append(img_filepath, Dict[img_filepath])            # Save the images that are in a certain interval into a pair (img1, img2, label)
        
        #create list of pair: (img, coords)        
        
        # for j in self.frame_interval:
        #     last_interval = frame_interval[-1]
        #     pairs = [(fn_list[i], fn_list[i+j])
        #              for i in range(len(self.samples) - last_interval)]
        #     pairs = pairs[0::self.sample_distance]  # every nth item
        #     for x, y in pairs:
        #         self.data.append((self.samples[x], self.samples[y], index))
        #     index += 1    
            
            
    def __len__(self):
        return len(self.data)
    
    # Open up each file as numpy array, then covert to tensor
    def __getitem__(self, idx):
        first_img_filepath, target = self.data[idx]
        first_img_to_tensor = Image.open(first_img_filepath)
        # if self.transform:
        #     first_img_to_tensor = self.transform(first_img_to_tensor)
        #     second_img_to_tensor = self.transform(second_img_to_tensor)        
        return first_img_to_tensor, target