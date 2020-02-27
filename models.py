import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import utils, models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from argparse import ArgumentParser

# customized dataset for loaidng pair of images
class CustomDataset(Dataset):
    def __init__(self, data_root, transform, frame_interval, sample_distance):
        self.samples = []
        self.data = []
        self.transform = transform
        self.frame_interval = frame_interval
        self.sample_distance = sample_distance


        for eachFolder in sorted(os.listdir(data_root)):
            img_folder = os.path.join(data_root, eachFolder)
            print ("folder", img_folder)
            for img in sorted(os.listdir(img_folder)):
                img_filepath = os.path.join(img_folder, img)
                
                self.samples.append(img_filepath)

        fn_list = [i for i in range(len(self.samples))]

        index = 0
        for j in self.frame_interval:
            last_interval = frame_interval[-1]
            pairs = [(fn_list[i], fn_list[i+j])
                     for i in range(len(self.samples) - last_interval)]
            pairs = pairs[0::self.sample_distance]  # every nth item
            for x, y in pairs:
                self.data.append((self.samples[x], self.samples[y], index))
            index += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        first_img_filepath, second_img_filepath, target = self.data[idx]
        first_img_to_tensor = Image.open(first_img_filepath)
        second_img_to_tensor = Image.open(second_img_filepath)

        if self.transform:
            first_img_to_tensor = self.transform(first_img_to_tensor)
            second_img_to_tensor = self.transform(second_img_to_tensor)

        return first_img_to_tensor, second_img_to_tensor, target





class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        
        # Convolution 1 
        self.channel_last = config.channel_last
        self.conv1 = nn.Conv2d(3, 6, 
                                config.kernel_size, 1, padding=1)

        # Convolution 2
        self.conv2 = nn.Conv2d(6, 16,
                                config.kernel_size, 1, padding=1)

        # Conv 2 Bath Norm
        self.conv2_bn = nn.BatchNorm2d(16)

        # Convolution 3
        self.conv3 = nn.Conv2d(16, 32,
                                config.kernel_size, 1, padding=1)
        
        # Conv 3 Bath Norm
        self.conv3_bn = nn.BatchNorm2d(32)
        
        # convolution 4
        # self.conv4 = nn.Conv2d(32, 64,
        #                        config.kernel_size, 1, padding=1)
        
        # fully connected 1
        self.img_size = config.img_resize //  (2**config.num_convs)

        self.fc1 = nn.Linear(32*self.img_size*self.img_size, 
                             120)
        
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, len(config.frame_interval))

    def forward_once(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))

        x = x.view(-1,  32*self.img_size*self.img_size)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input_1, input_2):
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        difference = output_1 - output_2

        concat = self.fc4(difference)
        return F.softmax(concat, dim=1)


def train(config, model, device, train_loader, optimizer, criterion, epoch):
    '''  Switch model to training mode.
         This is necessary for layers like dropout, batchnorm etc. which
         behave differently in training and evaluation mode
    '''
    model.train()
    
    train_corrects = 0.0
    train_running_loss = 0.0
    train_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader):
        img_1s = data[0].to(device)
        img_2s = data[1].to(device)
        labels = data[2].to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(img_1s, img_2s)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        # Getting Gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_corrects += (predicted == labels).sum().item()
        train_running_loss += loss.item()

        if batch_idx % config.log_interval == config.log_interval - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                   epoch + 1 , 
                   (batch_idx + 1 ) * len(data[0]), 
                   len(train_loader.sampler),
                   100.*(batch_idx + 1) / len(train_loader), 
                   train_running_loss / config.log_interval))
            train_loss += train_running_loss
            train_running_loss = 0.0

    train_accuracy = 100. * train_corrects / len(train_loader.sampler)
    train_loss /= len(train_loader.sampler)

    print('epoch :', (epoch+1))
    print('Train set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
            train_corrects, len(train_loader.sampler),
            train_accuracy, train_loss))
    
    wandb.log({
            "train_accuracy": train_accuracy,
            "train_loss": train_loss 
    }, commit=False)




def validate(config, model, device, val_loader, criterion, epoch):
    model.eval()

    val_corrects = 0
    val_loss = 0.0
   
    with torch.no_grad():
        for data in val_loader:
            img_1s = data[0].to(device)
            img_2s = data[1].to(device)
            labels = data[2].to(device)
            outputs = model(img_1s, img_2s)
            val_loss += criterion(outputs, labels).item()  # Sum up batch loass

            _, predicted = torch.max(outputs.data, 1)
            val_corrects += (predicted == labels).sum().item()
     
        val_accuracy = 100. * val_corrects / len(val_loader.sampler)
        val_loss /= len(val_loader.sampler)
    
        print('Val set: Accuracy: {}/{} ({:.0f}%), Average Loss: {:.6f}'.format(
                val_corrects, len(val_loader.sampler),
                val_accuracy, val_loss))
        
        wandb.log({ 
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })


def get_data_loaders(config):
    """
    Return splited training and test data 
    """
    data_transform = Compose([Resize((config.img_resize, config.img_resize)),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
    dataset = CustomDataset(config.img_folder, data_transform, 
                            config.frame_interval, config.sample_distance)
    random_seed = 0
    split = int(np.floor(config.validation_split * len(dataset)))
    indices = list(range(len(dataset)))
    if config.shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)

    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=config.train_batch_size, 
                              sampler=train_sampler)

    val_loader = DataLoader(dataset,
                            batch_size=config.val_batch_size,
                            sampler=val_sampler)

    return train_loader, val_loader



def run(config):
    train_loader, val_loader = get_data_loaders(config)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = LeNet(config).to(device)
    wandb.watch(model, log="all")        

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


    for epoch in range(config.epochs):
        train(config, model, device, train_loader, optimizer, criterion, epoch)
        validate(config, model, device, val_loader, criterion, epoch)
    


    # Save model to wandb
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='''how many batches to wait before 
                        logging training status (default: 10)''')
    parser.add_argument('--validation_split', type=float, default=0.3,
                        help='ratio of training and test data (default: 0.3)')
    parser.add_argument('--img_resize', type=int, default=128,
                        help='resize image (default: 128px)')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='conv layer kernel size (default: 3)')
    parser.add_argument('--channel_one', type=int, default=6,
                        help='conv layer one node (default: 6)')
    parser.add_argument('--channel_two', type=int, default=16,
                        help='conv layer two node (default: 16)')  
    parser.add_argument('--channel_last', type=int, default=32,
                        help='last conv layer node (default: 32)')
    parser.add_argument('--fc1', type=int, default=120,
                        help='fc1 layer node  (default: 120)')
    parser.add_argument('--fc2', type=int, default=84,
                        help='fc2 layer node  (default: 84)')
    parser.add_argument('--fc3', type=int, default=10,
                        help='fc3 layer node (default: 10)')
    parser.add_argument('--num_convs', type=int, default=3,
                        help='number of conv (default: 3)')

    parser.add_argument('--sample_distance', type=int, default=10,
                        help='number of conv (default: 10)')
    parser.add_argument('--img_folder', type=str, default='JPFuji-Train',
                        help='number of conv (default: JPFuji-Train:)')
    args = parser.parse_args()

    wandb.init(config=args, project="pytorch-dnn-uniqueframes")
 
    wandb.config.update({"frame_interval": [1, 3600, 7200, 10800, 14400]})
    wandb.config.update({"shuffle_dataset": True})

    # start the experiment
    run(wandb.config)
