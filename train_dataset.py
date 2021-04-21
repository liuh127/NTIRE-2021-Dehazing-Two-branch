from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import random
import os

#data augmentation for image rotate
def augment(hazy, clean):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        return hazy, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        return hazy, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        return hazy, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean


class dehaze_train_dataset(Dataset):
    def __init__(self, train_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open(os.path.join(train_dir, 'train.txt')):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        #'./NTIRE2021_Train_Hazy/'
        self.root_hazy=os.path.join(train_dir, 'hazy/')
        self.root_clean =os.path.join(train_dir, 'clean/')
        self.file_len = len(self.list_train)

    def __getitem__(self, index, is_train = True):
        if is_train:
            hazy = Image.open(self.root_hazy + self.list_train[index])
            clean=Image.open(self.root_clean + self.list_train[index])
            #crop a patch
            i,j,h,w = transforms.RandomCrop.get_params(hazy, output_size = (256,256))
            hazy_ = TF.crop(hazy, i, j, h, w)
            clean_ = TF.crop(clean, i, j, h, w)

            #data argumentation
            hazy_arg, clean_arg = augment(hazy_, clean_)
        hazy = self.transform(hazy_arg)
        clean = self.transform(clean_arg)
        return hazy,clean

    def __len__(self):
        return self.file_len
