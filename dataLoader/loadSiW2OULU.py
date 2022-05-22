from PIL import Image
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataLoader.transform_image.transform_color_space import TransformColorSpace
# from transform_image.transform_color_space import TransformColorSpace
import torch

data_dir_prefix = '/home/chang/FAS_DATASET/OULU_NPU/'
train_txt_path_prefix = '/home/chang/EL_FAS/generateTrain/OULU_NPUTrain_P'
train_txt_path_suffix = '_sampled.txt'
train_txt_path = '/home/chang/EL_FAS/generateTrain/SiW2OULU/SiW2OULUTrain.txt'


class SiW2OULU(Dataset):
    def __init__(self, input_size, phase='train', color_space='rgb', protocol='1', scale=1.0):

        self.color_space = color_space
        self.phase = phase
        protocol = str(protocol)
        assert protocol in ('1', '2', '3-1', '3-2', '3-3', '3-4', '3-5', '3-6',
                    '4-1', '4-2', '4-3', '4-4', '4-5', '4-6')
        self.protocol = protocol.split('-')
        self.scale = str(scale)
        if self.phase == 'train':
            self.transform = getTrainTransforms(input_size, self.color_space)
            print("!!!!!!training with color space ", self.color_space)
        elif self.phase == 'eval' or self.phase == 'test':
            # eval == Dev
            self.transform = getValidTestTransforms(input_size, self.color_space)
        else:
            self.transform = None

        self.image_dir_prefix = None
        self.protocol_file_dir = os.path.join(data_dir_prefix, 'Protocols', 'Protocol_'+self.protocol[0])

        if self.phase == 'train':
            try:
                print('Training with 90 subjects in SiW :')
                with open(train_txt_path, 'r') as f:
                    self.train_df = pd.read_csv(f, delimiter=' ', header=None, names=['img_path', 'label'])
                    self.train_img_list = list(self.train_df['img_path'])
                    self.label_list = list(self.train_df['label'])
            except:
                print('can not open Train protocol file, may be filepath is not exist')
                exit()

        elif self.phase == 'eval':
            # eval == Dev
            try:
                if len(self.protocol) == 2:
                    txt_name = 'Dev_' + self.protocol[1] + '.txt'
                else:
                    txt_name = 'Dev.txt'
                with open(os.path.join(self.protocol_file_dir, txt_name)) as f:
                    self.protocol_df = pd.read_csv(f, delimiter=',', header=None, names=['label', 'video_name'])
                    self.video_name_list = list(self.protocol_df['video_name'])
                    self.label_list = list(self.protocol_df['label'])

                    self.label_list = [1 if x == 1 else 0 for x in self.label_list]
                    self.image_dir_prefix = os.path.join(data_dir_prefix, 'Dev_imgs', 'cropped_face', self.scale)
            except:
                print('can not open Dev protocol file, may be filepath is not exist')
                exit()

        elif self.phase == 'test':
            try:
                if len(self.protocol) == 2:
                    txt_name = 'Test_' + self.protocol[1] + '.txt'
                else:
                    txt_name = 'Test.txt'
                with open(os.path.join(self.protocol_file_dir, txt_name)) as f:
                    self.protocol_df = pd.read_csv(f, delimiter=',', header=None, names=['label', 'video_name'])
                    self.video_name_list = list(self.protocol_df['video_name'])
                    self.label_list = list(self.protocol_df['label'])
                    self.label_list = [1 if x == 1 else 0 for x in self.label_list]
                    self.image_dir_prefix = os.path.join(data_dir_prefix, 'Test_imgs', 'cropped_face', self.scale)
            except:
                print('can not open Test protocol file, may be filepath is not exist')
                exit()

        else:
            print('phase have to be one of (train, eval, test)')
            exit()

    def __len__(self):
        # return len(self.video_name_list)
        return len(self.label_list)

    def __getitem__(self, idx):
        label = int(self.label_list[idx])
        label = np.array(label)

        if self.phase == 'train':
            image = Image.open(self.train_img_list[idx])
        else:
            videoname = str(self.video_name_list[idx])+'.avi'

            image_list = os.listdir(os.path.join(self.image_dir_prefix, videoname))
            image_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            image = Image.open(os.path.join(self.image_dir_prefix, videoname, image_list[0]))

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.phase == 'train':
            return image, label
        else:
            return image, label, self.video_name_list[idx]


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def getTrainTransforms(input_size, color_space):

    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        TransformColorSpace(color_space),
        transforms.ToTensor(),
        normalize,
    ])

def getValidTestTransforms(input_size, color_space):

    return transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Resize(input_size),
        TransformColorSpace(color_space),
        transforms.ToTensor(),
        normalize,
    ])