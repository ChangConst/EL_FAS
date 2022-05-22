from PIL import Image
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataLoader.transform_image.transform_color_space import TransformColorSpace
# from transform_image.transform_color_space import TransformColorSpace

data_dir_prefix = '/home/chang/FAS_DATASET/SiW_frames/'
train_txt_path_prefix = '/home/chang/EL_FAS/generateTrain/SiWTrainTXT/SiWTrainP'
train_txt_path_suffix = '.txt'
test_txt_path_prefix = '/home/chang/EL_FAS/generateTrain/SiWTrainTXT/SiWTestP'
test_txt_path_suffix = '.txt'

class SiW(Dataset):
    def __init__(self, input_size, phase='train', color_space='rgb', protocol='1', scale=1.0):

        self.color_space = color_space
        self.phase = phase
        protocol  = str(protocol)
        assert protocol in ('1', '2-1', '2-2', '2-3', '2-4', '3-1', '3-2')

        self.protocol = str(protocol)
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
        # self.protocol_file_dir = os.path.join(data_dir_prefix, 'Protocols', 'Protocol_'+self.protocol[0])

        if self.phase == 'train':
            try:
                train_txt_path = train_txt_path_prefix + self.protocol + train_txt_path_suffix
                print('Training with protocol :', self.protocol)
                with open(train_txt_path, 'r') as f:
                    self.train_df = pd.read_csv(f, delimiter=' ', header=None, names=['img_path', 'label'])
                    self.train_img_list = list(self.train_df['img_path'])
                    self.label_list = list(self.train_df['label'])
            except:
                print('can not open Train protocol file, may be file does not exist')
                exit()

        elif self.phase == 'test':
            try:
                test_txt_path = test_txt_path_prefix + self.protocol + test_txt_path_suffix
                with open(test_txt_path, 'r') as f:
                    self.test_df = pd.read_csv(f, delimiter=' ', header=None, names=['video_path', 'label'])
                    self.video_path_list = list(self.test_df['video_path'])
                    self.label_list = list(self.test_df['label'])
                    print('len video: ', len(self.video_path_list))
                    print('len label: ', len(self.label_list))
            except:
                print('can not open Test protocol file, may be file does not exist')
                exit()

        else:
            print('phase have to be one of (train, test)')
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

            video_path = self.video_path_list[idx]
            photo_list = os.listdir(video_path)
            test_video_name = self.video_path_list[idx].split(os.sep)[-1]
            photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            image = Image.open(os.path.join(video_path, photo_list[0]))

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.phase == 'train':
            return image, label
        else:

            return image, label, test_video_name


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