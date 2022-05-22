import random
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision.transforms import transforms

class TransformColorSpace(object):
    '''
    转换图片的颜色空间
    接受Image打开的图像类型，返回也是Image打开的图像类型
    '''
    def __init__(self, color_space='RGB'):
        assert color_space.lower() in ('rgb', 'ycbcr', 'hsv', 'ycrcb')
        self.color_space = color_space.lower()

    def __call__(self, img):
        if self.color_space == 'rgb':
            return img.convert('RGB')
        elif self.color_space == 'ycbcr':
            img_array = np.asarray(img)
            img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            # 交换cb和cr通道
            img_ycbcr = img_ycrcb.copy()
            img_ycbcr[:, :, 1] = img_ycrcb[:, :, 2]
            img_ycbcr[:, :, 2] = img_ycrcb[:, :, 1]
            return Image.fromarray(img_ycbcr)

        elif self.color_space == 'ycrcb':
            img_array = np.asarray(img)
            img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            return Image.fromarray(img_ycrcb)

        elif self.color_space == 'hsv':
            img_array = np.asarray(img)
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            return Image.fromarray(img_hsv)

# test_transform = transforms.Compose([
#     TransformColorSpace('ycbcr'),
#     transforms.ToTensor()
# ])
# img_path = 'C:/Users/DELL/Desktop/6x.png'
# image = Image.open(img_path)
# image = test_transform(image)
# image = np.asarray(image)
# print(image.shape)
# print(image)
# # image.show()








# class AddPepperNoise(object):
#     """增加椒盐噪声
#     Args:
#         snr （float）: Signal Noise Rate
#         p (float): 概率值，依概率执行该操作
#     """
#
#     def __init__(self, snr, p=0.9):
#         assert isinstance(snr, float) or (isinstance(p, float))
#         self.snr = snr
#         self.p = p
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): PIL Image
#         Returns:
#             PIL Image: PIL image.
#         """
#         if random.uniform(0, 1) < self.p:
#             img_ = np.array(img).copy()
#             h, w, c = img_.shape
#             signal_pct = self.snr
#             noise_pct = (1 - self.snr)
#             mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
#             mask = np.repeat(mask, c, axis=2)
#             img_[mask == 1] = 255   # 盐噪声
#             img_[mask == 2] = 0     # 椒噪声
#             return Image.fromarray(img_.astype('uint8')).convert('RGB')
#         else:
#             return img
#
#
# class ColorAugmentation(object):
#     # 这是加在.ToTensor()之后的图像处理
#     def __init__(self):
#         self.eig_vec = torch.Tensor([
#             [0.4009, 0.7192, -0.5675],
#             [-0.8140, -0.0045, -0.5808],
#             [0.4203, -0.6948, -0.5836],
#         ])
#         self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
#
#     def __call__(self, tensor):
#         assert tensor.size(0) == 3
#         alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
#         quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
#         tensor = tensor + quatity.view(3, 1, 1)
#         return tensor