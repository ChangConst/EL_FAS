import os
import pandas as pd

train_imgs_dir = '/home/chang/FAS_DATASET/OULU_NPU/Train_imgs/cropped_face/'
protocol_dir = '/home/chang/FAS_DATASET/OULU_NPU/Protocols/'

train_data_dir = '/home/chang/FAS_DATASET/OULU_NPU/Train_imgs/cropped_face/'
destination = '/home/chang/FAS_Project/generateTrain/'
def generateOULU_NPUTrainTXT(protocol, destination, scale=1.0):

    txt_name = 'OULU_NPUTrain_P' + protocol + '.txt'
    protocol_list = protocol.split('-')
    protocol_file_path = os.path.join(protocol_dir, 'Protocol_'+protocol_list[0])
    if len(protocol_list) == 2:
        protocol_file_path = os.path.join(protocol_file_path, 'Train_'+protocol_list[1]+'.txt')
    else:
        protocol_file_path = os.path.join(protocol_file_path, 'Train.txt')

    with open(protocol_file_path, 'r') as f:
        protocol_df = pd.read_csv(f, delimiter=',', header=None, names=['label', 'video_name'])
        video_name_list = list(protocol_df['video_name'])
        label_list = list(protocol_df['label'])
        label_list = [1 if x == 1 else 0 for x in label_list]

    with open(os.path.join(destination, txt_name), 'w') as f:
        for i in range(len(video_name_list)):
            video_name = video_name_list[i]+'.avi'
            print('process : ', video_name)
            label = label_list[i]

            img_dir = os.path.join(train_data_dir, str(scale), video_name)
            img_list = os.listdir(img_dir)
            for img_name in img_list:
                line = os.path.join(img_dir, img_name)+' '+str(label)+'\n'
                f.write(line)

if __name__ == '__main__':
    generateOULU_NPUTrainTXT('3-2', destination=destination, scale=1.0)
