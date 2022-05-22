import os

LIVE_LABEL = 1
SPOOF_LABEL = 0
SKIP_PHOTO = 10
db_dir = '/home/chang/FAS_DATASET/SiW_frames/'
save_dir = './'
train_live_path = os.path.join(db_dir, 'Train/live')
train_spoof_path = os.path.join(db_dir, 'Train/spoof')
test_live_path = os.path.join(db_dir, 'Test/live')
test_spoof_path = os.path.join(db_dir, 'Test/spoof')
train_live_subjects = os.listdir(train_live_path)
train_spoof_subjects = os.listdir(train_spoof_path)
test_live_subjects = os.listdir(test_live_path)
test_spoof_subjects = os.listdir(test_spoof_path)


def generateTXT():
    protocol_train_name = 'SiW2OULUTrain.txt'
    # protocol_test_name = 'SiWTestP1.txt'

    protocol_train_file = open(os.path.join(save_dir, protocol_train_name), 'w')
    # protocol_test_file = open(os.path.join(save_dir, protocol_test_name), 'w')

    for subject in train_live_subjects:
        video_name_list = os.listdir(os.path.join(train_live_path, subject))
        # print(video_name_list)
        for video_name in video_name_list:
            photo_list = os.listdir(os.path.join(train_live_path, subject, video_name))
            photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            photo_index = 0
            while photo_index < len(photo_list):
                line = os.path.join(train_live_path, subject, video_name, photo_list[photo_index]) + ' ' + str(LIVE_LABEL) + '\n'
                protocol_train_file.write(line)

                photo_index += SKIP_PHOTO

    for subject in train_spoof_subjects:
        video_name_list = os.listdir(os.path.join(train_spoof_path, subject))
        # print(video_name_list)
        for video_name in video_name_list:
            photo_list = os.listdir(os.path.join(train_spoof_path, subject, video_name))
            photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            photo_index = 0
            while photo_index < len(photo_list):
                line = os.path.join(train_spoof_path, subject, video_name, photo_list[photo_index]) + ' ' + str(
                    SPOOF_LABEL) + '\n'
                protocol_train_file.write(line)

                photo_index += SKIP_PHOTO

    protocol_train_file.close()

if __name__ == '__main__':
    generateTXT()
