import os

LIVE_LABEL = 1
SPOOF_LABEL = 0
SKIP_PHOTO = 10

db_dir = '/home/chang/FAS_DATASET/SiW_frames/'
save_dir = 'SiWTrainTXT'
train_live_path = os.path.join(db_dir, 'Train/live')
train_spoof_path = os.path.join(db_dir, 'Train/spoof')
test_live_path = os.path.join(db_dir, 'Test/live')
test_spoof_path = os.path.join(db_dir, 'Test/spoof')
train_live_subjects = os.listdir(train_live_path)
train_spoof_subjects = os.listdir(train_spoof_path)
test_live_subjects = os.listdir(test_live_path)
test_spoof_subjects = os.listdir(test_spoof_path)


def generateP1():
    protocol_train_name = 'SiWTrainP1.txt'
    protocol_test_name = 'SiWTestP1.txt'

    protocol_train_file = open(os.path.join(save_dir, protocol_train_name), 'w')
    protocol_test_file = open(os.path.join(save_dir, protocol_test_name), 'w')

    for subject in train_live_subjects:
        video_name_list = os.listdir(os.path.join(train_live_path, subject))
        for video_name in video_name_list:
            photo_list = os.listdir(os.path.join(train_live_path, subject, video_name))
            photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            for photo in photo_list:
                frame_number = int(photo.split('_')[1].split('.')[0])
                if frame_number < 60:
                    line = os.path.join(train_live_path, subject, video_name, photo) + ' ' + str(LIVE_LABEL) + '\n'
                    protocol_train_file.write(line)

    for subject in train_spoof_subjects:
        video_name_list = os.listdir(os.path.join(train_spoof_path, subject))
        for video_name in video_name_list:
            photo_list = os.listdir(os.path.join(train_spoof_path, subject, video_name))
            photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            for photo in photo_list:
                frame_number = int(photo.split('_')[1].split('.')[0])
                if frame_number < 60:
                    line = os.path.join(train_spoof_path, subject, video_name, photo) + ' ' + str(SPOOF_LABEL) + '\n'
                    protocol_train_file.write(line)

    for subject in test_live_subjects:
        video_name_list = os.listdir(os.path.join(test_live_path, subject))
        for video_name in video_name_list:
            pic_list = os.listdir(os.path.join(test_live_path, subject, video_name))
            pic_list = [ele for ele in pic_list if ele.endswith('jpg')]
            if len(pic_list) > 0:
                line = os.path.join(test_live_path, subject, video_name) + ' ' + str(LIVE_LABEL) + '\n'
                protocol_test_file.write(line)

    for subject in test_spoof_subjects:
        video_name_list = os.listdir(os.path.join(test_spoof_path, subject))
        for video_name in video_name_list:
            pic_list = os.listdir(os.path.join(test_spoof_path, subject, video_name))
            pic_list = [ele for ele in pic_list if ele.endswith('jpg')]
            if len(pic_list) > 0:
                line = os.path.join(test_spoof_path, subject, video_name) + ' ' + str(SPOOF_LABEL) + '\n'
                protocol_test_file.write(line)

    protocol_train_file.close()
    protocol_test_file.close()

def generateP2(skip_photo=1):
    train_mediums = [[1, 2, 3], [1, 3, 4], [2, 3, 4], [1, 2, 4]]
    test_mediums = [4, 2, 1, 3]
    for i in range(len(test_mediums)):
        protocol_train_name = 'SiWTrainP2-' + str(i+1) + '.txt'
        protocol_test_name = 'SiWTestP2-' + str(i+1) + '.txt'

        protocol_train_file = open(os.path.join(save_dir, protocol_train_name), 'w')
        protocol_test_file = open(os.path.join(save_dir, protocol_test_name), 'w')

        for subject in train_spoof_subjects:
            video_name_list = os.listdir(os.path.join(train_spoof_path, subject))
            # filter video_name
            video_name_list = [ele for ele in video_name_list if int(ele.split('-')[3]) in train_mediums[i]]
            # print(video_name_list)
            for video_name in video_name_list:
                photo_list = os.listdir(os.path.join(train_spoof_path, subject, video_name))
                photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                photo_index = 0
                while photo_index < len(photo_list):
                    line = os.path.join(train_spoof_path, subject, video_name, photo_list[photo_index]) + ' ' + str(SPOOF_LABEL) + '\n'
                    protocol_train_file.write(line)

                    photo_index += skip_photo

        for subject in train_live_subjects:
            video_name_list = os.listdir(os.path.join(train_live_path, subject))
            for video_name in video_name_list:
                photo_list = os.listdir(os.path.join(train_live_path, subject, video_name))
                photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                photo_index = 0
                while photo_index < len(photo_list):
                    line = os.path.join(train_live_path, subject, video_name, photo_list[photo_index]) + ' ' + str(LIVE_LABEL) + '\n'
                    protocol_train_file.write(line)

                    photo_index += skip_photo

        for subject in test_spoof_subjects:
            video_name_list = os.listdir(os.path.join(test_spoof_path, subject))
            # filter video name
            video_name_list = [ele for ele in video_name_list if int(ele.split('-')[3]) == test_mediums[i]]

            for video_name in video_name_list:
                pic_list = os.listdir(os.path.join(test_spoof_path, subject, video_name))
                pic_list = [ele for ele in pic_list if ele.endswith('jpg')]

                if len(pic_list) > 0:
                    line = os.path.join(test_spoof_path, subject, video_name) + ' ' + str(SPOOF_LABEL) + '\n'
                    protocol_test_file.write(line)


        for subject in test_live_subjects:
            video_name_list = os.listdir(os.path.join(test_live_path, subject))
            for video_name in video_name_list:
                pic_list = os.listdir(os.path.join(test_live_path, subject, video_name))
                pic_list = [ele for ele in pic_list if ele.endswith('jpg')]

                if len(pic_list) > 0:
                    line = os.path.join(test_live_path, subject, video_name) + ' ' + str(LIVE_LABEL) + '\n'
                    protocol_test_file.write(line)


        protocol_train_file.close()
        protocol_test_file.close()

def generateP3(skip_photo=1):
    train_types = [2, 3]
    test_types = [3, 2]
    for i in range(len(train_types)):
        protocol_train_name = 'SiWTrainP3-' + str(i + 1) + '.txt'
        protocol_test_name = 'SiWTestP3-' + str(i + 1) + '.txt'

        protocol_train_file = open(os.path.join(save_dir, protocol_train_name), 'w')
        protocol_test_file = open(os.path.join(save_dir, protocol_test_name), 'w')

        for subject in train_spoof_subjects:
            video_name_list = os.listdir(os.path.join(train_spoof_path, subject))
            # filter video_name
            video_name_list = [ele for ele in video_name_list if int(ele.split('-')[2])==train_types[i]]
            # print(video_name_list)
            for video_name in video_name_list:
                photo_list = os.listdir(os.path.join(train_spoof_path, subject, video_name))
                photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                photo_index = 0
                while photo_index < len(photo_list):
                    line = os.path.join(train_spoof_path, subject, video_name, photo_list[photo_index]) + ' ' + str(
                        SPOOF_LABEL) + '\n'
                    protocol_train_file.write(line)

                    photo_index += skip_photo

        for subject in train_live_subjects:
            video_name_list = os.listdir(os.path.join(train_live_path, subject))
            for video_name in video_name_list:
                photo_list = os.listdir(os.path.join(train_live_path, subject, video_name))
                photo_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                photo_index = 0
                while photo_index < len(photo_list):
                    line = os.path.join(train_live_path, subject, video_name, photo_list[photo_index]) + ' ' + str(
                        LIVE_LABEL) + '\n'
                    protocol_train_file.write(line)

                    photo_index += skip_photo

        for subject in test_spoof_subjects:
            video_name_list = os.listdir(os.path.join(test_spoof_path, subject))
            video_name_list = [ele for ele in video_name_list if int(ele.split('-')[2]) == test_types[i]]
            for video_name in video_name_list:
                pic_list = os.listdir(os.path.join(test_spoof_path, subject, video_name))
                pic_list = [ele for ele in pic_list if ele.endswith('jpg')]

                if len(pic_list) > 0:
                    line = os.path.join(test_spoof_path, subject, video_name) + ' ' + str(SPOOF_LABEL) + '\n'
                    protocol_test_file.write(line)

        for subject in test_live_subjects:
            video_name_list = os.listdir(os.path.join(test_live_path, subject))
            for video_name in video_name_list:
                pic_list = os.listdir(os.path.join(test_live_path, subject, video_name))
                pic_list = [ele for ele in pic_list if ele.endswith('jpg')]

                if len(pic_list) > 0:
                    line = os.path.join(test_live_path, subject, video_name) + ' ' + str(LIVE_LABEL) + '\n'
                    protocol_test_file.write(line)

        protocol_train_file.close()
        protocol_test_file.close()

if __name__ == '__main__':
    generateP1()
    generateP2(skip_photo=SKIP_PHOTO)
    generateP3(skip_photo=SKIP_PHOTO)