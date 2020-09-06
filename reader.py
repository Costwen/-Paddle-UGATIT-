import paddle
import os
from PIL import  Image
import numpy as np
import random
    
'''图像随机裁剪'''
def random_crop(image, crop_shape, padding=None):
    oshape = image.size
    if padding:
        # print(image.size)
        oshape_pad = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        img_pad = image.resize(oshape_pad)
        nh = random.randint(0, oshape_pad[0] - crop_shape[0])
        nw = random.randint(0, oshape_pad[1] - crop_shape[1])
        image_crop = img_pad.crop((nh, nw, nh+crop_shape[0], nw+crop_shape[1]))
        return image_crop
    else:
        print("WARNING!!! nothing to do!!!")
        return image

'''获取图片路径'''
def generate_img_array_from_path(data_arr_A,data_arr_B,dataA_paths,dataB_paths, shuffle=False):
    n = len(dataA_paths)
    def reader():
        for i in range(n):
            if i == 0 and shuffle:
                np.random.shuffle(data_arr_A)
                np.random.shuffle(data_arr_B)
                np.random.shuffle(data_arr_B)
            img_A = Image.open(dataA_paths[data_arr_A[i]])
            img_B = Image.open(dataB_paths[data_arr_B[i]])
            if i % 3 != 0:
                img_A = random_crop(img_A,(256, 256),30)
                img_B = random_crop(img_B,(256, 256),30)
            yield np.array(img_A).astype(np.float32), np.array(img_B).astype(np.float32)
    return reader

'''reader生成'''
def reader(batch_size):
    np.random.seed(3280)
    trainA_paths = os.listdir(os.path.join('selfie2anime','trainA'))
    trainA_paths = [os.path.join('selfie2anime','trainA', path) for path in trainA_paths]
    trainB_paths = os.listdir(os.path.join('selfie2anime','trainB'))
    trainB_paths = [os.path.join('selfie2anime','trainB', path) for path in trainB_paths]
    testA_paths = os.listdir(os.path.join('selfie2anime','testA'))
    testA_paths = [os.path.join('selfie2anime','testA', path) for path in testA_paths]
    testB_paths = os.listdir(os.path.join('selfie2anime','testB'))
    testB_paths = [os.path.join('selfie2anime','testB', path) for path in testB_paths]
    
    train_arr_A = np.array(range(0, len(trainA_paths)))
    train_arr_B = np.array(range(0, len(trainB_paths)))
    test_arr = np.array(range(0, len(testA_paths)))

    train_reader = paddle.batch(generate_img_array_from_path(train_arr_A,train_arr_B, trainA_paths, trainB_paths, shuffle=True), batch_size=batch_size , drop_last= True)
    test_reader = paddle.batch(generate_img_array_from_path(test_arr, test_arr,testA_paths, testB_paths, shuffle=False), batch_size=1)
    return train_reader, test_reader
