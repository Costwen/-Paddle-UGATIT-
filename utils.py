import numpy as np
import paddle.fluid as fluid
import cv2
def denorm(x):
    return x * 0.5 + 0.5
    
def tensor2numpy(x):
    x = fluid.layers.transpose(x, (1,2,0))
    return x.numpy()

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def totensor(imgs, id, type):
    imgs = fluid.dygraph.to_variable(imgs)
    imgs = imgs / 255.
    imgs = fluid.layers.transpose(imgs, (0,3,1,2))
    if type == 'train':
        imgs = fluid.layers.image_resize(imgs, (256,256))
        if id%3==0:
            imgs = fluid.layers.flip(imgs, [3])
        imgs = (imgs - 0.5) / 0.5
    else :
        imgs = fluid.layers.image_resize(imgs, (256,256))
        imgs = (imgs - 0.5) / 0.5
    return imgs
