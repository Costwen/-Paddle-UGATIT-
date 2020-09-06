import paddle.fluid as fluid
import numpy as np
from networks import ResnetGenerator
from PIL import Image
import os
import gobalvar as gl
from utils import denorm,tensor2numpy
fake_path = "fake"
real_source = "selfie2anime/testA"

def totensor(imgs):
    imgs = fluid.dygraph.to_variable(imgs)
    imgs = imgs / 255.
    imgs = fluid.layers.transpose(imgs, (0,3,1,2))
    imgs = fluid.layers.image_resize(imgs, (256,256))
    imgs = (imgs - 0.5) / 0.5
    return imgs


if __name__ == "__main__":
    gl._init()
    gl.set_value('rho',0)
    real_paths = os.listdir(real_source)
    with fluid.dygraph.guard():
        genA2B = ResnetGenerator(in_channels=3, out_channels=3, ngf= 64, n_blocks=4)
        genA2B_para, gen_A2B_opt = fluid.load_dygraph("Parameters/genA2B124.pdparams")
        genA2B.load_dict(genA2B_para)
        count = 0
        for real_image_path in real_paths:
            real_image_path = os.path.join(real_source, real_image_path)
            img = np.array(Image.open(real_image_path).convert("RGB")).astype(np.float32)
            img = img[np.newaxis,:,:,:]
            img = totensor(img)
            fakeA2B,_,_ = genA2B(img)
 
            a = (tensor2numpy(denorm(fakeA2B[0]))*255).astype(np.uint8)
            a = Image.fromarray(a)
            save_path = os.path.join(fake_path, "%04d"%(count)+"_fake.png")
            count += 1
            a.save(save_path)