from UGATIT import UGATIT
import paddle.fluid as fluid
class args():
    def __init__(self):
        '''是否使用轻量级模型'''
        self.light = False
        '''一次训练的batchsize大小'''
        self.batch_size = 1
    
        '''输出结果的频率'''
        self.print_freq = 200
        self.lr = 0.0001
        self.weight_decay = 0.0001
        self.adv_weight = 1
        self.cycle_weight = 10
        self.cam_weight = 1000
        self.identity_weight = 10
        '''开始训练的轮数'''
        self.start = 133
        '''是否加载预训练模型'''
        self.pretrain = True
        self.ch = 64
        self.n_res = 4
        self.n_dis = 6
        self.img_size = 256
        self.img_ch = 3
"""main"""
def main():
    config = args()
    # open session
    gan = UGATIT(config)
    # build graph
    gan.build_model()
    gan.train()
if __name__ == '__main__':
    with fluid.dygraph.guard():
        main()
