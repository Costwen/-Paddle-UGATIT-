from networks import ResnetGenerator,Discriminator
import time
from reader import reader
import paddle.fluid as fluid
from loss import mse_loss,bce_loss
from utils import denorm,tensor2numpy,cam,totensor
from PIL import Image
import numpy as np
import gobalvar as gl
class UGATIT():
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'
        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.start = args.start
        self.pretrain = args.pretrain

        self.lr1 = fluid.layers.polynomial_decay(args.lr, 1000000, 1e-9, 1)
        self.lr2 = fluid.layers.polynomial_decay(args.lr, 1000000, 1e-9, 1)
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        print()
        print("##### Information #####")
        print("# light : ", self.light)
        # print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        # print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)
        
    '''将rho层的参数限制在[0,1]'''
    def fileter_func(Parameter):
        return Parameter.name.count('rho')

    def build_model(self):
        '''DataLoader'''
        gl._init()
        gl.set_value('rho',0)
        l2 = fluid.regularizer.L2Decay(self.weight_decay)
        self.train_reader, self.test_reader = reader(self.batch_size)
        self.genA2B = ResnetGenerator(in_channels=3, out_channels=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(in_channels=3, out_channels=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(in_channels=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(in_channels=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(in_channels=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(in_channels=3, ndf=self.ch, n_layers=5)
        self.clip = fluid.clip.GradientClipByValue(1,0,need_clip=self.fileter_func)
        self.G_opt = fluid.optimizer.Adam(learning_rate=self.lr1, beta1=0.5, beta2=0.999, 
        regularization=l2, parameter_list=self.genA2B.parameters()+self.genB2A.parameters())
        self.D_opt = fluid.optimizer.Adam(learning_rate=self.lr2, beta1=0.5, beta2=0.999, 
        regularization=l2, parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters())
        self.L1loss = fluid.dygraph.L1Loss()
        self.BCELoss = fluid.dygraph.BCELoss()
    def train(self):
        epochs = 1000
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        print('training start !')
        start_time = time.time()

        '''加载预训练模型'''
        if self.pretrain:
            str_genA2B = "Parameters/genA2B%03d.pdparams"%(self.start-1)
            str_genB2A = "Parameters/genB2A%03d.pdparams"%(self.start-1)
            str_disGA = "Parameters/disGA%03d.pdparams"%(self.start-1)
            str_disGB = "Parameters/disGB%03d.pdparams"%(self.start-1)
            str_disLA = "Parameters/disLA%03d.pdparams"%(self.start-1)
            str_disLB = "Parameters/disLB%03d.pdparams"%(self.start-1)
            genA2B_para, gen_A2B_opt = fluid.load_dygraph(str_genA2B)
            genB2A_para, gen_B2A_opt = fluid.load_dygraph(str_genB2A)
            disGA_para, disGA_opt = fluid.load_dygraph(str_disGA)
            disGB_para, disGB_opt = fluid.load_dygraph(str_disGB)
            disLA_para, disLA_opt = fluid.load_dygraph(str_disLA)
            disLB_para, disLB_opt = fluid.load_dygraph(str_disLB)
            self.genA2B.load_dict(genA2B_para)
            self.genB2A.load_dict(genB2A_para)
            self.disGA.load_dict(disGA_para)
            self.disGB.load_dict(disGB_para)
            self.disLA.load_dict(disLA_para)
            self.disLB.load_dict(disLB_para)
        for epoch in range(self.start,epochs):
            for block_id, data in enumerate(self.train_reader()):
                real_A = np.array([x[0] for x in data], np.float32)
                real_B = np.array([x[1] for x in data], np.float32)
                real_A = totensor(real_A,block_id,'train')
                real_B = totensor(real_B,block_id,'train')

                # Update D

                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = mse_loss(1, real_GA_logit) + mse_loss(0, fake_GA_logit)
                D_ad_cam_loss_GA = mse_loss(1, real_GA_cam_logit) + mse_loss(0, fake_GA_cam_logit)
            
                D_ad_loss_LA = mse_loss(1, real_LA_logit) + mse_loss(0, fake_LA_logit)
                D_ad_cam_loss_LA = mse_loss(1, real_LA_cam_logit) + mse_loss(0, fake_LA_cam_logit)

                D_ad_loss_GB = mse_loss(1, real_GB_logit) + mse_loss(0, fake_GB_logit)
                D_ad_cam_loss_GB = mse_loss(1, real_GB_cam_logit) + mse_loss(0, fake_GB_cam_logit)
            
                D_ad_loss_LB = mse_loss(1, real_LB_logit) + mse_loss(0, fake_LB_logit)
                D_ad_cam_loss_LB = mse_loss(1, real_LB_cam_logit) + mse_loss(0, fake_LB_cam_logit)

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
    
                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_opt.minimize(Discriminator_loss)
                self.disGA.clear_gradients(), self.disGB.clear_gradients(), self.disLA.clear_gradients(), self.disLB.clear_gradients()

                # Update G

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
                print("fake_A2B.shape:",fake_A2B.shape)
                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                
                G_ad_loss_GA =  mse_loss(1, fake_GA_logit)
                G_ad_cam_loss_GA = mse_loss(1, fake_GA_cam_logit)

                G_ad_loss_LA =  mse_loss(1, fake_LA_logit)
                G_ad_cam_loss_LA = mse_loss(1, fake_LA_cam_logit)

                G_ad_loss_GB =  mse_loss(1, fake_GB_logit)
                G_ad_cam_loss_GB = mse_loss(1, fake_GB_cam_logit)

                G_ad_loss_LB =  mse_loss(1, fake_LB_logit)
                G_ad_cam_loss_LB = mse_loss(1, fake_LB_cam_logit)
                
                G_recon_loss_A = self.L1loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1loss(fake_B2B, real_B)

                G_cam_loss_A = bce_loss(1, fake_B2A_cam_logit) + bce_loss(0, fake_A2A_cam_logit)
                G_cam_loss_B = bce_loss(1, fake_A2B_cam_logit) + bce_loss(0, fake_B2B_cam_logit)

                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_opt.minimize(Generator_loss)
                self.genA2B.clear_gradients(), self.genB2A.clear_gradients()
                
                print("[%5d/%5d] time: %4.4f d_loss: %.5f, g_loss: %.5f" % (epoch, block_id, time.time() - start_time, Discriminator_loss.numpy(), Generator_loss.numpy()))
                print("G_loss_A: %.5f G_loss_B: %.5f"%(G_loss_A.numpy(), G_loss_B.numpy()))
                print("G_ad_loss_GA: %.5f   G_ad_loss_GB: %.5f"%(G_ad_loss_GA.numpy(), G_ad_loss_GB.numpy()))
                print("G_ad_loss_LA: %.5f   G_ad_loss_LB: %.5f"%(G_ad_loss_LA.numpy(), G_ad_loss_LB.numpy()))
                print("G_cam_loss_A:%.5f  G_cam_loss_B:%.5f"%(G_cam_loss_A.numpy(), G_cam_loss_B.numpy()))
                print("G_recon_loss_A:%.5f  G_recon_loss_B:%.5f"%(G_recon_loss_A.numpy(), G_recon_loss_B.numpy()))
                print("G_identity_loss_A:%.5f  G_identity_loss_B:%.5f"%(G_identity_loss_B.numpy(), G_identity_loss_B.numpy()))
                
                if epoch % 2== 1 and block_id % self.print_freq == 0:

                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    # B2A = np.zeros((self.img_size * 7, 0, 3))
                    for eval_id, eval_data in enumerate(self.test_reader()):
                        if eval_id == 10:
                            break
                        real_A = np.array([x[0] for x in eval_data], np.float32)
                        real_B = np.array([x[1] for x in eval_data], np.float32)
                        real_A = totensor(real_A,eval_id, 'eval')
                        real_B = totensor(real_B,eval_id, 'eval')
                        
                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        a = tensor2numpy(denorm(real_A[0]))
                        b = cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size)
                        c = tensor2numpy(denorm(fake_A2A[0]))
                        d = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                        e = tensor2numpy(denorm(fake_A2B[0]))
                        f = cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size)
                        g = tensor2numpy(denorm(fake_A2B2A[0]))
                        A2B = np.concatenate((A2B,(np.concatenate((a,b,c,d,e,f,g))*255).astype(np.uint8)),1).astype(np.uint8)
                    A2B = Image.fromarray(A2B)
                    A2B.save('Images/%d_%d.png'%(epoch,block_id))
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            if epoch%4 == 0:
                    fluid.save_dygraph(self.genA2B.state_dict(), "Parameters/genA2B%03d"%(epoch))
                    fluid.save_dygraph(self.genB2A.state_dict(), "Parameters/genB2A%03d"%(epoch))
                    fluid.save_dygraph(self.disGA.state_dict(), "Parameters/disGA%03d"%(epoch))
                    fluid.save_dygraph(self.disGB.state_dict(), "Parameters/disGB%03d"%(epoch))
                    fluid.save_dygraph(self.disLA.state_dict(), "Parameters/disLA%03d"%(epoch))
                    fluid.save_dygraph(self.disLB.state_dict(), "Parameters/disLB%03d"%(epoch))
                    


