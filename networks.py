import paddle.fluid as fluid
import numpy as np
import gobalvar as gl
import sys
sys.setrecursionlimit(10000)
class PadConvInRelu(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride = 1, padding = 0, has_bias = False, has_relu=True):
        super(PadConvInRelu, self).__init__()
        self.padding = padding
        self.conv = fluid.dygraph.Conv2D(in_channels, out_channels, filter_size, stride=stride, bias_attr=has_bias)
        self.has_relu = has_relu
    def forward(self, input):
        x = fluid.layers.pad2d(input, paddings=[self.padding, self.padding, self.padding, self.padding], mode='reflect')
        x = self.conv(x)
        x = fluid.layers.instance_norm(x)
        if self.has_relu:
            x = fluid.layers.relu(x)
        return x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, in_channels, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
    default_initializer=fluid.initializer.ConstantInitializer(0.9)
    ,name = 'rho_'+str(gl.get_value('rho')))
        gl.set_value('rho',gl.get_value('rho')+1)
    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input, gamma, beta):
        
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        ex_rho = fluid.layers.expand(self.rho, expand_times = [input.shape[0], 1, 1, 1])
        # ex_rho = self.rho
        # print("rho",fluid.layers.reduce_max(self.rho).numpy())
        out = ex_rho * out_in + (1 - ex_rho)*out_ln
        out = out * gamma + beta
        return out

class ILN(fluid.dygraph.Layer):
    def __init__(self, in_channels, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
    default_initializer=fluid.initializer.ConstantInitializer(0)
    ,name = 'rho_'+str(gl.get_value('rho')))
        gl.set_value('rho',gl.get_value('rho')+1)
        self.gamma = fluid.layers.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = fluid.layers.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))

    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        ex_rho = fluid.layers.expand(self.rho, expand_times = [input.shape[0], 1, 1, 1])
        ex_gamma = fluid.layers.expand(self.gamma, expand_times = [input.shape[0], 1, 1, 1])
        ex_beta = fluid.layers.expand(self.beta, expand_times = [input.shape[0], 1, 1, 1])
        # ex_rho = self.rho
        # ex_gamma = self.gamma
        # ex_beta = self.beta
        out = ex_rho * out_in + (1 - ex_rho) * out_ln
        out = out * ex_gamma + ex_beta

        return out

class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, in_channels):
        super(ResnetAdaILNBlock, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(in_channels, in_channels, filter_size=3, stride=1, bias_attr=False)
        self.conv2 = fluid.dygraph.Conv2D(in_channels, in_channels, filter_size=3, stride=1, bias_attr=False)
        self.norm1 = adaILN(in_channels)
        self.norm2 = adaILN(in_channels)
    def forward(self, input, gamma, beta):
        x = fluid.layers.pad2d(input, [1,1,1,1], mode='reflect')
        x = self.conv1(x)
        x = self.norm1(x, gamma, beta)
        x = fluid.layers.relu(x)
        x = fluid.layers.pad2d(x, [1,1,1,1], mode='reflect')
        x = self.conv2(x)
        x = self.norm2(x, gamma, beta)
        return fluid.layers.elementwise_add(x, input)
        

class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, in_channels, has_bias=False):
        super(ResnetBlock, self).__init__()
        self.PadConvInRelu1 = PadConvInRelu(in_channels, in_channels, filter_size=3, stride=1, padding=1, has_bias=has_bias)
        self.PadConvIn2 = PadConvInRelu(in_channels, in_channels, filter_size=3, stride=1, padding=1, has_relu=False)
    def forward(self, input):
        x = self.PadConvInRelu1(input)
        x = self.PadConvIn2(x)
        return fluid.layers.elementwise_add(x, input)

class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()
    def forward(self, x):
        x = fluid.layers.relu(x)
        return x

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale
    def forward(self, x):
        x = fluid.layers.interpolate(x, scale=self.scale, resample='NEAREST')
        return x

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, paddings=1):
        super(ReflectionPad2d, self).__init__()
        self.paddings = paddings
    def forward(self, x):
        pad = self.paddings
        x = fluid.layers.pad2d(x, [pad, pad, pad, pad], mode='reflect')
        return x
class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, x):
        x = fluid.layers.tanh(x)
        return x

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks>=0)
        super(ResnetGenerator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        '''DownBlock'''
        DownBlock = []
        DownBlock += [PadConvInRelu(in_channels, ngf, filter_size=7, stride=1, padding=3)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [PadConvInRelu(ngf*mult, ngf*mult*2, filter_size=3, stride=2, padding=1)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf*mult)]

        self.DownBlock = fluid.dygraph.Sequential(*DownBlock)
        self.gap_fc = fluid.dygraph.Linear(ngf*mult, 1, bias_attr=False)
        self.gmp_fc = fluid.dygraph.Linear(ngf*mult, 1, bias_attr=False)
        self.conv1x1 = fluid.dygraph.Conv2D(ngf*mult*2, ngf*mult, 1, 1, bias_attr=True)
        
        if self.light:
            FC = [fluid.dygraph.Linear(ngf*mult, ngf*mult, bias_attr=False),
                  ReLU(),
                  fluid.dygraph.Linear(ngf*mult, ngf*mult, bias_attr=False),
                  ReLU(),
                 ]
        else:
            FC = [fluid.dygraph.Linear(img_size // mult * img_size // mult * ngf * mult, ngf*mult, bias_attr=False),
                ReLU(),
                fluid.dygraph.Linear(ngf*mult, ngf*mult, bias_attr=False),
                ReLU(),
                ]
        self.FC = fluid.dygraph.Sequential(*FC)
        self.gamma = fluid.dygraph.Linear(ngf*mult, ngf*mult, bias_attr=False)
        self.beta = fluid.dygraph.Linear(ngf*mult, ngf*mult, bias_attr=False)
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_'+str(i+1), ResnetAdaILNBlock(ngf*mult)) 

        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            UpBlock2 += [Upsample(scale=2),
                            ReflectionPad2d(1),
                            fluid.dygraph.Conv2D(ngf*mult, int(ngf*mult/2), 3, 1, 0, bias_attr=False),
                            ILN(int(ngf*mult/2)),
                            ReLU()
                            ]
        UpBlock2 += [ReflectionPad2d(3),
                     fluid.dygraph.Conv2D(ngf, out_channels, filter_size=7, stride=1, bias_attr=False),
                     Tanh()]
        self.UpBlock2 = fluid.dygraph.Sequential(*UpBlock2)

    def forward(self, input):
        feature_map = self.DownBlock(input)
        gap = fluid.layers.adaptive_pool2d(feature_map, pool_size=1, pool_type='avg')
        gmp = fluid.layers.adaptive_pool2d(feature_map, pool_size=1, pool_type='max')
        gap = fluid.layers.reshape(gap, shape=[gap.shape[0], -1])
        gmp = fluid.layers.reshape(gmp, shape=[gmp.shape[0], -1])
        
        gap_logit = self.gap_fc(gap)
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = fluid.layers.reshape(gap_weight, shape=[1,-1,1,1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight =  self.gmp_fc.parameters()[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, shape=[1,-1,1,1])

        gap = feature_map * gap_weight
        gmp = feature_map * gmp_weight
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], axis=1)
        
        x = fluid.layers.concat([gap, gmp], axis=1)
        x = fluid.layers.relu(self.conv1x1(x))
        heat_map = fluid.layers.reduce_sum(x, dim = [1] , keep_dim=True)
        
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='max')
            x_ = self.FC(fluid.layers.reshape(x_, (x_.shape[0], -1)))
        else:
            # print(fluid.layers.reshape(x, (x.shape[0], -1)).shape)
            x_ = self.FC(fluid.layers.reshape(x, (x.shape[0], -1)))
        
        gamma, beta = fluid.layers.reshape(self.gamma(x_),shape=[x.shape[0], x.shape[1], 1, 1]), fluid.layers.reshape(self.beta(x_),shape=[x.shape[0], x.shape[1], 1, 1])
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma,beta) # 所有的用同一个gamma和beta?
        out = self.UpBlock2(x)
        
        return  out, cam_logit, heat_map

class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, value=0.2):
        super(LeakyReLU, self).__init__()
        self.value = value
    def forward(self, x):
        x = fluid.layers.leaky_relu(x, self.value)
        return x

class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,layer,dim=0,power_iters=1,eps=1e-12,dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype) 
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, in_channels, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ ReflectionPad2d(1),
                Spectralnorm(fluid.dygraph.Conv2D(in_channels, ndf, filter_size=4, stride=2, bias_attr=True)),
                LeakyReLU(0.2)
                ]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ ReflectionPad2d(1),
                        Spectralnorm(fluid.dygraph.Conv2D(ndf*mult, ndf*mult*2, filter_size=3, stride=2, bias_attr=True)),
                        LeakyReLU(0.2)
            ]
        mult = 2 ** (n_layers -2 - 1)
        model += [ReflectionPad2d(1),
                Spectralnorm(fluid.dygraph.Conv2D(ndf*mult, ndf*mult*2, filter_size=4, stride=1, bias_attr=True)),
                LeakyReLU(0.2)
        ]
        self.model = fluid.dygraph.Sequential(*model)
        mult = 2 ** (n_layers -2)
        self.gap_fc = Spectralnorm(fluid.dygraph.Linear(ndf*mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(fluid.dygraph.Linear(ndf*mult, 1, bias_attr=False))
        self.conv1x1 = fluid.dygraph.Conv2D(ndf*mult*2, ndf*mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLU()
        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(fluid.dygraph.Conv2D(ndf*mult, 1, filter_size=4, stride=1, bias_attr=False))

    def forward(self, input):
        feature_map = self.model(input)
        gap = fluid.layers.adaptive_pool2d(feature_map, pool_size=1, pool_type='avg')
        gmp = fluid.layers.adaptive_pool2d(feature_map, pool_size=1, pool_type='max')
        
        gap = fluid.layers.reshape(gap, shape=[gap.shape[0], -1])
        gmp = fluid.layers.reshape(gmp, shape=[gmp.shape[0], -1])
        gap_logit = self.gap_fc(gap)
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = fluid.layers.reshape(gap_weight, shape=[1,-1,1,1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight =  self.gmp_fc.parameters()[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, shape=[1,-1,1,1])
        
        gap = feature_map * gap_weight
        gmp = feature_map * gmp_weight
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], axis=1)

        x = fluid.layers.concat([gap, gmp], axis=1)
        x = self.leaky_relu(self.conv1x1(x))
        
        heat_map = fluid.layers.reduce_sum(x, dim=[1], keep_dim=True)
        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logit, heat_map

if __name__ == "__main__":
    with fluid.dygraph.guard():
        input = np.ones((3,3,256,256),dtype=np.float32)
        input = fluid.dygraph.to_variable(input)
        # model = ResnetGenerator(3,3)
        model = Discriminator(3)
        output,_,_ = model(input)
        
