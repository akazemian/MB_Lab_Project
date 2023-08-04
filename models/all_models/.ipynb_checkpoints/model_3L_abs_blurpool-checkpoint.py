from models.layer_operations.convolution import StandardConvolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
from models.layer_operations.random_proj import RandomProjection
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                c1_bp: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                c2_bp: nn.Module,
                mp2: nn.Module,
                c3: nn.Module,
                c3_bp: nn.Module,
                mp3: nn.Module,
                batches_3: int,
                nl1: nn.Module,
                global_mp: bool,
                rp: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.c1_bp = c1_bp 
        self.mp1 = mp1
        
        self.c2 = c2
        self.c2_bp = c2_bp
        self.mp2 = mp2
        
        self.c3 = c3
        self.c3_bp = c3_bp
        self.mp3 = mp3
        self.batches_3 = batches_3
        
        self.nl1 = nl1
        self.global_mp = global_mp
        self.rp = rp
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
        x = self.nl1(x)
        
        x = self.c1_bp(x)
        if self.print_shape:
            print('c1 bp', x.shape)
             
        x = self.mp1(x)
        if self.print_shape:
            print('mp1', x.shape)
            
            
        #conv layer 2
        x = self.c2(x)
        if self.print_shape:
            print('conv2', x.shape)        
            
        x = self.nl1(x)
        
        x = self.c2_bp(x)
        if self.print_shape:
            print('c2 bp', x.shape)
        
        x = self.mp2(x)
        if self.print_shape:
            print('mp2', x.shape)
            
            
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        #conv layer 3
        conv_3 = []
        for i in range(self.batches_3):
            conv_3.append(self.c3(x).to(device)) 
        x = torch.cat(conv_3,dim=1)
        if self.print_shape:
            print('conv3', x.shape)

            
        
        x = self.nl1(x)
        
        x = self.c3_bp(x)
        if self.print_shape:
            print('c3 bp', x.shape)
        
        x = self.mp3(x)
        if self.print_shape:
            print('mp3', x.shape)
        
        # x = self.mp3_bp(x)
        # if self.print_shape:
        #     print('mp3 bp', x.shape)
        
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
            print('gmp', x.shape)
            
        
        if self.rp is not None:
            x = self.rp(x)
            print('rp', x.shape)
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    


    
    
  

    
class EngModel3LAbsBP:
        
        
    def __init__(self, curv_params = {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=2000,filters_3=10000,batches_3 = 1,bp_filter_size=4,global_mp=False,num_projections=None):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.bp_filter_size = bp_filter_size
        self.num_projections = num_projections
        self.global_mp = global_mp
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        c1_bp = BlurPool(36, filt_size=self.bp_filter_size, stride=2)
        mp1 = nn.MaxPool2d(kernel_size=3)

        
        c2 = nn.Conv2d(36, self.filters_2, kernel_size=(9, 9))
        c2_bp = BlurPool(self.filters_2, filt_size=self.bp_filter_size, stride=2)
        mp2 = nn.MaxPool2d(kernel_size=2,stride=1)
        
        
        c3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7))
        c3_bp = BlurPool(self.filters_3*self.batches_3, filt_size=self.bp_filter_size, stride=2)
        mp3 = nn.MaxPool2d(kernel_size=2,stride=1)

        nl1 = NonLinearity('abs')
        rp = None
        if self.num_projections is not None:
            rp = RandomProjection(out_channels=self.num_projections)
        last = Output()
        

        
        return Model(
                c1 = c1,
                c1_bp = c1_bp,
                mp1 = mp1,
                c2 = c2,
                c2_bp = c2_bp,
                mp2 = mp2,
                c3 = c3,
                c3_bp = c3_bp,
                mp3 = mp3,
                batches_3 = self.batches_3,
                nl1 = nl1,
                global_mp = self.global_mp,
                rp = rp,
                last = last
        )