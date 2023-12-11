import torch
import os
import warnings
from net_model.train import training

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')
training()
