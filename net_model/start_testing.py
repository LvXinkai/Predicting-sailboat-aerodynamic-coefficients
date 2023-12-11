import os
import warnings
import torch
from net_model.test import testing

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')
testing()
