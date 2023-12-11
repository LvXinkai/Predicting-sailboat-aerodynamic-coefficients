import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from net_model.net.net import MultiScale_ResUnet
from net_model.net.base_model import BaseModel, init_weights


class SurrogateModel(BaseModel):
    def __init__(self, opt=None):  # in_ch=1, out_ch=1, Fixed number of input and output channels
        super(SurrogateModel, self).__init__()
        self.opt = opt
        self.init(opt)
        self.model_names = ['net']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Training device: ' + str(self.device))
        self.net = MultiScale_ResUnet(1, 1).to(self.device)
        # self.net = ().to(self.device)
        init_weights(self.net)

        if self.opt.phase == 'test':
            return

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.5, 0.9))  # , betas=(0.5, 0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.epoch_update_lr, gamma=0.9)
        self.optimizer.zero_grad()

        self.mae_loss = nn.L1Loss()

        self.loss = 0  # 编码器总损失
        self.mid_loss = None
        self.characteristic = None
        self.net_out = None
        self.mask = None

    def get_current_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def update_lr(self):
        self.scheduler.step()

    def set_input(self, input_data):
        self.input = input_data['data'].to(self.device)

    def forward_AE(self):
        self.loss = self.mae_loss(self.input, self.net_out)

    def backward_AE(self):
        self.loss.backward()

    def get_characteristic(self):
        return self.characteristic.cpu().detach().numpy()

    def optimizer_parameters(self):
        self.net_out, self.characteristic = self.net(self.input)

        self.optimizer.zero_grad()
        self.forward_AE()
        self.backward_AE()
        self.optimizer.step()

    def get_current_losses(self):
        sur_loss = {'mae_loss': round(self.loss.item(), 4)}
        return sur_loss

    def get_current_visuals(self):
        return {'input/encoder_in': self.encoder_in.detach().cpu().numpy(),
                'characteristic': self.characteristic.detach().cpu().numpy(),
                'decoder_out': self.decoder_out.detach().cpu().numpy()
                }

    def get_current_visuals_tensor(self):
        return {'input/encoder_in': self.encoder_in.detach().cpu(),
                'characteristic': self.characteristic.detach().cpu(),
                'decoder_out': self.decoder_out.detach().cpu()
                }

    def evaluate(self, data_in):
        out, characteristic = self.net(data_in)
        if characteristic is not None:
            characteristic = characteristic.detach().cpu().numpy()
        return out.detach().cpu().numpy(), characteristic


