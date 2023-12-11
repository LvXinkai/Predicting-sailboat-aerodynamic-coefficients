import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from con_coffes_mapping_model.OAMM_net import AlphaToBeta, BetaToR
from net_model.net.base_model import BaseModel, init_weights


class OPAC_Model(BaseModel):
    def __init__(self, opt=None):  # in_ch=1, out_ch=1, Fixed number of input and output channels
        super(OPAC_Model, self).__init__()
        self.opt = opt
        self.init(opt)
        self.model_names = ['cp_net', 'mf_net']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Training device: ' + str(self.device))
        self.cp_net = AlphaToBeta().to(self.device)
        self.mf_net = BetaToR().to(self.device)

        init_weights(self.cp_net)
        init_weights(self.mf_net)

        if self.opt.phase == 'test':
            return

        self.cp_optimizer = torch.optim.Adam(self.cp_net.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.cp_scheduler = lr_scheduler.StepLR(self.cp_optimizer, step_size=opt.epoch_update_lr, gamma=0.9)
        self.cp_optimizer.zero_grad()

        self.mf_optimizer = torch.optim.Adam(self.mf_net.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.mf_scheduler = lr_scheduler.StepLR(self.mf_optimizer, step_size=opt.epoch_update_lr, gamma=0.9)
        self.mf_optimizer.zero_grad()

        self.cp_mae_loss = nn.MSELoss()
        self.mf_mae_loss = nn.MSELoss()

        self.cp_loss = 0
        self.cp_net_out = None
        self.cp_true_result = None

        self.mf_loss = 0
        self.mf_net_out = None
        self.mf_output = None

    def get_current_lr(self):
        return self.cp_optimizer.state_dict()['param_groups'][0]['lr']

    def update_lr(self):
        self.cp_scheduler.step()
        self.mf_scheduler.step()

    def set_input(self, input_data):
        self.input = input_data['bjm'].to(self.device)
        self.cp_true_result = input_data['data'].to(self.device)
        self.mf_output = input_data['x_and_y_force_c'].to(self.device)

    def cp_forward_AE(self):
        self.cp_loss = self.cp_mae_loss(self.cp_true_result, self.cp_net_out)

    def mf_forward_AE(self):
        self.mf_loss = self.mf_mae_loss(self.mf_output, self.mf_net_out)

    def cp_backward_AE(self):
        self.cp_loss.backward()

    def mf_backward_AE(self):
        self.mf_loss.backward()

    def optimizer_parameters(self):
        self.cp_net_out = self.cp_net(self.input)
        self.cp_optimizer.zero_grad()
        self.cp_forward_AE()
        self.cp_backward_AE()
        self.cp_optimizer.step()

        self.mf_net_out = self.mf_net(self.cp_net_out.data)
        self.mf_optimizer.zero_grad()
        self.mf_forward_AE()
        self.mf_backward_AE()
        self.mf_optimizer.step()

    def get_current_losses(self):
        sur_loss = {'cp_loss': round(self.cp_loss.item(), 4), 'mf_loss': round(self.mf_loss.item(), 4)}
        return sur_loss

    def get_current_visuals(self):
        return {'input/encoder_in': self.encoder_in.detach().cpu().numpy(),
                'decoder_out': self.decoder_out.detach().cpu().numpy()
                }

    def get_current_visuals_tensor(self):
        return {'input/encoder_in': self.encoder_in.detach().cpu(),
                'decoder_out': self.decoder_out.detach().cpu()
                }

    def evaluate(self, data_in):
        out = self.mf_net(self.cp_net(data_in))
        return out.detach().cpu().numpy()
