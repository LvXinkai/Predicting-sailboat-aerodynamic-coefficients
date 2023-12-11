import torch
import datetime
import os
import warnings

from net_model.net.net_model import SurrogateModel

from dataset_operate.dataset_dataloader import get_dataloader
from net_model.options.train_options import TrainOptions

torch.set_grad_enabled(True)


def training():
    config = TrainOptions().parse()

    print('Loading data...')
    dataloader = get_dataloader(path_to_excel=config.excel_file_path,
                                data_path=config.dataset_path,
                                phase='train',  # ['test', 'train']
                                batch_size=config.batch_size)
    len_dl = len(dataloader)
    print('\t-->> Data loading completed.')

    # 配置模型
    print('Load Model Configuration...')
    theModel = SurrogateModel(opt=config)
    theModel.train()
    theModel.print_networks()
    print('\t-->> Model configuration completed.')

    print('Started training...')
    for epoch in range(config.epochs):
        print('\n\t-->> epoch:[%d, %5d]' % (epoch + 1, config.epochs),
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for i, data in enumerate(dataloader):
            if (i + 1) % 10 == 0:
                print(str(i + 1) + ' ', end='')
            theModel.set_input(data)
            theModel.optimizer_parameters()

            if (i + 1) % config.viz_steps == 0:
                ret_loss = theModel.get_current_losses()

                print('\n\t\t-->> batch_num:[%d, %7d] mae_loss: %.5f, lr: %.7f' % ( i+1, len_dl, ret_loss['mae_loss'], theModel.get_current_lr()),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if (epoch + 1) % config.train_spe == 0:
            theModel.save_networks(epoch + 1, ret_loss['mae_loss'])
        theModel.update_lr()
