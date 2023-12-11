import torch
import datetime

from con_coffes_mapping_model.dataloader import get_dataloader
from con_coffes_mapping_model.OAMM_model import OPAC_Model
from con_coffes_mapping_model.OAMM_train_options import TrainOptions

torch.set_grad_enabled(True)


def training():
    config = TrainOptions().parse()

    print('Loading data...')
    dataloader = get_dataloader(path_to_excel=config.excel_file_path,
                                data_path=config.dataset_path,
                                phase='train',  # ['test', 'train']
                                batch_size=config.batch_size,
                                # shuffle=False
                                )

    len_dl = len(dataloader)
    print('\t-->> Data loading completed.')

    # 配置模型
    print('Load Model Configuration...')
    theModel = OPAC_Model(opt=config)  # in_ch=1, out_ch=1
    theModel.train()
    theModel.print_networks()
    print('\t-->> Model configuration completed.')
    print('Started training...')
    for epoch in range(config.epochs):
        print('\t-->> epoch:[%d, %5d]' % (epoch + 1, config.epochs),
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end='')
        for i, data in enumerate(dataloader):
            theModel.set_input(data)
            theModel.optimizer_parameters()

            if (i + 1) % config.viz_steps == 0:
                ret_loss = theModel.get_current_losses()

                print('\t\t-->> batch_num:[%d, %7d] cp_loss: %.5f, mf_loss: %.5f, lr: %.7f' % ( i+1, len_dl, ret_loss['cp_loss'],
                                                                                                ret_loss['mf_loss'], theModel.get_current_lr()),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if (epoch + 1) % config.train_spe == 0:
            # print('saving ae_model ..')
            theModel.save_networks(epoch + 1, ret_loss['mf_loss'])
        theModel.update_lr()


training()
