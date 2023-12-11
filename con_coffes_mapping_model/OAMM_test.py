import time
import torch
import os
import numpy as np

from con_coffes_mapping_model.OAMM_test_options import TestOptions
from con_coffes_mapping_model.dataloader import get_dataloader
from con_coffes_mapping_model.OAMM_model import OPAC_Model

torch.set_grad_enabled(True)


def testing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestOptions().parse()

    if not os.path.isdir(config.dataset_path):
        print('Invalid testing data file/folder path.')
        exit(1)

    print('configuring model..')

    theModel = OPAC_Model(opt=config)
    theModel.eval()

    md_path = 'Well_trained_OAMM'
    theModel.load_networks(os.path.join(config.model_folder, md_path,
                                        '1600_net_mf_net_loss_0.0001.pth'), 'mf_net')
    theModel.load_networks(os.path.join(config.model_folder, md_path,
                                        '1600_net_cp_net_loss_0.0001.pth'), 'cp_net')

    dataloader = get_dataloader(path_to_excel=config.excel_file_path,
                                data_path=config.dataset_path,
                                phase='test',  # ['test', 'train', 'draw_pictures']
                                batch_size=config.batch_size,
                                shuffle=False)
    len_dl = len(dataloader)
    sum_mean = np.zeros((2,), dtype=np.float32)
    absolute_difference = np.zeros((2,), dtype=np.float32)
    start_time = time.time()
    gt_net = np.zeros((len_dl, 2, 2), dtype=np.float32)
    as_angle = np.zeros((len_dl,), dtype=np.float32)
    for t, data in enumerate(dataloader):
        print('Test Case:[%d, %5d], Source File: %s' % (t + 1, len_dl, data['path']))
        net_data = theModel.evaluate(data['bjm'].to(device))
        net_data = np.squeeze(net_data)
        true_result = np.squeeze(data['x_and_y_force_c'].numpy())
        abs_data = np.abs((net_data-true_result)/true_result)
        absolute_difference += np.abs(net_data-true_result)

        gt_net[t][0] = true_result
        gt_net[t][1] = net_data
        as_angle[t] = data['bjm_original'][0][2]

        sum_mean += abs_data
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time/len_dl)
    error = np.round(sum_mean.mean() / len_dl * 100, decimals=4)
    print('error: ' + str(error) + '%')
    print('error[0]: ' + str(sum_mean[0] / len_dl * 100) + '%   — —   error[1]: ' + str(sum_mean[1] / len_dl * 100) + '%')
    absolute_difference /= len_dl
    print('absolute_difference: ' + str(np.round(absolute_difference.mean(), decimals=4)))
    print('absolute_difference[0]: ' + str(absolute_difference[0]) + '   — —   error[1]: ' + str(absolute_difference[1]))

    return gt_net, as_angle, absolute_difference


testing()
