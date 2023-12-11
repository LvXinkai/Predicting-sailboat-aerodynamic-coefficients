import time
import torch
import os
import numpy as np
import pandas as pd
from net_model.options.test_options import TestOptions
from net_model.net.net_model import SurrogateModel
from dataset_operate.dataset_dataloader import get_dataloader,restore_maxmin_normalized_data
import matplotlib.pyplot as plt

torch.set_grad_enabled(True)


def calculate_deviation(source_data, net_data, mask, data_min, data_max):
    data_shape = mask.shape
    count = np.count_nonzero(mask)
    deviation = np.zeros(count)

    count = 0
    zero_num_index = []
    for g in range(data_shape[0]):
        for h in range(data_shape[1]):
            for k in range(data_shape[2]):
                if mask[g][h][k] == 1:
                    if source_data[g][h][k] == 0:
                        if data_min == 0:
                            zero_num_index.append(count)
                        else:
                            net_data_0 = (net_data[g][h][k] * (data_max - data_min)) + data_min
                            source_data_0 = (source_data[g][h][k] * (data_max - data_min)) + data_min
                            deviation[count] = np.abs((net_data_0 - source_data_0) / source_data_0)
                            # print('-***- The normalized value is 0, the original value is not 0, and the error rate is obtained after restoration.：' + str(deviation[count]))
                    else:
                        deviation[count] = np.abs((net_data[g][h][k] - source_data[g][h][k]) / source_data[g][h][k])
                    count += 1
                    # average += deviation[g][h][k]

    delete_num = len(zero_num_index)
    if delete_num > 0:
        # print('-***- The normalized value and the original value are both 0, and the number of points is deleted.：' + str(delete_num))
        for i in range(delete_num):
            np.delete(deviation, zero_num_index[i])

    d_mean = round(np.mean(deviation), 6)
    d_median = round(np.median(deviation), 6)
    d_var = round(np.var(deviation), 6)
    d_min = round(np.min(deviation), 6)
    d_max = round(np.max(deviation), 6)

    return {
        # 'info': 'The absolute value of the deviation rate, the minimum value is 0',
        'mean': d_mean,
        'median': d_median,
        'var': d_var,
        'min': d_min,
        'max': d_max
    }


def testing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TestOptions().parse()

    if not os.path.isdir(config.dataset_path):
        print('Invalid testing data file/folder path.')
        exit(1)

    print('configuring model..')
    theModel = SurrogateModel(opt=config)
    theModel.eval()
    theModel.load_networks(os.path.join(config.model_folder,
                                        'Well_trained_MSR_UNet/200_MSR_UNet_loss_0.0005.pth'))

    dataloader = get_dataloader(path_to_excel=config.excel_file_path,
                                data_path=config.dataset_path,
                                phase='test',  # ['test', 'train']
                                batch_size=config.batch_size,
                                shuffle=False)
    len_dl = len(dataloader)

    # x = np.linspace(0, 63, 64)
    # y = np.linspace(0, 63, 64)
    # X, Y = np.meshgrid(x, y)

    errors = np.zeros((len_dl,), dtype=np.float32)
    for t, data in enumerate(dataloader):
        print('Test Case:[%d, %5d], Source File: %s' % (t + 1, len_dl, data['path']))
        # start_time = time.time()
        net_data, features = theModel.evaluate(data['data'].to(device))
        # end_time = time.time()
        # run_time = end_time - start_time
        # print('stop time：', run_time)

        net_data = np.squeeze(net_data)
        features = np.squeeze(features)
        source_data = np.squeeze(data['data'].numpy())
        mask = np.squeeze(data['mask'].numpy())

        info = calculate_deviation(source_data,
                                   net_data,
                                   mask,
                                   data['data_min'].numpy()[0],
                                   data['data_max'].numpy()[0])
        print(info)
        errors[t] = info['mean']
        print('------------------ segmentation ------------------')

        # net_data = restore_maxmin_normalized_data(net_data,
        #                                           data['data_min'].numpy(),
        #                                           data['data_max'].numpy())
        # image_path = os.path.join(config.module_result_save_path, data['path'][0][:-4])
        # if not os.path.exists(image_path):
        #     os.makedirs(image_path)
        # np.save(os.path.join(config.module_result_save_path, data['path'][0]), net_data)

        # fig, ax = plt.subplots()
        # for i in [63, 48, 36, 24, 12, 0]:
        #     # for i in [255, 204, 153, 102, 51, 0]:
        #     img_data = np.copy(net_data[i, :, :])
        #     # coordinates = np.argwhere(source_data[-1][i] == 0)
        #     # for img_i in coordinates:
        #     #     img_data[img_i[0]][img_i[1]] = np.NaN
        #     cs = ax.contourf(X, Y, img_data)
        #     cbar = fig.colorbar(cs)
        #     # plt.pcolormesh(X, Y, static_pressure[i, :, :], cmap='RdBu_r', zorder=1)
        #     plt.savefig(os.path.join(image_path, str(i) + '.png'), dpi=800)
        #     cbar.remove()
        #     plt.cla()
        # plt.close(fig)


    n_up4 = np.percentile(errors, (25)) * 100
    n_mid4 = np.percentile(errors, (50)) * 100
    n_down4 = np.percentile(errors, (75)) * 100
    n_min = errors.min() * 100
    n_max = errors.max() * 100
    n_mean = errors.mean() * 100

    np.around(errors, decimals=2)
    print('error:')
    print('\tup4(75%): ' + str(np.round(n_up4, decimals=4)) + '%\t' + str(np.round(n_up4, decimals=2)) + '%')
    print('\tmid4(50%): ' + str(np.round(n_mid4, decimals=4)) + '%\t' + str(np.round(n_mid4, decimals=2)) + '%')
    print('\tdown4(25%): ' + str(np.round(n_down4, decimals=4)) + '%\t' + str(np.round(n_down4, decimals=2)) + '%')
    print('\tmin: ' + str(np.round(n_min, decimals=4)) + '%\t' + str(np.round(n_min, decimals=2)) + '%')
    print('\tmax: ' + str(np.round(n_max, decimals=4)) + '%\t' + str(np.round(n_max, decimals=2)) + '%')
    print('\tmean: ' + str(np.round(n_mean, decimals=4)) + '%\t' + str(np.round(n_mean, decimals=2)) + '%')
