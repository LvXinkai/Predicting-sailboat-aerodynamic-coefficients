import argparse
import os
import time


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # ------------------ train_options.py ------------------
        self.parser.add_argument('--batch_size', type=int, default=80, help='input batch size')
        self.parser.add_argument('--dataset_path', type=str, default='../dataset/Flow_Field_Characteristics',
                                 help='the file storing testing file paths')
        self.parser.add_argument('--excel_file_path', type=str, default='../dataset/CoefficientMapping_test_train.xlsx',
                                 help='the file storing testing file paths')

        self.parser.add_argument('--epochs', type=int, default=1600)

        self.parser.add_argument('--viz_steps', type=int, default=5)
        self.parser.add_argument('--train_spe', type=int, default=50)
        # ------------------------------------------------------------
        # ------------------------- for net -------------------------
        self.parser.add_argument('--model_name', type=str, default='OAMM_Model')

        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--epoch_update_lr', type=int, default=15)

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2')

        self.parser.add_argument('--checkpoint_dir', type=str, default='../dataset/OAMM', help='models are saved here')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--model_folder', type=str, default='model_folder', help='Save the folder for the model')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            str_id = int(str_id)
            if str_id >= 0:
                self.opt.gpu_ids.append(str(str_id))

        self.opt.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.model_name
        self.opt.model_folder += '_b' + str(self.opt.batch_size)
        self.opt.model_folder += '_lr' + str(self.opt.lr)

        if os.path.isdir(self.opt.checkpoint_dir) is False:
            os.mkdir(self.opt.checkpoint_dir)

        self.opt.model_folder = os.path.join(self.opt.checkpoint_dir, self.opt.model_folder)
        if os.path.isdir(self.opt.model_folder) is False:
            os.mkdir(self.opt.model_folder)

        print('------------ Options -------------')
        print('epochs: %s' % (self.opt.epochs))
        print('batch_size: %s' % (self.opt.batch_size))
        print('lr: %s' % (self.opt.lr))
        # print('mask_type: %s' % (self.opt.mask_type))
        print('-------------- End ----------------')

        return self.opt
