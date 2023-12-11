import argparse


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--dataset_path', type=str, default='../dataset/FlowField_64_V_and_P_IDW_mask',
                                 help='the file storing testing file paths')
        self.parser.add_argument('--excel_file_path', type=str, default='../dataset/FlowField_test_train.xlsx',
                                 help='the file storing testing file paths')
        self.parser.add_argument('--module_result_save_path', type=str, default='../checkpoints/module_result',
                                 help='the file storing testing file paths')
        self.parser.add_argument('--model_folder', type=str, default='../checkpoints/MSR_UNet',
                                 help='the file storing testing file paths')

        self.parser.add_argument('--phase', type=str, default='test')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt
