import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.datasets import MULTI_DataLoader
from lib.core import function
from lib.utils import utils
import numpy as np
import time

test_files = {
    "WFLW":None,
    "300W":None,
    "AFLW":None,
    "COFW":None
}

test_files["WFLW"] = [ 
    # "data/wflw/face_landmarks_wflw_test_blur.csv",
    # "data/wflw/face_landmarks_wflw_test_expression.csv",
    # "data/wflw/face_landmarks_wflw_test_illumination.csv",
    # "data/wflw/face_landmarks_wflw_test_largepose.csv",
    # "data/wflw/face_landmarks_wflw_test_makeup.csv",
    # "data/wflw/face_landmarks_wflw_test_occlusion.csv",
    "data/wflw/face_landmarks_wflw_test.csv"
    ]

test_files["300W"] = [
    "data/300w/face_landmarks_300w_valid_challenge.csv",
    "data/300w/face_landmarks_300w_valid_common.csv",
    "data/300w/face_landmarks_300w_valid.csv",
    "data/300w/face_landmarks_300w_test.csv"
    ]

test_files["AFLW"] = [
    "data/aflw/face_landmarks_aflw_test_frontal.csv",
    "data/aflw/face_landmarks_aflw_test.csv",
]

test_files["COFW"] = [
    "data/cofw/COFW_test_color.mat"
]



def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--resume_checkpoints',type=str,default="mix/checkpoints/wflw/mix_loss_wa101/37_checkpoint.pth")
    parser.add_argument('--main_data', type=str, default="wflw")
    parser.add_argument('--gpus',type=str,default='7')
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)

    parser.add_argument('--aux_datas',type=str,default='300w')
    parser.add_argument('--aux_ratios',type=str,default='5,1')
    parser.add_argument('--heatmap_sigma',type=float,default=1.5)
    parser.add_argument('--aug_sigma',type=float,default=1.5)

    parser.add_argument('--model_dir',type=str,default="5_1_1")
    
    parser.add_argument('--backbone_lr',type=float,default=1.5e-4)
    parser.add_argument('--main_lr',type=float,default=2e-4)
    parser.add_argument('--aux_lr',type=float,default=2e-4)
    parser.add_argument('--ratios_decay',type=float,default=1.0)
    parser.add_argument('--show_others',type=int,default=0)
    
    parser.add_argument('--mix_loss',default=False,action='store_true',help="use mix loss in trainging")
    parser.add_argument('--data_aug',default=False,action='store_true',help="control aux datas augmentation")

    parser.add_argument('--loss_alpha',type=float,default=1.0)
    parser.add_argument('--loss_decay',type=float,default=1.0)
    args = parser.parse_args()
    main_cfg = os.path.join("experiments",args.main_data,"face_alignment_{}_hrnet_w18.yaml".format(args.main_data))
    update_config(config,main_cfg)
    
    config["MODEL"]["SIGMA"] = args.heatmap_sigma
    return args


def main():

    # init
    args = parse_args()
    args.aux_datas = args.aux_datas.split(',')
    args.aux_ratios = np.asarray(args.aux_ratios.split(','),dtype=np.float32)
    
    # create logger and save folder
    logger = utils.create_logger_direct("Test_{}_{}".format(config.DATASET.DATASET,args.model_dir))
    model_save_dir = os.path.join("mix","checkpoints",args.main_data,args.model_dir)
    utils.check_mkdir(model_save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    backbone , heads , aux_configs = models.get_face_alignment_nets(config,args.aux_datas,aug_sigma=args.aug_sigma)
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    backbone = nn.DataParallel(backbone,range(gpu_nums)).cuda()
    devices = torch.device('cuda:0')
    backbone.to(devices)

    for key in heads.keys():
        heads[key] = nn.DataParallel(heads[key],range(gpu_nums)).cuda()
        devices = torch.device('cuda:0')
        heads[key].to(devices)

    best_nme = 10 # Init a big loss
    if os.path.isfile(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):        
        checkpoint = torch.load(args.resume_checkpoints)    
        backbone.load_state_dict(checkpoint['backbone'])
        heads[str(config.MODEL.NUM_JOINTS)].load_state_dict(checkpoint['heads'][str(config.MODEL.NUM_JOINTS)])
        # for key in heads.keys():
        #     heads[key].load_state_dict(checkpoint['heads'][key])
        best_nme = checkpoint['best_nme']
        logger.info("restore epoch : {} best nme : {} ".format(checkpoint['epoch'],checkpoint['best_nme']))


    for file_path in test_files[config.DATASET.DATASET]:

        config["DATASET"]["TESTSET"] = file_path
        logger.info("Test {}".format(file_path))

        main_dataset = get_dataset(config)
        main_val_dataset = main_dataset(config,is_train=False)
        main_val_loader = DataLoader(
            dataset=main_val_dataset,
            batch_size=args.batch_size * gpu_nums,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )

        nme, predictions = function.mix_inference(config, main_val_loader, backbone,heads[str(config.MODEL.NUM_JOINTS)])
        




if __name__ == '__main__':
    main()