import os
import pprint
import argparse
import time
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from lib.utils import utils
from lib.core import function
from lib.datasets import MULTI_DataLoader
from lib.datasets import get_dataset
from lib.config import config, update_config
import lib.models as models

same_index = {
    '98': [33, 46, 60, 64, 68, 72, 54, 76, 82, 16],
    '68': [17, 26, 36, 39, 42, 45, 30, 48, 54, 8],
    '29': [0, 1, 8, 10, 11, 9, 20, 22, 23, 28],
    '19': [0, 5, 6, 8, 9, 11, 13, 15, 17, 18]
}


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--main_data', type=str, default="wflw")
    parser.add_argument('--aux_datas', type=str, default='300w')
    parser.add_argument('--resume_checkpoints', type=str, default="")
    parser.add_argument('--gpus', type=str, default='7')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--show_others', type=int, default=1)

    parser.add_argument('--aux_ratios', type=str, default='5,1')
    parser.add_argument('--heatmap_sigma', type=float, default=1.5)
    parser.add_argument('--aug_sigma', type=float, default=1.5)
    parser.add_argument('--model_dir', type=str, default="5_1_1")
    parser.add_argument('--backbone_lr', type=float, default=1.5e-4)
    parser.add_argument('--main_lr', type=float, default=2e-4)
    parser.add_argument('--aux_lr', type=float, default=2e-4)
    
    parser.add_argument('--ratios_decay', type=float, default=1.0)
    
    

    parser.add_argument('--mix_loss', default=False,
                        action='store_true', help="use mix loss in trainging")
    parser.add_argument('--data_aug', default=False,
                        action='store_true', help="control aux datas augmentation")

    parser.add_argument('--loss_alpha', type=float, default=0.003)
    parser.add_argument('--loss_decay', type=float, default=1.0)
    args = parser.parse_args()
    main_cfg = os.path.join("experiments", args.main_data,
                            "face_alignment_{}_hrnet_w18.yaml".format(args.main_data))
    update_config(config, main_cfg)

    config["MODEL"]["SIGMA"] = args.heatmap_sigma
    return args


def main():

    # init
    args = parse_args()
    args.aux_datas = args.aux_datas.split(',')
    args.aux_ratios = np.asarray(args.aux_ratios.split(','), dtype=np.float32)

    # create logger and save folder
    logger = utils.create_logger_direct(
        "TESTAux_{}_{}_{}".format(args.main_data,args.aux_datas,args.model_dir))
    model_save_dir = os.path.join(
        "mix", "checkpoints", args.main_data, args.model_dir)
    utils.check_mkdir(model_save_dir)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    backbone, heads, aux_configs = models.get_face_alignment_nets(
        config, args.aux_datas, aug_sigma=args.aug_sigma)
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    backbone = nn.DataParallel(backbone, range(gpu_nums)).cuda()
    devices = torch.device('cuda:0')
    backbone.to(devices)

    optimizer_backbone = optim.Adam(
        filter(lambda p: p.requires_grad, backbone.parameters()),
        lr=args.backbone_lr,
    )

    lr_scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_backbone, config.TRAIN.LR_STEP,
        config.TRAIN.LR_FACTOR, -1
    )

    optimizer_heads = {}
    schedulers_heads = {}

    for key in heads.keys():
        heads[key] = nn.DataParallel(heads[key], range(gpu_nums)).cuda()
        devices = torch.device('cuda:0')
        heads[key].to(devices)
        main_flag = True if key == str(config.MODEL.NUM_JOINTS) else False
        if main_flag:
            optimizer_heads[key] = optim.Adam(
                filter(lambda p: p.requires_grad, heads[key].parameters()),
                lr=args.main_lr
            )

            schedulers_heads[key] = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_heads[key], config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR, -1
            )
        else:
            optimizer_heads[key] = optim.Adam(
                filter(lambda p: p.requires_grad, heads[key].parameters()),
                lr=args.aux_lr
            )

            schedulers_heads[key] = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_heads[key], aux_configs[key].TRAIN.LR_STEP,
                aux_configs[key].TRAIN.LR_FACTOR, -1
            )

    best_nme = 10  # Init a big loss
    if os.path.isfile(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):
        checkpoint = torch.load(args.resume_checkpoints)
        backbone.load_state_dict(checkpoint['backbone'])
        for key in heads.keys():
            heads[key].load_state_dict(checkpoint['heads'][key])
        best_nme = checkpoint['best_nme']
        logger.info("restore epoch : {} best nme : {} ".format(
            checkpoint['epoch'], checkpoint['best_nme']))

    epoch = checkpoint['epoch']
    main_dataset = get_dataset(config)
    # main_train_dataset = main_dataset(config,is_train=True, is_aug=True)
    main_val_dataset = main_dataset(config, is_train=False)
    # main_train_loader = DataLoader(
    #     dataset=main_train_dataset,
    #     batch_size=args.batch_size * gpu_nums,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )

    main_val_loader = DataLoader(
        dataset=main_val_dataset,
        batch_size=args.batch_size * gpu_nums,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    aux_dataloader = {'train': [], "test": {}}
    for key in aux_configs.keys():
        temp_dataset = get_dataset(aux_configs[key])
        temp_train_dataset = temp_dataset(
            aux_configs[key], is_train=True, is_aug=args.data_aug)
        # aux_dataloader['train'].append(DataLoader(
        #     dataset = temp_train_dataset,
        #     batch_size= args.batch_size * gpu_nums,
        #     shuffle= True,
        #     num_workers = args.workers,
        #     pin_memory= True
        # ))

        if args.show_others:
            temp_test_dataset = temp_dataset(aux_configs[key], is_train=False)
            aux_dataloader['test'][key] = DataLoader(
                dataset=temp_test_dataset,
                batch_size=args.batch_size * gpu_nums,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )

    # mix_train_dataloader = MULTI_DataLoader(main_train_loader,aux_dataloader['train'],args.aux_ratios)
    ratio_speed_array = [1] + [args.ratios_decay] * \
        (args.aux_ratios.size - 1)  # each epoch ratio will reduce
    last_loss_alpha = [1, 1, 1]  # main_right, aux_left, aux_right

    if args.show_others:
        # validate aux dataset
        for key in aux_configs.keys():
            function.mix_val(
                aux_configs[key], aux_dataloader['test'][key], backbone, heads[key], criterion, epoch)

    # validate main dataset
    val_nme, predictions = function.mix_val(
        config, main_val_loader, backbone, heads[str(config.MODEL.NUM_JOINTS)], criterion, epoch)


if __name__ == '__main__':
    main()
