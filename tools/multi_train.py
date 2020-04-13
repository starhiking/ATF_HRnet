# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

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
# import torch.distributed as dist
# dist.init_process_group(backend='nccl')

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--main_data', type=str, default="wflw")
    parser.add_argument('--aux_datas',type=str,default='300w,aflw')
    parser.add_argument('--aux_ratios',type=str,default='5,1,1')

    parser.add_argument('--resume_checkpoints',type=str,default="")
    parser.add_argument('--model_dir',type=str,default="5_1_1")

    parser.add_argument('--gpus',type=str,default='7')
    parser.add_argument('--backbone_lr',type=float,default=1.5e-4)
    parser.add_argument('--main_lr',type=float,default=2e-4)
    parser.add_argument('--aux_lr',type=float,default=1e-4)
    parser.add_argument('--batch_size',type=int,default=48)
    parser.add_argument('--ratios_decay',type=float,default=1.0)
    parser.add_argument('--workers',type=int,default=4)

    args = parser.parse_args()
    main_cfg = os.path.join("experiments",args.main_data,"face_alignment_{}_hrnet_w18.yaml".format(args.main_data))
    update_config(config,main_cfg)
    return args


def main():

    # init
    args = parse_args()
    args.aux_datas = args.aux_datas.split(',')
    args.aux_ratios = np.asarray(args.aux_ratios.split(','),dtype=np.float32)
    
    # create logger and save folder
    logger = utils.create_logger_direct("Mix_{}_{}".format(config.DATASET.DATASET,args.model_dir))
    model_save_dir = os.path.join("mix","checkpoints",args.main_data,args.model_dir)
    utils.check_mkdir(model_save_dir)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    backbone , heads , aux_configs = models.get_face_alignment_nets(config,args.aux_datas)
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    backbone = nn.DataParallel(backbone,range(gpu_nums)).cuda()
    devices = torch.device('cuda:0')
    backbone.to(devices)

    optimizer_backbone = optim.Adam(
        filter(lambda p: p.requires_grad, backbone.parameters()),
        lr = args.backbone_lr,
    )

    lr_scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_backbone, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, -1
    )

    optimizer_heads = {}
    schedulers_heads = {}

    for key in heads.keys():
        heads[key] = nn.DataParallel(heads[key],range(gpu_nums)).cuda()
        devices = torch.device('cuda:0')
        heads[key].to(devices)
        main_flag = True if key == str(config.MODEL.NUM_JOINTS) else False
        if main_flag:
            optimizer_heads[key] = optim.Adam(
                filter(lambda p: p.requires_grad,heads[key].parameters()),
                lr = args.main_lr
            )

            schedulers_heads[key] = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_heads[key],config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR,-1
            )
        else:
            optimizer_heads[key] = optim.Adam(
                filter(lambda p: p.requires_grad,heads[key].parameters()),
                lr = args.aux_lr
            )

            schedulers_heads[key] = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_heads[key],aux_configs[key].TRAIN.LR_STEP,
                aux_configs[key].TRAIN.LR_FACTOR,-1
            )

    best_nme = 10 # Init a big loss
    if os.path.isfile(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):        
        checkpoint = torch.load(args.resume_checkpoints)    
        backbone.load_state_dict(checkpoint['backbone'])
        for key in heads.keys():
            heads[key].load_state_dict(checkpoint['heads'][key])
        best_nme = checkpoint['best_nme']
        logger.info("restore epoch : {} best nme : {} ".format(checkpoint['epoch'],checkpoint['best_nme']))


    main_dataset = get_dataset(config)
    main_train_dataset = main_dataset(config,is_train=True)
    main_val_dataset = main_dataset(config,is_train=False)
    main_train_loader = DataLoader(
        dataset=main_train_dataset,
        batch_size=args.batch_size * gpu_nums,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    main_val_loader = DataLoader(
        dataset=main_val_dataset,
        batch_size=args.batch_size * gpu_nums,
        shuffle=True,
        num_workers=args.workers ,
        pin_memory=True
    )

    aux_dataloader = {'train':[],"test":{}}
    for key in aux_configs.keys():
        temp_dataset = get_dataset(aux_configs[key])
        temp_train_dataset = temp_dataset(aux_configs[key],is_train=True)
        temp_test_dataset = temp_dataset(aux_configs[key],is_train=False)
        aux_dataloader['train'].append(DataLoader(
            dataset = temp_train_dataset,
            batch_size= args.batch_size * gpu_nums,
            shuffle= True,
            num_workers = args.workers,
            pin_memory= False
        ))

        aux_dataloader['test'][key] = DataLoader(
            dataset=temp_test_dataset,
            batch_size=args.batch_size * gpu_nums ,
            shuffle=True,
            num_workers=args.workers ,
            pin_memory=False
        )

    mix_train_dataloader = MULTI_DataLoader(main_train_loader,aux_dataloader['train'],args.aux_ratios)
    ratio_speed_array = [1] + [args.ratios_decay] * (args.aux_ratios.size -1) # each epoch ratio will reduce

    for epoch in range(0,config.TRAIN.END_EPOCH):
        logger.info("Use {} train epoch {}".format(utils.lr_repr(optimizer_backbone),epoch))
        
        # train
        mix_train_dataloader.init_iter()
        train_loss = function.AverageMeter()
        train_loss.reset()

        backbone.train()
        for key in heads.keys():
            heads[key].train()
        
        nme_count = 0
        nme_batch_sum = 0

        start_time = time.time()

        while mix_train_dataloader.get_iter_flag():

            inp,target,meta = mix_train_dataloader.get_iter()
            current_landmark_num = str(target.size(1))
            feature_map = backbone(inp).cuda()
            output = heads[current_landmark_num](feature_map).cuda()

            target = target.cuda(non_blocking = True)
            loss = criterion(output,target)

            # optimize 
            optimizer_backbone.zero_grad()
            optimizer_heads[current_landmark_num].zero_grad()

            loss.backward()

            optimizer_backbone.step()
            optimizer_heads[current_landmark_num].step()
            train_loss.update(loss.item(),inp.size(0))

        logger.info("{}'epoch train time :{:<6.2f}s loss :{:.8f} ".format(epoch,time.time()-start_time,train_loss.avg))

        # adjust ratios
        args.aux_ratios = args.aux_ratios * ratio_speed_array
        mix_train_dataloader.change_ratios(args.aux_ratios)

        # validate main dataset
        val_nme, predictions = function.mix_val(config,main_val_loader,backbone,heads[str(config.MODEL.NUM_JOINTS)],criterion,epoch)
        
        # validate aux dataset
        for key in aux_configs.keys():
            function.mix_val(aux_configs[key],aux_dataloader['test'][key],backbone,heads[key],criterion,epoch)
        
        # save better checkpoint
        if val_nme < best_nme:
            best_nme = val_nme
            file_path = os.path.join(model_save_dir,"{}_checkpoint.pth".format(epoch))
            best_path = os.path.join(model_save_dir,"best.pth")
            logger.info('saving checkpoint  to {}'.format(file_path))
            torch.save({
                "best_nme":best_nme,
                "backbone":backbone.state_dict(),
                "{}".format(config.MODEL.NUM_JOINTS):heads[str(config.MODEL.NUM_JOINTS)].state_dict(),
                "heads":{key:val.state_dict() for (key,val) in heads.items()},
                "epoch":epoch
            },file_path)
            if os.path.islink(best_path):
                os.remove(best_path)
            # symlink is create a relative path file : a is exist file and relative path,b is link and absolute path
            os.symlink(os.path.join("./","{}_checkpoint.pth".format(epoch)),best_path)
        
        # lr step
        lr_scheduler_backbone.step()
        for key in schedulers_heads.keys():
            schedulers_heads[key].step()


if __name__ == '__main__':
    main()