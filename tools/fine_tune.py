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

same_index = {
    '98':[33,46,60,64,68,72,54,76,82,16],
    '68':[17,26,36,39,42,45,30,48,54,8],
    '29':[0,1,8,10,11,9,20,22,23,28],
    '19':[0,5,6,8,9,11,13,15,17,18]
}


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--main_data', type=str, default="300w")
    # parser.add_argument('--aux_datas',type=str,default='300w,aflw')
    # parser.add_argument('--aux_ratios',type=str,default='5,1,1')
    parser.add_argument('--heatmap_sigma',type=float,default=1.5)
    # parser.add_argument('--aug_sigma',type=float,default=1.5)

    parser.add_argument('--resume_checkpoints',type=str,default="mix/checkpoints/wflw/mix_loss_w341_/44_checkpoint.pth")
    parser.add_argument('--model_dir',type=str,default="5_1_1")
    
    parser.add_argument('--gpus',type=str,default='7')
    # parser.add_argument('--backbone_lr',type=float,default=1.5e-4)
    parser.add_argument('--main_lr',type=float,default=2e-4)
    # parser.add_argument('--aux_lr',type=float,default=2e-4)
    parser.add_argument('--batch_size',type=int,default=48)
    # parser.add_argument('--ratios_decay',type=float,default=1.0)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--nme',type=float,default=0)
    # parser.add_argument('--show_others',type=int,default=0)
    
    # parser.add_argument('--mix_loss',default=False,action='store_true',help="use mix loss in trainging")
    # parser.add_argument('--data_aug',default=False,action='store_true',help="control aux datas augmentation")

    # parser.add_argument('--loss_alpha',type=float,default=0.003)
    # parser.add_argument('--loss_decay',type=float,default=1.0)
    args = parser.parse_args()
    main_cfg = os.path.join("experiments",args.main_data,"face_alignment_{}_hrnet_w18.yaml".format(args.main_data))
    update_config(config,main_cfg)
    
    config["MODEL"]["SIGMA"] = args.heatmap_sigma
    return args


def main():

    # init
    args = parse_args()
    # args.aux_datas = args.aux_datas.split(',')
    # args.aux_ratios = np.asarray(args.aux_ratios.split(','),dtype=np.float32)
    
    # create logger and save folder
    logger = utils.create_logger_direct("FineTune_{}_{}".format(config.DATASET.DATASET,args.model_dir))
    model_save_dir = os.path.join("mix","checkpoints",args.main_data,args.model_dir)
    utils.check_mkdir(model_save_dir)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_nums = len(args.gpus.split(','))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    backbone , heads , aux_configs = models.get_face_alignment_nets(config,[],aug_sigma=1.5)
    # aux_configs = {}
    logger.info("Head nums: {}".format(len(heads)))
    main_joints = config.MODEL.NUM_JOINTS
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    backbone = nn.DataParallel(backbone,range(gpu_nums)).cuda()
    devices = torch.device('cuda:0')
    backbone.to(devices)

    # optimizer_backbone = optim.Adam(
    #     filter(lambda p: p.requires_grad, backbone.parameters()),
    #     lr = args.backbone_lr,
    # )

    # lr_scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer_backbone, config.TRAIN.LR_STEP,
    #         config.TRAIN.LR_FACTOR, -1
    # )

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
        if str(main_joints) in checkpoint['heads'].keys():
            heads[str(main_joints)].load_state_dict(checkpoint['heads'][str(main_joints)])
            logger.info("Load head {} finished".format(main_joints))
        # for key in heads.keys():           
        #     heads[key].load_state_dict(checkpoint['heads'][key])
        best_nme = checkpoint['best_nme']
        logger.info("restore epoch : {} best nme : {} ".format(checkpoint['epoch'],checkpoint['best_nme']))

    if args.nme:
        best_nme = args.nme

    main_dataset = get_dataset(config)
    main_train_dataset = main_dataset(config,is_train=True, is_aug=True)
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
        num_workers=args.workers,
        pin_memory=True
    )

    # aux_dataloader = {'train':[],"test":{}}
    # for key in aux_configs.keys():
    #     temp_dataset = get_dataset(aux_configs[key])
    #     temp_train_dataset = temp_dataset(aux_configs[key],is_train=True, is_aug=args.data_aug)
    #     aux_dataloader['train'].append(DataLoader(
    #         dataset = temp_train_dataset,
    #         batch_size= args.batch_size * gpu_nums,
    #         shuffle= True,
    #         num_workers = args.workers,
    #         pin_memory= True
    #     ))

    #     if args.show_others:
    #         temp_test_dataset = temp_dataset(aux_configs[key],is_train=False)
    #         aux_dataloader['test'][key] = DataLoader(
    #             dataset=temp_test_dataset,
    #             batch_size=args.batch_size * gpu_nums ,
    #             shuffle=True,
    #             num_workers=args.workers ,
    #             pin_memory=True
    #         )

    # mix_train_dataloader = MULTI_DataLoader(main_train_loader,aux_dataloader['train'],args.aux_ratios)
    # ratio_speed_array = [1] + [args.ratios_decay] * (args.aux_ratios.size -1) # each epoch ratio will reduce
    # last_loss_alpha = [1,1,1] # main_right, aux_left, aux_right

    for epoch in range(0,config.TRAIN.END_EPOCH):
        logger.info("Use {} train epoch {}".format(utils.lr_repr(optimizer_heads[str(main_joints)]),epoch))
        
        # train
        # mix_train_dataloader.init_iter()
        train_loss = function.AverageMeter()
        train_loss.reset()

        # main_left_loss = function.AverageMeter()
        # main_left_loss.reset()
        # main_right_loss = function.AverageMeter()
        # main_right_loss.reset()

        # aux_left_loss = function.AverageMeter()
        # aux_left_loss.reset()
        # aux_right_loss = function.AverageMeter()
        # aux_right_loss.reset()

        backbone.eval()
        for key in heads.keys():
            heads[key].train()
        
        nme_count = 0
        nme_batch_sum = 0

        start_time = time.time()

        for i, (inp, target, meta) in enumerate(main_train_loader):
            feature_map = backbone(inp).cuda()
            output = heads[str(main_joints)](feature_map).cuda()
            target = target.cuda(non_blocking = True)
            loss = criterion(output,target)

        # while mix_train_dataloader.get_iter_flag():

        #     inp,target,meta = mix_train_dataloader.get_iter()
        #     current_landmark_num = str(target.size(1))
        #     feature_map = backbone(inp).cuda()
        #     output = heads[current_landmark_num](feature_map).cuda()

        #     target = target.cuda(non_blocking = True)
        #     loss = criterion(output,target)

        #     if args.mix_loss:
        #         if output.size(1) == config.MODEL.NUM_JOINTS:
        #             # when iteration is main
        #             main_left_loss.update(loss.item(),inp.size(0))
        #             main_indexs = same_index[current_landmark_num]
        #             for key in aux_configs.keys():
        #                 temp_output = heads[key](feature_map).cuda()
        #                 temp_indexs = same_index[str(temp_output.size(1))]
        #                 right_loss = criterion(output[:,main_indexs],temp_output[:,temp_indexs])
        #                 main_right_loss.update(right_loss.item(),inp.size(0))  
        #                 print('main: left loss {} right loss {}'.format(loss.item(),last_loss_alpha[0]*right_loss.item()))
        #                 loss = loss + last_loss_alpha[0] * right_loss
        #         else :
        #             # when iteration is auxiliary
        #             aux_left_loss.update(loss.item(),inp.size(0))
        #             main_output = heads[str(config.MODEL.NUM_JOINTS)](feature_map).cuda()
        #             main_indexs = same_index[str(config.MODEL.NUM_JOINTS)]
        #             aux_indexs  = same_index[current_landmark_num]
        #             right_loss = criterion(output[:,aux_indexs],main_output[:,main_indexs])
        #             aux_right_loss.update(right_loss.item(),inp.size(0))
        #             print('aux: left loss {} right loss {}'.format(last_loss_alpha[1]*loss.item(),last_loss_alpha[2]*right_loss.item()))
        #             loss = last_loss_alpha[1] * loss + last_loss_alpha[2] * right_loss

            # optimize 
            # optimizer_backbone.zero_grad()
            optimizer_heads[str(main_joints)].zero_grad()

            loss.backward()

            # optimizer_backbone.step()
            optimizer_heads[str(main_joints)].step()
            train_loss.update(loss.item(),inp.size(0))

            # adjust loss alpha
            '''
                ratios : 
                    1,   0.2
                    0.5, 0.1
            '''
            # last_loss_alpha[0] = 0.2 * main_left_loss.avg / main_right_loss.avg
            # last_loss_alpha[1] = 0.5 * main_left_loss.avg / aux_left_loss.avg
            # last_loss_alpha[2] = 0.1 * main_left_loss.avg / aux_right_loss.avg

        logger.info("{}'epoch train time :{:<6.2f}s loss :{:.8f} ".format(epoch,time.time()-start_time,train_loss.avg))

        # adjust ratios
        # args.aux_ratios = args.aux_ratios * ratio_speed_array
        # mix_train_dataloader.change_ratios(args.aux_ratios)
        # args.loss_alpha = args.loss_alpha * args.loss_decay
        
        
        # print("Change Training data ratios : {} Mixed Loss alpha: {:.4f}".format(args.aux_ratios,args.loss_alpha))

        # validate main dataset
        val_nme, predictions = function.mix_val(config,main_val_loader,backbone,heads[str(config.MODEL.NUM_JOINTS)],criterion,epoch)
        
        # if args.show_others:
        #     # validate aux dataset
        #     for key in aux_configs.keys():
        #         function.mix_val(aux_configs[key],aux_dataloader['test'][key],backbone,heads[key],criterion,epoch)
        
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
        schedulers_heads[str(main_joints)].step()
        # lr_scheduler_backbone.step()
        # for key in schedulers_heads.keys():
        #     schedulers_heads[key].step()


if __name__ == '__main__':
    main()