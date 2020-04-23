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
from lib.core import function
from lib.utils import utils

# import torch.distributed as dist
# dist.init_process_group(backend='nccl')

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        type=str, default="experiments/wflw/face_alignment_wflw_hrnet_w18.yaml")

    parser.add_argument('--load_folder',type=str,default='output/WFLW/face_alignment_wflw_hrnet_w18')
    parser.add_argument('--save_folder',type=str,default='output/WFLW/face_alignment_wflw_hrnet_w18/lossL1')
    parser.add_argument('--load_epoch',type=bool,default=False,help="If load epoch and lr infos")
    parser.add_argument('--load_best',type=bool,default=True,help="If load best checkpoint.")
    parser.add_argument('--pretrained',type=str,default="")
    parser.add_argument('--gpus',type=str,default="0")
    parser.add_argument('--lr',type=float,default=1e-4)

    args = parser.parse_args()
    update_config(config, args)
    
    config["TRAIN"]["LR"] = args.lr
    if args.pretrained :
        config["MODEL"]["PRETRAINED"] = args.pretrained

    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    final_output_dir = args.load_folder
    save_output_dir = args.save_folder
    if not os.path.exists(save_output_dir):
        os.mkdir(save_output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config.GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    devices = torch.device("cuda:0")
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    model.to(devices)
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=range(torch.cuda.device_count()))
    logger.info("load model success.")

    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    # criterion = torch.nn.SmoothL1Loss(size_average=True).cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            if args.load_epoch : 
                last_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            best_checkpoint_path = os.path.join(final_output_dir,'model_best.pth')
            if args.load_best and os.path.exists(best_checkpoint_path):
                best_checkpoint = torch.load(best_checkpoint_path)
                model.load_state_dict(best_checkpoint)
                best_nme = checkpoint['best_nme']
                logger.info("=> loaded best checkpoint.")
        else:
            logger.info("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_dataset = dataset_type(config,is_train=True)
    val_dataset = dataset_type(config,is_train=False)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*torch.cuda.device_count(),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        # sampler=train_sampler
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*torch.cuda.device_count(),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
        # sampler=val_sampler
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        
        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(save_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model.state_dict(),
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, save_output_dir, 'checkpoint_{}.pth'.format(epoch))

        lr_scheduler.step()


    final_model_state_file = os.path.join(save_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()