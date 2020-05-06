import os
import pprint
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function


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

    parser.add_argument('--cfg', help='experiment configuration filename',
                         type=str, default="experiments/wflw/face_alignment_wflw_hrnet_w18.yaml")
    parser.add_argument('--model-file', help='model parameters',  type=str, default="hrnetv2_pretrained/HR18-WFLW.pth")

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config.GPUS)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)

    devices = torch.device("cuda:0")
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()
    model.to(devices)

    print("load model success.")

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    print("load weight paramters success.")

    for file_path in test_files[str(config.DATASET.DATASET)]:
        config["DATASET"]["TESTSET"] = file_path
        dataset_type = get_dataset(config)

        test_loader = DataLoader(
            dataset=dataset_type(config,
                                is_train=False),
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        )
        # while True:
        start_time = time.time()

        print(file_path)
        nme, predictions = function.inference(config, test_loader, model)

        print("epoch time : {}".format(time.time()-start_time))

        # torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

