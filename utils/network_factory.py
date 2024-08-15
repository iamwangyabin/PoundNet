import torch
import torchvision


def get_model(conf):
    print("Model loaded..")
    if conf.arch == 'poundnet':
        from networks.poundnet_detector import PoundNet
        model = PoundNet(conf)

    return model







