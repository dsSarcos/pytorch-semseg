import copy
import torchvision.models as models

from ptsemseg.models.fcn import fcn8s, fcn16s, fcn32s
from ptsemseg.models.segnet import segnet
from ptsemseg.models.unet import unet
from ptsemseg.models.pspnet import pspnet
#from ptsemseg.models.icnet import icnet
from ptsemseg.models.linknet import linknet
from ptsemseg.models.frrn import frrn

import torch
import os

import timm

#from crfasrnn_pytorch.crfasrnn.crfasrnn_model import CrfRnnNet
#from crfrnn_layer.crfrnn.crf import CRF
from DPT.dpt.models import DPTSegmentationModel


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    #vgg_path = "/home/myid/das59179/checkpoints/vgg16_patches_best_model.pkl"

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        import torch.nn as nn
        vgg16.classifier[6] = nn.Linear(4096, n_classes)
        '''
        if True:
            if os.path.isfile(vgg_path):
                print(
                    "Loading vgg16 from checkpoint"
                )
                import torch.nn as nn
                vgg16.classifier[6] = nn.Linear(4096, n_classes)
                checkpoint = torch.load(vgg_path)

                model_state = checkpoint['model_state']
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for k, v in model_state.items():
                    if 'module.' in k:
                        k = k.replace('module.','')
                    new_state_dict[k]=v


                vgg16.load_state_dict(new_state_dict)
        else:
            print("No checkpoint found at '{}'".format(model_dict["vgg_path"]))
        '''
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        import torch.nn as nn
        vgg16.classifier[6] = nn.Linear(4096, n_classes)
        '''
        if True:
            if os.path.isfile(vgg_path):
                print(
                    "Loading vgg16 from checkpoint"
                )
                import torch.nn as nn
                vgg16.classifier[6] = nn.Linear(4096, n_classes)
                checkpoint = torch.load(vgg_path)
                
                model_state = checkpoint['model_state']
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for k, v in model_state.items():
                    if 'module' in k:
                        k = k.replace('module.', '')
                    new_state_dict[k] = v

                vgg16.load_state_dict(new_state_dict)
        else:
            print("No checkpoint found at '{}'".format(model_dict["vgg_path"]))
            '''
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    #elif name == "icnet":
    #    model = model(n_classes=n_classes, **param_dict)

    #elif name == "icnetBN":
    #    model = model(n_classes=n_classes, **param_dict)

    elif name == "vgg16":
        import torch.nn as nn
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, n_classes)

    elif name == "crfrnn":
        model = model(n_classes = n_classes, **param_dict)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.init_vgg16_params(vgg16)

    elif name == 'crfrnn2':
        model = model(n_classes, **param_dict)

    elif name == "dpt_large":
        model = model(num_classes = n_classes, backbone="vitl16_384", **param_dict)

    elif name == 'dpt_hybrid':
        model = model(num_classes = n_classes, backbone="vitb_rn50_384", **param_dict)

    elif name == 'dpt_base':
        model = model(num_classes = n_classes, backbone="vitb16_384", **param_dict)

    elif name == 'resnet50':
        model = timm.create_model('vit_base_resnet50_384', pretrained=True, num_classes=n_classes)

    elif name == 'vit_base':
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=n_classes)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            #"icnet": icnet,
            #"icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "vgg16": models.vgg16,
            #"crfrnn": CRF,
            #"crfrnn": CrfRnnNet,
            'dpt_large': DPTSegmentationModel,
            'dpt_hybrid': DPTSegmentationModel,
            'dpt_base': DPTSegmentationModel,
            'resnet50': None,
            'vit_base': None,
        }[name]
    except:
        raise ("Model {} not available".format(name))
