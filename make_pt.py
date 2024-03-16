import torch
import timm
from collections import OrderedDict

basepath = '/home/myid/das59179/checkpoints/'

savepath = '/home/myid/das59179/experiment/'

def convert(path):
    checkpoint = torch.load(basepath+path, map_location="cpu")

    temp_dict = checkpoint['model_state']
    state_dict = OrderedDict()
    for k, v in temp_dict.items():
        if 'module.' in k:
            k = k.replace('module.','') 
        state_dict[k] = v
    
    #print(state_dict)

    state = {
            "state_dict": state_dict
    }

    torch.save(state, savepath + path)




if __name__ == '__main__':
    #basepath = '/home/myid/das59179/checkpoints/'

    #respath = 'resnet50_patches_best_model.pkl'

    #vitpath = 'vit_base_patches_best_model.pkl'
    
    #convert(respath)

    #convert(vitpath)

    #import torchvision.models as models

    #model = timm.create_model('vit_base_resnet50_384', pretrained=True)

    #print(model.state_dict())

    #model = torch.load(savepath + respath)
    #print(model)

    dpt_b = 'dpt_base_dsm_best_model.pkl'
    dpt_h = 'dpt_hybrid_dsm_best_model.pkl'
    fcn8s = 'fcn8s_dsm_best_model.pkl'
    segnet = 'segnet_dsm_best_model.pkl'

    convert(dpt_b)
    convert(dpt_h)
    convert(fcn8s)
    convert(segnet)
