#import yaml
import torch
#import argparse
import timeit
import numpy as np

from torch.utils import data


from ptsemseg.models import get_model
from ptsemseg.loader.DSM import DSMLoader
from ptsemseg.metrics import averageMeter
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True

models = 'dpt_base', 'dpt_hybrid', 'fcn8s', 'segnet'

local_path = "/home/diego/cs4960R/SemSegSOTA/Datasets/dense_semantic_mapping/full_images"
dst = DSMLoader(local_path, is_transform=True, img_size=(480,480))

valloader = data.DataLoader(dst, batch_size=1, num_workers=0, shuffle=False)

image = None
label = None

for i, data_samples in enumerate(valloader):
    image, label = data_samples
    if i == 2:
        break

image = image.to("cuda:0")

#print(image)

outputs = [torch.squeeze(image.cpu()), torch.squeeze(label.cpu())]

for model in models:
    print(model)
    path = "/home/diego/cs4960R/MatSee/checkpoints/{}_dsm_best_model.pkl".format(model)
    model = get_model({"arch": model}, n_classes=23)
    state = convert_state_dict(torch.load(path)["model_state"])
    model.load_state_dict(state)
    model.to("cuda:0")
    model.eval()
    output = model(image)
    pred = torch.squeeze(output.data.max(1)[1].cpu())
    outputs.append(pred)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

for x in outputs:
     #x = torch.squeeze(x)
     print(x.shape)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    xda = True
    for i, img in enumerate(imgs):
        img = img.detach()
        if xda:
            img = img.numpy()[::-1,:,:]
            img = np.transpose(img, [1,2,0])
            img = dst.std * img + dst.mean
            #img = F.to_pil_image(img)
            #axs[0, i].imshow(np.asarray(img))
            xda = False
        else:
            img = dst.decode_segmap(img.numpy())
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#torch.stack(outputs)
#out = make_grid(outputs)

#mshow(out, title=[models[x] for x in range(len(models))] )
show(outputs)

plt.waitforbuttonpress(timeout=100)
'''
for i, output in outputs:
        imgs = imgs.numpy()[::-1, :, :]
        imgs = np.transpose(imgs, [1, 2, 0])
        imgs = dst.std * imgs + dst.mean
        f, axarr = plt.subplots(len(outputs), 2)
        for j in range(len(outputs)):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
           break
        else:
            plt.close()
'''
    


        

