from ptsemseg.loader.DSM import DSMLoader
from torch.utils import data
import numpy as np

import matplotlib.pylab as plt

import cv2

data_path = "/home/diego/cs4960R/SemSegSOTA/Datasets/dense_semantic_mapping/full_images"

dst = DSMLoader(
        data_path,
        split="val",
        is_transform=True,
        img_size=(500,500),
    )
bs=5
valloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
for i, data_samples in enumerate(valloader):
        imgs, labels = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        imgs = dst.std * imgs + dst.mean
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            plt.save("input/{}.jpg".format(j), imgs[j])
            plt.save("labels/{}.png".format(j),dst.decode_segmap(labels.numpy()[j]))
            
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()


