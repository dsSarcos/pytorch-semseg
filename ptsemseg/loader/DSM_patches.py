import collections
import torch
#import torchvision.transforms as transforms
import numpy as np

from skimage.io import imread

from torch.utils import data

from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage

class VGGPatchesLoader(data.Dataset):
    def __init__(
            self,
            root,
            split="training",
            img_size=(256,256),
            test_mode=False,
            is_transform = False,
            augmentations=None,
    ):
        self.classes = ["brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair", "leather", "metal", "mirror", "other", "painted", "paper", "plastic", "polyshedstone", "skin", "sky", "stone", "tile", "wallpaper", "water", "wood"]
        self.root = root
        self.n_classes = 23
        self.img_size = img_size
        self.mean = np.array([132.364979754 / 255.0, 117.155860847 / 255.0, 101.518156079 / 255.0])
        self.std = np.array([131.931048762 / 255.0, 116.707819938 / 255.0, 100.898974988 / 255.0])
        self.test_mode = test_mode
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.transform = Compose([
            ToPILImage(mode="RGB"),
            Resize(img_size),
            ToTensor(),
            Normalize(self.mean, self.std)
        ])

        split_map = {"training": "train", "val": "test"}
        self.split = split_map[split]

        self.images = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)

        for split in ["train", "test"]:
            imgs = []
            lbls = []
            with open(root + f'/{split}.txt') as f:
                lines = f.readlines()
                for line in lines:
                    imgname, lbl = line.strip().split(' ')
                    imgs.append(imgname)
                    lbls.append(lbl)
            self.images[split] = imgs
            self.labels[split] = lbls

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, idx):
        img_path = self.root + self.images[self.split][idx]        

        img = imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = int(self.labels[self.split][idx])

        if self.is_transform:
            img = img[:,:,::-1] # RGB -> BGR
            img = self.transform(img)

        return img, lbl
   


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    path = '/home/diego/cs4960R/SemSegSOTA/Datasets/dense_semantic_mapping/patch_images/patch_images_256_new'

    dst = VGGPatchesLoader(path, is_transform=True)

    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    title = dst.classes
    
    def imshow(inp, title=None):
        #inp = inp[:,:,::-1]
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([132.364979754 / 255.0, 117.155860847 / 255.0, 101.518156079 / 255.0])
        std = np.array([131.931048762 / 255.0, 116.707819938 / 255.0, 100.898974988 / 255.0])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    for i, data_sample in enumerate(trainloader):
        # Get a batch of training data
        inputs, classes = data_sample

        # Make a grid from batch
        out = make_grid(inputs)

        imshow(out, title=[dst.classes[x] for x in classes])

        a = input()
        if a == "ex":
            break
        else:
            plt.close()