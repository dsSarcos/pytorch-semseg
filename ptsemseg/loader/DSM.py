import collections
import torch
import numpy as np

from skimage.io import imread, imsave
#from cv2 import imread
from skimage.transform import resize as imresize

from torch.utils import data

#from torchvision.io import read_image as imread, ImageReadMode
from torchvision.transforms import Compose as Compos, Normalize, Resize, ToTensor, ToPILImage

#from ptsemseg.utils import recursive_glob
#from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale

dsm =[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, ]

class DSMLoader(data.Dataset):
    def __init__(
            self,
            root,
            split="training",
            img_size=(500,500),
            test_mode=False,
            augmentations=None,
            is_transform=False,
            img_norm=True,
    ):
        self.root = root
        self.n_classes = 23
        self.img_size = img_size
        self.mean = np.array([132.364979754 / 255.0, 117.155860847 / 255.0, 101.518156079 / 255.0])
        self.std = np.array([131.931048762 / 255.0, 116.707819938 / 255.0, 100.898974988 / 255.0])
        self.test_mode = test_mode
        self.augmentations = augmentations
        self.is_transform = is_transform
        self.transform = Compos([
            ToPILImage(mode="RGB"),
            Resize(img_size),
            ToTensor(),
            Normalize(self.mean, self.std)
        ])

        self.img_norm=img_norm

        split_map = {"training": "train", "val": "test"}
        self.split = split_map[split]

        self.images = collections.defaultdict(list)

        self.cmap = self.color_map(normalized=False)

        for split in ["train", "test"]:
            imgs = []
            with open(root + f'/{split}.txt') as f:
                lines = f.readlines()
                for line in lines:
                    imgname = line.strip()
                    imgs.append(imgname)
            self.images[split] = imgs

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, idx):
        img_path = self.root + '/original_scaled/' + self.images[self.split][idx] + '.jpg'
        lbl_path = self.root + '/label_scaled/' + self.images[self.split][idx] + '.png'
        

        img = imread(img_path)

        lbl = imread(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img = img[:,:,::-1] # RGB -> BGR
            img = self.transform(img)
            lbl = self.target_transform(lbl)

        return img, lbl
    
    def target_transform(self,lbl):
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = imresize(lbl, (self.img_size[0], self.img_size[1]), order=0)
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        lbl = torch.from_numpy(lbl).long()
        return lbl
    
    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == '__main__':
    '''
    import cv2 as cv
    path = '/home/diego/cs4960R/SemSegSOTA/Datasets/dense_semantic_mapping/full_images'

    test = DSMLoader(path, is_transform=False)

    #labels = ["brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair", "leather", "metal", "mirror", "other", "painted", "paper", "plastic", "polyshedstone", "skin", "sky", "stone", "tile", "wallpaper", "water", "wood", "useless"]

    for i in range(10):

        sample = test[i]

        
        img, label = sample

        print(img)
        print(label)
        print(np.unique(label))

        cv.imshow('image', img)
        cv.imshow('label', label)

        cv.waitKey(0)

        cv.destroyAllWindows()

    print(len(test))
    '''
    import matplotlib.pyplot as plt
    import cv2
    #augmentations = Compose([Scale(512), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "/home/diego/cs4960R/SemSegSOTA/Datasets/dense_semantic_mapping/full_images"
    dst = DSMLoader(local_path, is_transform=True, split="val")
    
    image, label = dst[2]
    cv2.imshow("pic", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(image)
    #print(label)
    #trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=False)
    #for i, data_samples in enumerate(trainloader):
    #    imgs, labels = data_samples
    #    imgs = imgs.numpy()[:, ::-1, :, :]
    #    imgs = np.transpose(imgs, [0, 2, 3, 1])
    #    imgs = dst.std * imgs + dst.mean
    #    f, axarr = plt.subplots(bs, 2)
    #    for j in range(bs):
    #        axarr[j][0].imshow(imgs[j])
    #        axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    #    plt.show()
    #    a = input()
    #    if a == "ex":
    #       break
    #   else:
    #        plt.close()
    
