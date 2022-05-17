import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import math
from tqdm import tqdm
from loguru import logger

class SaliencyDataset(Dataset):
    def __init__(self, data_root, img_list, gt_list, name="LEDOV_train", img_size=(416, 416), transform=None, target_transform=None, cache=False):
        super().__init__()
        self.img_size=img_size
        self.data_root = data_root
        self.img_list = []
        self.heatmap_list = []
        self.resized_info = []
        self.img_info = []
        f = open(img_list)
        lines = f.readlines()
        f.close()
        for line in lines:
            img_path, h, w = line.strip().split()
            h, w = int(h), int(w)
            self.img_list.append(img_path)
            r = min(self.img_size[0] / h, self.img_size[1] / w)
            self.resized_info.append([int(h * r), int(w * r)])
            self.img_info.append((h, w))

        f = open(gt_list)
        self.heatmap_list = f.readlines()
        f.close()
        self.heatmap_list = [x.strip() for x in self.heatmap_list]

        print(len(self.img_list), len(self.heatmap_list))
   
        assert(len(self.img_list) == len(self.heatmap_list))
        self.transform = transform
        self.target_transform = target_transform
        self.name = name
        self.imgs = None
        if cache:
            self._cache_images()
        
    def __len__(self):
        return len(self.heatmap_list)

    def __del__(self):
        del self.imgs

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_root, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.img_list), max_h, max_w, 4),
                dtype=np.uint8,
                mode="w+",
            )
            self.resized_info = [[] for x in range(len(self.img_list))]
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(64, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.img_list)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.img_list))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.img_list), max_h, max_w, 4),
            dtype=np.uint8,
            mode="r+",
        )

    def load_resized_img(self, index):
        img_gt = self.load_image(index)
        r = min(self.img_size[0] / img_gt.shape[0], self.img_size[1] / img_gt.shape[1])
        resized_img = cv2.resize(
            img_gt,
            (int(img_gt.shape[1] * r), int(img_gt.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        im = cv2.imread(os.path.join(self.data_root, self.img_list[index]))
        gt = cv2.imread(os.path.join(self.data_root, self.heatmap_list[index]))
        gt = gt[:,:,:1]
        img_gt = np.dstack([im, gt])
        assert img_gt is not None, f"file named {self.img_list[index]} not found"

        return img_gt

    def padding_resized_img(self, resized_img):
        img_shape = resized_img.shape
        padded_img = np.ones((self.img_size[0], self.img_size[1], img_shape[2]), dtype=np.uint8) * 114
        padded_img[: img_shape[0], : img_shape[1]] = resized_img
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img


    def __getitem__(self, index):
        #print(index)
        img_gt = self.pull_item(index)
        img_gt = self.padding_resized_img(img_gt)
        im = img_gt[:,:,:3].copy()
        gt = img_gt[:,:,3].copy()
        im = im.transpose((2,0,1))
        im = torch.from_numpy(im)
        im = im.float().div(255)
        im = im.sub_(torch.FloatTensor([0.485,0.456,0.406]).view(3,1,1)).div_(torch.FloatTensor([0.229,0.224,0.225]).view(3,1,1))
        
        gt = torch.from_numpy(gt)
        gt = gt.float().div(255)
        #gt = gt.unsqueeze(0)
        # sample = {'image': im, 'flow': flowarr, 'gt': gt, 'fixsac': torch.FloatTensor([self.fixsac[index]]), 'imname': self.listTrainFiles[index]}
        return im, gt.unsqueeze(0), self.img_info[index], -1

    def pull_item(self, index):
        resized_info = self.resized_info[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            #print(pad_img.shape)
            #print(resized_info)
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
        return img


if __name__ == '__main__':
    data_root = '../../../datasets/LEDOV/LEDOV/img_and_heatmap/val_1/'
    img_list = '../../../datasets/LEDOV/LEDOV/img_and_heatmap/val_imgs1.lst'
    gt_list = '../../../datasets/LEDOV/LEDOV/img_and_heatmap/val_gts.lst'
    dataset = SaliencyDataset(data_root, img_list, gt_list, name="LEDOV_val", img_size=(416, 416), transform=None, target_transform=None, cache=True)
    STTrainLoader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    import cv2
    #STValData = STDataset(imgPath, imgPath_s, gtPath, listFolders, listValFiles, listValGtFiles, listfixsacVal, fixsacPath)
    #STValLoader = DataLoader(dataset=STValData, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    #print(len(STValLoader))
    #print(len(STTrainLoader))
    for i in tqdm(STTrainLoader):
        img, target, _, _ = i
        for ii in range(10):
            im = img[ii, :,:,:]
            im = im.mul_(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1)).add_(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1))
            im = im*255.0
            im = im.numpy().astype(np.uint8)
            im = im.transpose((1,2,0))
            cv2.imwrite('%d.jpg'%ii, im)
            label = target[ii,:,:,: ]*255.0
            label = label.squeeze().numpy().astype(np.uint8)
            cv2.imwrite('%d_label.png'%ii, label)
        print(img.shape, target.shape)
        break
        


    
