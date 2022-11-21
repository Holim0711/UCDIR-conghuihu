import os
import random
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def folder_content_getter(folder_path):

    cate_names = list(np.sort(os.listdir(folder_path)))

    if 'domainnet' in folder_path.lower():
        cate_names = ['bird', 'feather', 'teapot', 'tiger', 'whale', 'windmill', 'zebra']

    image_path_list = []
    image_cate_list = []

    for cate_name in cate_names:
        sub_folder_path = os.path.join(folder_path, cate_name)
        if os.path.isdir(sub_folder_path):
            image_names = list(np.sort(os.listdir(sub_folder_path)))
            for image_name in image_names:
                image_path = os.path.join(sub_folder_path, image_name)
                image_path_list.append(image_path)
                image_cate_list.append(cate_names.index(cate_name))

    return image_path_list, image_cate_list

class EvalDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,
                 datasetB_dir):

        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])


        if datasetA_dir == datasetB_dir and 'deepfashion1' in datasetA_dir.lower():
            print("----- DeepFashion1 Test Dataset -----")
            data = open(os.path.join(datasetA_dir, 'Eval/list_eval_partition.txt')).readlines()[2:]
            data = list(map(str.split, data))
            data = list(filter(lambda x: x[-1] == 'test', data))

            self.image_paths_A = sorted(list(set(map(itemgetter(0), data))))
            self.image_paths_B = sorted(list(set(map(itemgetter(1), data))))

            self.image_cates_A = [0] * len(self.image_paths_A)
            self.image_cates_B = [0] * len(self.image_paths_B)

            bboxes = open(os.path.join(datasetA_dir, 'Anno/list_bbox_consumer2shop.txt')).readlines()[2:]
            bboxes = list(map(str.split, bboxes))
            bboxes = {img: (int(x1), int(y1), int(x2), int(y2)) for img, _, _,  x1, y1, x2, y2 in bboxes}
            self.bboxes_A = [bboxes[x] for x in self.image_paths_A]
            self.bboxes_B = [bboxes[x] for x in self.image_paths_B]

            def reduce_multiple_results(data):
                d = defaultdict(set)
                [d[u].add(i) for u, i in data]
                return list(d.items())

            indexed_data = [
                (self.image_paths_A.index(u), self.image_paths_B.index(i))
                for u, i, _, _ in tqdm(data)
            ]
            self.indexed_data = reduce_multiple_results(indexed_data)

            def get_img_path(x):
                return os.path.join(datasetA_dir, 'Img', x)

            self.image_paths_A = list(map(get_img_path, self.image_paths_A))
            self.image_paths_B = list(map(get_img_path, self.image_paths_B))
        else:
            self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir)
            self.image_paths_B, self.image_cates_B = folder_content_getter(datasetB_dir)

        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        box_A = self.bboxes_A[index_A]
        box_B = self.bboxes_B[index_B]

        image_A = self.transform(Image.open(image_path_A).convert('RGB').crop(box_A))
        image_B = self.transform(Image.open(image_path_B).convert('RGB').crop(box_B))

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)


class TrainDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,
                 datasetB_dir,
                 aug_plus):

        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if aug_plus:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )

        if datasetA_dir == datasetB_dir and 'deepfashion1' in datasetA_dir.lower():
            print("----- DeepFashion1 Training Dataset -----")
            data = open(os.path.join(datasetA_dir, 'Eval/list_eval_partition.txt')).readlines()[2:]
            data = list(map(str.split, data))
            data = list(filter(lambda x: x[-1] == 'train', data))

            self.image_paths_A = sorted(list(set(map(itemgetter(0), data))))  # Consumer
            self.image_paths_B = sorted(list(set(map(itemgetter(1), data))))  # Shop

            self.image_cates_A = [0] * len(self.image_paths_A)
            self.image_cates_B = [0] * len(self.image_paths_B)

            bboxes = open(os.path.join(datasetA_dir, 'Anno/list_bbox_consumer2shop.txt')).readlines()[2:]
            bboxes = list(map(str.split, bboxes))
            bboxes = {img: (int(x1), int(y1), int(x2), int(y2)) for img, _, _,  x1, y1, x2, y2 in bboxes}
            self.bboxes_A = [bboxes[x] for x in self.image_paths_A]
            self.bboxes_B = [bboxes[x] for x in self.image_paths_B]

            def get_img_path(x):
                return os.path.join(datasetA_dir, 'Img', x)

            self.image_paths_A = list(map(get_img_path, self.image_paths_A))
            self.image_paths_B = list(map(get_img_path, self.image_paths_B))
        else:
            self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir)
            self.image_paths_B, self.image_cates_B = folder_content_getter(datasetB_dir)

        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        if index >= self.domainA_size:
            index_A = random.randint(0, self.domainA_size - 1)
        else:
            index_A = index

        if index >= self.domainB_size:
            index_B = random.randint(0, self.domainB_size - 1)
        else:
            index_B = index

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        box_A = self.bboxes_A[index_A]
        box_B = self.bboxes_B[index_B]

        x_A = Image.open(image_path_A).convert('RGB').crop(box_A)
        q_A = self.transform(x_A)
        k_A = self.transform(x_A)

        x_B = Image.open(image_path_B).convert('RGB').crop(box_B)
        q_B = self.transform(x_B)
        k_B = self.transform(x_B)

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return [q_A, k_A], index_A, [q_B, k_B], index_B, target_A, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)


class DetTrainDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,
                 datasetB_dir):

        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])


        if datasetA_dir == datasetB_dir and 'deepfashion1' in datasetA_dir.lower():
            print("----- Deterministic DeepFashion1 Training Dataset -----")
            data = open(os.path.join(datasetA_dir, 'Eval/list_eval_partition.txt')).readlines()[2:]
            data = list(map(str.split, data))
            data = list(filter(lambda x: x[-1] == 'train', data))

            self.image_paths_A = sorted(list(set(map(itemgetter(0), data))))  # Consumer
            self.image_paths_B = sorted(list(set(map(itemgetter(1), data))))  # Shop

            self.image_cates_A = [0] * len(self.image_paths_A)
            self.image_cates_B = [0] * len(self.image_paths_B)

            bboxes = open(os.path.join(datasetA_dir, 'Anno/list_bbox_consumer2shop.txt')).readlines()[2:]
            bboxes = list(map(str.split, bboxes))
            bboxes = {img: (int(x1), int(y1), int(x2), int(y2)) for img, _, _,  x1, y1, x2, y2 in bboxes}
            self.bboxes_A = [bboxes[x] for x in self.image_paths_A]
            self.bboxes_B = [bboxes[x] for x in self.image_paths_B]

            def get_img_path(x):
                return os.path.join(datasetA_dir, 'Img', x)

            self.image_paths_A = list(map(get_img_path, self.image_paths_A))
            self.image_paths_B = list(map(get_img_path, self.image_paths_B))
        else:
            self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir)
            self.image_paths_B, self.image_cates_B = folder_content_getter(datasetB_dir)

        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        box_A = self.bboxes_A[index_A]
        box_B = self.bboxes_B[index_B]

        image_A = self.transform(Image.open(image_path_A).convert('RGB').crop(box_A))
        image_B = self.transform(Image.open(image_path_B).convert('RGB').crop(box_B))

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)
