# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from PIL import Image
from torchvision.datasets.folder import ImageFolder, default_loader
import torch
import warnings

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class TxtMultiLabelDataset(torch.utils.data.Dataset):
    """
    Multi-label dataset from a txt file.

    Each line:
        <img_path> <v0,v1,v2,...,v{C-1}>
    where vi is 0/1.
    """
    def __init__(self, txt_path: str, transform=None, num_classes=None):
        self.txt_path = txt_path
        self.transform = transform
        self.samples = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Bad line in {txt_path}: {line}")
                img_path, label_str = parts[0], parts[1]
                label_str = label_str.replace(' ', '')
                vec = [float(x) for x in label_str.split(',') if x != '']
                self.samples.append((img_path, vec))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {txt_path}")

        inferred = len(self.samples[0][1])
        if num_classes is None:
            num_classes = inferred

        if num_classes != inferred:
            fixed = []
            for path, vec in self.samples:
                if len(vec) < num_classes:
                    vec = vec + [0.0] * (num_classes - len(vec))
                elif len(vec) > num_classes:
                    vec = vec[:num_classes]
                fixed.append((path, vec))
            self.samples = fixed

        self.nb_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n = len(self.samples)

        # 最多尝试 n 次，避免极端情况下全是坏图导致死循环
        for k in range(n):
            j = (idx + k) % n
            img_path, vec = self.samples[j]

            # 跳过 0 字节
            try:
                if os.path.getsize(img_path) == 0:
                    warnings.warn(f"Skip empty image: {img_path}")
                    continue
            except OSError as e:
                warnings.warn(f"Skip unreadable path: {img_path} ({e})")
                continue

            # 尝试读图
            try:
                with Image.open(img_path) as im:
                    if im.mode == "P":
                        im = im.convert("RGBA")
                    img = im.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                warnings.warn(f"Skip bad image: {img_path} ({e})")
                continue

            # transform
            if self.transform is not None:
                img = self.transform(img)

            # target 永远是 tensor
            target = torch.tensor(vec, dtype=torch.float32)
            return img, target


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    # datasets.py 里 build_dataset 的 TXT 分支替换为下面这样
    elif args.data_set == 'TXT':
        nb_classes = None
        if getattr(args, 'label_txt', ''):
            with open(args.label_txt, 'r', encoding='utf-8') as lf:
                nb_classes = sum(1 for _ in lf if _.strip())

        if getattr(args, 'predict_only', False):
            # 推理：只读图片路径
            if not args.val_txt:
                raise ValueError("predict_only requires --val-txt to be set")
            dataset = TxtImageOnlyDataset(args.val_txt, transform=transform)
            # 推理也需要 nb_classes 用于输出向量长度（来自 label.txt）
            if nb_classes is None:
                raise ValueError("predict_only requires --label-txt to infer num classes")
            return dataset, nb_classes

        # 训练/验证：仍然走你的多标签 0/1 向量
        list_path = args.train_txt if is_train else (args.val_txt or args.train_txt)
        dataset = TxtMultiLabelDataset(list_path, transform=transform, num_classes=nb_classes)
        nb_classes = dataset.nb_classes
        return dataset, nb_classes



    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

# datasets.py 里新增（或放在 TxtMultiLabelDataset 下面）
import os
import warnings
import torch
from PIL import Image, UnidentifiedImageError

class TxtImageOnlyDataset(torch.utils.data.Dataset):
    """
    Inference dataset from a txt file.

    Each line:
        <img_path>
    """
    def __init__(self, txt_path: str, transform=None):
        self.txt_path = txt_path
        self.transform = transform
        self.samples = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                # 只取第一个字段当作路径（即使行里有多余空格）
                img_path = p.split()[0]
                self.samples.append(img_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n = len(self.samples)
        for k in range(n):
            j = (idx + k) % n
            img_path = self.samples[j]

            try:
                if os.path.getsize(img_path) == 0:
                    warnings.warn(f"Skip empty image: {img_path}")
                    continue
            except OSError as e:
                warnings.warn(f"Skip unreadable path: {img_path} ({e})")
                continue

            try:
                with Image.open(img_path) as im:
                    if im.mode == "P":
                        im = im.convert("RGBA")
                    img = im.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                warnings.warn(f"Skip bad image: {img_path} ({e})")
                continue

            if self.transform is not None:
                img = self.transform(img)

            # 返回 img + 路径（推理要用）
            return img, img_path

        # 全部坏图兜底
        raise RuntimeError(f"All images are unreadable around idx={idx}")
