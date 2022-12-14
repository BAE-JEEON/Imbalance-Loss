# import logging
# import math
# import random

# import numpy as np
# from PIL import Image
# from torchvision import datasets
# from torchvision import transforms

# from .randaugment import RandAugmentMC

# logger = logging.getLogger(__name__)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
# cifar100_mean = (0.5071, 0.4867, 0.4408)
# cifar100_std = (0.2675, 0.2565, 0.2761)
# normal_mean = (0.5, 0.5, 0.5)
# normal_std = (0.5, 0.5, 0.5)




# def get_cifar10(args, root, l_samples, u_samples):
#     transform_labeled = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(size=32,
#                             padding=int(32*0.125),
#                             padding_mode='reflect'),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
#     ])
#     base_dataset = datasets.CIFAR10(root, train=True, download=True)
#     train_labeled_idxs, train_unlabeled_idxs= train_split(args, base_dataset.targets, l_samples, u_samples)

#     train_labeled_dataset = CIFAR10SSL(
#         root, train_labeled_idxs, train=True,
#         transform=transform_labeled)

#     train_unlabeled_dataset = CIFAR10SSL(
#         root, train_unlabeled_idxs, train=True,
#         transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

#     test_dataset = datasets.CIFAR10(
#         root, train=False, transform=transform_val, download=False)

#     return train_labeled_dataset, train_unlabeled_dataset, test_dataset

# # def get_cifar10(args, root, l_samples, u_samples):
# #     transform_labeled = transforms.Compose([
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomCrop(size=32,
# #                             padding=int(32*0.125),
# #                             padding_mode='reflect'),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# #     ])
# #     transform_val = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# #     ])
# #     base_dataset = datasets.CIFAR10(root, train=True, download=True)
# #     train_labeled_idxs, train_unlabeled_idxs2= train_split(args, base_dataset.targets, l_samples, u_samples)
# #     print('train_unlabeled_idxs2',train_unlabeled_idxs2, len(train_unlabeled_idxs2))

# #     print('base_dataset.data', base_dataset.data[[train_unlabeled_idxs]])
# #     train_unlabeled_dataset = []
# #     for i in range(args.num_classes):
# #         idxs = train_unlabeled_idxs2[i]
# #         train_unlabeled_dataset.extend(CIFAR10SSL(
# #         root, idxs, train=True,
# #         transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std, class_num = i)))
# #     random.shuffle(train_unlabeled_dataset) 

# #     train_labeled_dataset = CIFAR10SSL(
# #         root, train_labeled_idxs, train=True,
# #         transform=transform_labeled)



# #     test_dataset = datasets.CIFAR10(
# #         root, train=False, transform=transform_val, download=False)

# #     return train_labeled_dataset, train_unlabeled_dataset, test_dataset


# def get_cifar100(args, root):

#     transform_labeled = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(size=32,
#                             padding=int(32*0.125),
#                             padding_mode='reflect'),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

#     base_dataset = datasets.CIFAR100(
#         root, train=True, download=True)

#     train_labeled_idxs, train_unlabeled_idxs = x_u_split(
#         args, base_dataset.targets)

#     train_labeled_dataset = CIFAR100SSL(
#         root, train_labeled_idxs, train=True,
#         transform=transform_labeled)

#     train_unlabeled_dataset = CIFAR100SSL(
#         root, train_unlabeled_idxs, train=True,
#         transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

#     test_dataset = datasets.CIFAR100(
#         root, train=False, transform=transform_val, download=False)

#     return train_labeled_dataset, train_unlabeled_dataset, test_dataset


# def train_split(args, labels, n_labeled_per_class, n_unlabeled_per_class):
#     labels = np.array(labels)
#     train_labeled_idxs = []
#     train_unlabeled_idxs = []

#     for i in range(args.num_classes):
#         idxs = np.where(labels == i)[0]
#         train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
#         train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

#     if args.expand_labels or args.num_labeled < args.batch_size:
#         num_expand_x = math.ceil(
#             args.batch_size * args.eval_step / args.num_labeled)
#         train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(num_expand_x)])
#         train_unlabeled_idxs = np.hstack([train_unlabeled_idxs for _ in range(num_expand_x)])

#     np.random.shuffle(train_labeled_idxs)
#     np.random.shuffle(train_unlabeled_idxs)

#     return train_labeled_idxs, train_unlabeled_idxs

# # def train_split(args, labels, n_labeled_per_class, n_unlabeled_per_class):
# #     labels = np.array(labels)
# #     train_labeled_idxs = []
# #     train_unlabeled_idxs = []
# #     train_unlabeled_idxs2 = []

# #     for i in range(args.num_classes):
# #         idxs = np.where(labels == i)[0]
# #         train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
# #         train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
# #         train_unlabeled_idxs2.append(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    

# #     if args.expand_labels or args.num_labeled < args.batch_size:
# #         num_expand_x = math.ceil(
# #             args.batch_size * args.eval_step / args.num_labeled)
# #         train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(num_expand_x)])
# #         train_unlabeled_idxs = np.hstack([train_unlabeled_idxs for _ in range(num_expand_x)])

# #     np.random.shuffle(train_labeled_idxs)
# #     np.random.shuffle(train_unlabeled_idxs)

# #     return train_labeled_idxs, train_unlabeled_idxs2



# class TransformFixMatch(object):
#     def __init__(self, mean, std, class_num):
#         self.weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                 padding=int(32*0.125),
#                                 padding_mode='reflect')])
#         self.strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                 padding=int(32*0.125),
#                                 padding_mode='reflect'),
#             RandAugmentMC(n=2, m = 10+2*(class_num+1))])
#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)])

#     def __call__(self, x):
#         weak = self.weak(x)
#         strong = self.strong(x)
#         return self.normalize(weak), self.normalize(strong)


# class CIFAR10SSL(datasets.CIFAR10):
#     def __init__(self, root, indexs, train=True,
#                 transform=None, target_transform=None,
#                 download=False):
#         super().__init__(root, train=train,
#                         transform=transform,
#                         target_transform=target_transform,
#                         download=download)
#         if indexs is not None:
#             self.data = self.data[indexs]
#             self.targets = np.array(self.targets)[indexs]
            

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target


# class CIFAR100SSL(datasets.CIFAR100):
#     def __init__(self, root, indexs, train=True,
#                 transform=None, target_transform=None,
#                 download=False):
#         super().__init__(root, train=train,
#                         transform=transform,
#                         target_transform=target_transform,
#                         download=download)
#         if indexs is not None:
#             self.data = self.data[indexs]
#             self.targets = np.array(self.targets)[indexs]

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target


# DATASET_GETTERS = {'cifar10': get_cifar10,
#                 'cifar100': get_cifar100,
#                 }

import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)




def get_cifar10(args, root, l_samples, u_samples):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                            padding=int(32*0.125),
                            padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs= train_split(args, base_dataset.targets, l_samples, u_samples)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                            padding=int(32*0.125),
                            padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(args, labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(num_expand_x)])
        train_unlabeled_idxs = np.hstack([train_unlabeled_idxs for _ in range(num_expand_x)])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                transform=None, target_transform=None,
                download=False):
        super().__init__(root, train=train,
                        transform=transform,
                        target_transform=target_transform,
                        download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                transform=None, target_transform=None,
                download=False):
        super().__init__(root, train=train,
                        transform=transform,
                        target_transform=target_transform,
                        download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                'cifar100': get_cifar100,
                }

