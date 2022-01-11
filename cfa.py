import argparse
import ssl
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import random
import warnings
from tqdm import tqdm
from typing import List
from os import mkdir, remove, listdir
from os.path import join, exists, expanduser, isfile

parser = argparse.ArgumentParser('./cfa.py',
                                 description='Tackling catastrophic forgetting with knowledge amalgamation')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use',
                    choices=['usps', 'mnist', 'cifar10', 'cifar100', 'omniglot', 'omniglot10'])
parser.add_argument('--seed', type=int, default=None, metavar='N', help='Set a seed to compare runs')
parser.add_argument('--n_tasks', type=int, default=5, metavar='N', help='Number of tasks',
                    choices=[5, 10, 50])
parser.add_argument('--cuda', action='store_true', help='enable CUDA')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--cfl_lr', type=float, default=None, help='Common feature amalgamation learning rate')
parser.add_argument('--batch_size', type=int, default=2 ** 3, help='Batch size for base model training',
                    choices=[2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8])
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per task (base learning)')
parser.add_argument('--force_base_retraining', type=bool, default=False, help='Force base model retraining')
parser.add_argument('--amalgamation_strategy', type=str, default='all_together', help='Amalgamation Strategy',
                    choices=['all_together', 'one_at_a_time'])
parser.add_argument('--amalgamation_epochs', type=int, default=1000, help='Amalgamation epochs',
                    choices=[10, 100, 500, 1000])
parser.add_argument('--memory_size', type=int, default=500, help='Size of sample memory',
                    choices=[100, 200, 500, 1000])
parser.add_argument('--n_omniglot_augmentations', type=int, default=10,
                    help='Number of augmentation loops for omniglot dataset', choices=[0, 10])
parser.add_argument('--omniglot_test_ratio', type=float, default=0.3, help='Ratio of train to test in omniglot')


class CommonFeatureLearningLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super(CommonFeatureLearningLoss, self).__init__()
        self.beta = beta

    def forward(self, hs, ht, ft_, ft):
        kl_loss = 0.0
        mse_loss = 0.0
        for ht_i in ht:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                kl_loss += torch.nn.functional.kl_div(torch.log_softmax(hs, dim=1), torch.softmax(ht_i, dim=1))
        for i in range(len(ft_)):
            mse_loss += torch.nn.functional.mse_loss(ft_[i], ft[i])

        return kl_loss + self.beta * mse_loss


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                     kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=planes, out_channels=planes,
                                     kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.downsample = None
        if stride > 1 or inplanes != planes:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                                                  kernel_size=(1, 1), stride=stride, bias=False))

        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class CommonFeatureBlocks(torch.nn.Module):
    def __init__(self, n_student_channels: int, n_teacher_channels: List[int], n_hidden_channel: int):
        super(CommonFeatureBlocks, self).__init__()

        ch_s = n_student_channels  # Readability
        ch_ts = n_teacher_channels  # Readability
        ch_h = n_hidden_channel  # Readability

        self.align_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.align_t.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ch_t, out_channels=2 * ch_h, kernel_size=(1, 1), bias=False),
                torch.nn.ReLU(inplace=True)))

        self.align_s = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_s, out_channels=2 * ch_h, kernel_size=(1, 1), bias=False),
            torch.nn.ReLU(inplace=True))

        self.extractor = torch.nn.Sequential(ResidualBlock(inplanes=2 * ch_h, planes=ch_h, stride=1),
                                             ResidualBlock(inplanes=ch_h, planes=ch_h, stride=1),
                                             ResidualBlock(inplanes=ch_h, planes=ch_h, stride=1))

        self.dec_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.dec_t.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ch_h, out_channels=ch_t, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=ch_t, out_channels=ch_t, kernel_size=(1, 1), stride=1, padding=0, bias=False)))

    def forward(self, fs, ft):
        aligned_t = [align(f) for align, f in zip(self.align_t, ft)]
        aligned_s = self.align_s(fs)
        ht = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)
        ft_ = [dec(h) for dec, h in zip(self.dec_t, ht)]
        return hs, ht, ft_


class CustomDataLoader(torch.utils.data.Dataset):
    dataset = []
    transforms = None

    def __init__(self, dataset, transforms: T = None, n_tasks: int = 1):
        self.dataset = dataset
        self.transforms = transforms
        self.n_tasks = n_tasks
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.dataset.data = torch.tensor(self.dataset.data)
            self.dataset.targets = torch.tensor(self.dataset.targets)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < len(self):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                return {'x': self.transforms(self.dataset.data[idx]),
                        'y': self.dataset.targets[idx]}
        else:
            return None

    def __len__(self):
        return len(self.dataset)

    def shuffle(self):
        p = torch.randperm(len(self))
        self.dataset.data = self.dataset.data[p]
        self.dataset.targets = self.dataset.targets[p]


class MNISTUSPSDataLoader(CustomDataLoader):
    def __init__(self, dataset, transforms: T = None, n_tasks: int = 1):
        super(MNISTUSPSDataLoader, self).__init__(dataset, transforms, n_tasks)


class CIFARDataLoader(CustomDataLoader):
    def __init__(self, dataset, transforms: T = None, n_tasks: int = 1):
        super(CIFARDataLoader, self).__init__(dataset, transforms, n_tasks)


class OmniglotDataLoader(CustomDataLoader):
    def __init__(self, dataset, transforms: T = None):
        super(OmniglotDataLoader, self).__init__(dataset, transforms, 50)


class MyDataLoader:
    class MNIST(MNISTUSPSDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transforms)
            MNISTUSPSDataLoader.__init__(self, dataset, transforms, n_tasks)

    class USPS(MNISTUSPSDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.USPS(root='./data', train=train, download=True, transform=transforms)
            MNISTUSPSDataLoader.__init__(self, dataset, transforms, n_tasks)

    class CIFAR10(CIFARDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transforms)
            CIFARDataLoader.__init__(self, dataset, transforms, n_tasks)

    class CIFAR100(CIFARDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transforms)
            CIFARDataLoader.__init__(self, dataset, transforms, n_tasks)

    class Omniglot(OmniglotDataLoader):
        alphabets = [
            {'name': 'Alphabet_of_the_Magi', 'n_classes': 20},
            {'name': 'Anglo-Saxon_Futhorc', 'n_classes': 29},
            {'name': 'Arcadian', 'n_classes': 26},
            {'name': 'Armenian', 'n_classes': 41},
            {'name': 'Asomtavruli_(Georgian)', 'n_classes': 40},
            {'name': 'Balinese', 'n_classes': 24},
            {'name': 'Bengali', 'n_classes': 46},
            {'name': 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'n_classes': 14},
            {'name': 'Braille', 'n_classes': 26},
            {'name': 'Burmese_(Myanmar)', 'n_classes': 34},
            {'name': 'Cyrillic', 'n_classes': 33},
            {'name': 'Early_Aramaic', 'n_classes': 22},
            {'name': 'Futurama', 'n_classes': 26},
            {'name': 'Grantha', 'n_classes': 43},
            {'name': 'Greek', 'n_classes': 24},
            {'name': 'Gujarati', 'n_classes': 48},
            {'name': 'Hebrew', 'n_classes': 22},
            {'name': 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'n_classes': 16},
            {'name': 'Japanese_(hiragana)', 'n_classes': 52},
            {'name': 'Japanese_(katakana)', 'n_classes': 47},
            {'name': 'Korean', 'n_classes': 40},
            {'name': 'Latin', 'n_classes': 26},
            {'name': 'Malay_(Jawi_-_Arabic)', 'n_classes': 40},
            {'name': 'Mkhedruli_(Georgian)', 'n_classes': 41},
            {'name': 'N_Ko', 'n_classes': 33},
            {'name': 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'n_classes': 14},
            {'name': 'Sanskrit', 'n_classes': 42},
            {'name': 'Syriac_(Estrangelo)', 'n_classes': 23},
            {'name': 'Tagalog', 'n_classes': 17},
            {'name': 'Tifinagh', 'n_classes': 55},
            {'name': 'Angelic', 'n_classes': 20},
            {'name': 'Atemayar_Qelisayer', 'n_classes': 26},
            {'name': 'Atlantean', 'n_classes': 26},
            {'name': 'Aurek-Besh', 'n_classes': 26},
            {'name': 'Avesta', 'n_classes': 26},
            {'name': 'Ge_ez', 'n_classes': 26},
            {'name': 'Glagolitic', 'n_classes': 45},
            {'name': 'Gurmukhi', 'n_classes': 45},
            {'name': 'Kannada', 'n_classes': 41},
            {'name': 'Keble', 'n_classes': 26},
            {'name': 'Malayalam', 'n_classes': 47},
            {'name': 'Manipuri', 'n_classes': 40},
            {'name': 'Mongolian', 'n_classes': 30},
            {'name': 'Old_Church_Slavonic_(Cyrillic)', 'n_classes': 45},
            {'name': 'Oriya', 'n_classes': 46},
            {'name': 'Sylheti', 'n_classes': 28},
            {'name': 'Syriac_(Serto)', 'n_classes': 23},
            {'name': 'Tengwar', 'n_classes': 25},
            {'name': 'Tibetan', 'n_classes': 42},
            {'name': 'ULOG', 'n_classes': 26}
        ]

        def __init__(self, transforms: T = None, train: bool = True, is_omniglot10: bool = True):
            assert 0 < args.omniglot_test_ratio < 1., 'Test ratio must be between [0, 1]'

            class MyRotationTransform:
                def __init__(self, angle):
                    self.angle = angle

                def __call__(self, x):
                    return T.functional.rotate(x, self.angle)

            def get_datasets(transforms, train: bool = True):
                def list_files(root, suffix, prefix=False):
                    root = expanduser(root)
                    files = list(filter(lambda p: isfile(join(root, p)) and p.endswith(suffix), listdir(root)))

                    if prefix is True:
                        files = [join(root, d) for d in files]

                    return files

                def retrieve_test_train(dataset: torchvision.datasets.Omniglot, train: bool = True):
                    target_folder = dataset.target_folder
                    dataset._character_images = []

                    for char in dataset._characters:
                        if is_omniglot10 \
                                and not any([char10 in char for char10 in [f'character{i+1:02}' for i in range(10)]]):
                            continue

                        chars = [[(image, idx) for image in list_files(join(target_folder, character), '.png')]
                                 for idx, character in enumerate(dataset._characters) if char in character][
                            0]
                        if train:
                            chars = chars[int(len(chars) * args.omniglot_test_ratio):]
                        else:
                            chars = chars[:int(len(chars) * args.omniglot_test_ratio)]
                        dataset._character_images.append(chars)

                    dataset._flat_character_images = sum(dataset._character_images, [])

                    return dataset

                dataset_background = torchvision.datasets.Omniglot(root='./data', background=True,
                                                                   download=True, transform=transforms)
                max_background_class = max([x[1] for x in dataset_background._flat_character_images])
                dataset_evaluation = torchvision.datasets.Omniglot(root='./data', background=False,
                                                                   download=True, transform=transforms,
                                                                   target_transform=lambda x: x + max_background_class)

                dataset_background = retrieve_test_train(dataset_background, train)
                dataset_evaluation = retrieve_test_train(dataset_evaluation, train)

                return dataset_background, dataset_evaluation

            dataset = []

            cor_transforms = [T.Compose([
                T.RandomAffine(0, translate=(torch.rand(1), torch.rand(1))),
                T.ToTensor()
            ]) for _ in range(args.n_omniglot_augmentations)]

            rot_transforms = [T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                MyRotationTransform(int(torch.rand(1) * 40 - 20)),
                T.RandomRotation(20),
                T.ToTensor()
            ]) for _ in range(args.n_omniglot_augmentations)]

            dataset.extend(list(get_datasets(T.ToTensor(), train)))
            if train:
                [dataset.extend(list(get_datasets(cor, train))) for cor in cor_transforms]
                [dataset.extend(list(get_datasets(rot, train))) for rot in rot_transforms]

            dataset = torch.utils.data.ConcatDataset(dataset)
            dataset.data = []
            dataset.targets = []
            for idx, (sample, label) in enumerate(dataset):
                dataset.data.append(sample)
                dataset.targets.append(label)

            dataset.data = torch.cat(dataset.data)

            OmniglotDataLoader.__init__(self, dataset, transforms)

    def __init__(self, n_tasks: int = 1):
        self.data = []

        self.n_features = None
        self.n_classes = None
        self.min_class = 0
        self.max_class = np.inf
        self.n_tasks = n_tasks
        self.task_info = {}

        ssl._create_default_https_context = ssl._create_unverified_context

    def __len__(self):
        return len(self.data)

    def load_dataset(self, dataset: str, train: bool = True):
        class MyRotationTransform:
            def __init__(self, angle):
                self.angle = angle

            def __call__(self, x):
                return T.functional.rotate(x, self.angle)

        class MyTransposeTransform:
            def __call__(self, x):
                return x.T

        transforms = T.Compose([
            T.ToPILImage(),
            T.Grayscale(3),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        if dataset in ['mnist', 'usps']:
            if dataset == 'mnist':
                self.__load_mnist(transforms, train)
            elif dataset == 'usps':
                self.__load_usps(transforms, train)
        elif dataset in ['omniglot', 'omniglot10']:
            self.__load_omniglot(transforms, train, dataset == 'omniglot10')
        elif dataset in ['cifar10', 'cifar100']:
            transforms = T.Compose([
                MyTransposeTransform(),
                T.ToPILImage(),
                T.Resize((224, 224)),
                MyRotationTransform(-90),
                T.ToTensor()
            ])
            if dataset == 'cifar10':
                self.__load_cifar10(transforms, train)
            elif dataset == 'cifar100':
                self.__load_cifar100(transforms, train)

        # Initialize internal variables
        self.number_classes()
        self.number_features()

    def __load_mnist(self, transforms: T = None, train: bool = True):
        self.data = self.MNIST(transforms, train, self.n_tasks)

    def __load_usps(self, transforms: T = None, train: bool = True):
        self.data = self.USPS(transforms, train, self.n_tasks)

    def __load_cifar10(self, transforms: T = None, train: bool = True):
        self.data = self.CIFAR10(transforms, train, self.n_tasks)

    def __load_cifar100(self, transforms: T = None, train: bool = True):
        self.data = self.CIFAR100(transforms, train, self.n_tasks)

    def __load_omniglot(self, transforms: T = None, train: bool = True, is_omniglot10: bool = False):
        self.data = self.Omniglot(transforms, train, is_omniglot10)

    def shuffle(self):
        self.data.shuffle()

    def number_classes(self, force_count: bool = False):
        if self.n_classes is None or force_count:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.min_class = int(torch.tensor(self.data.dataset.targets).min(dim=0).values)
                self.max_class = int(torch.tensor(self.data.dataset.targets).max(dim=0).values)
            self.n_classes = 0
            for i in range(self.n_tasks):
                if isinstance(self.data, self.Omniglot):
                    n_elem_per_class = self.Omniglot.alphabets[i]['n_classes'] if args.dataset == 'omniglot' else 10
                    if args.dataset in ['omniglot10', 'omniglot']:
                        offset = np.sum([_['n_classes'] for _ in self.Omniglot.alphabets][:i])
                    self.task_info[i + 1] = {'n_elem': n_elem_per_class,
                                             'classes_list': list(range(0 if i == 0 else offset,
                                                                       (0 if i == 0 else offset) + n_elem_per_class)),
                                             'last_searched_internal_idx': 0,
                                             'last_searched_external_idx': 0}
                else:
                    n_elem_per_class = len(range(self.min_class, self.max_class + 1)) // self.n_tasks
                    self.task_info[i + 1] = {'n_elem': n_elem_per_class,
                                             'classes_list': list(range(self.min_class + i * n_elem_per_class,
                                                                        self.min_class + i * n_elem_per_class + n_elem_per_class)),
                                             'last_searched_internal_idx': 0,
                                             'last_searched_external_idx': 0}
                self.n_classes += self.task_info[i + 1]['n_elem']

        return self.n_classes

    def number_features(self, force_count: bool = False, specific_sample: int = None):
        if self.n_features is None or force_count or specific_sample is not None:
            item = 0 if specific_sample is None else specific_sample
            self.n_features = int(np.prod(self.get_x(item).shape))
        return self.n_features

    def get_x_y(self, item: int, n_task: int = 1):
        if item <= self.task_info[n_task]['last_searched_external_idx']:
            if item < self.task_info[n_task]['last_searched_external_idx']:
                self.task_info[n_task]['last_searched_internal_idx'] = 0
                self.task_info[n_task]['last_searched_external_idx'] = 0
                while item < self.task_info[n_task]['last_searched_external_idx']:
                    self.get_x(item=self.task_info[n_task]['last_searched_external_idx'], n_task=n_task)
            idx = self.task_info[n_task]['last_searched_internal_idx']
        else:
            idx = self.task_info[n_task]['last_searched_internal_idx'] + 1
            self.task_info[n_task]['last_searched_internal_idx'] = idx
            self.task_info[n_task]['last_searched_external_idx'] = item

        while self.data[idx] is not None and idx < len(self) and item < len(self):
            data = self.data[idx]
            if data is not None:
                self.task_info[n_task]['last_searched_internal_idx'] = idx
                y = data['y']
                if int(y) in self.task_info[n_task]['classes_list']:
                    y = y - self.min_class - self.task_info[n_task]['classes_list'][0]
                    return data['x'], y
            idx += 1
        return None, None

    def get_x(self, item: int, n_task: int = 1):
        return self.get_x_y(item, n_task)[0]

    def get_y(self, item: int, n_task: int = 1):
        return self.get_x_y(item, n_task)[1]


class MyComplexModel(torch.nn.Module):
    @property
    def feature_dimension(self):
        return list(self.resnet.layer4.children())[-1].bn2.num_features

    @property
    def soft_output(self):
        assert self.handles is not None
        return (self.layer1.output + \
               self.layer2.output + \
               self.layer3.output + \
               self.layer4.output + \
               self.resnet.fc.output)/5

    @property
    def features(self):
        assert self.handles is not None
        return self.resnet.layer4.output

    def __init__(self, n_input: int, n_output: int):
        super(MyComplexModel, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.handles = {}

        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 1, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(1),
            torch.nn.Linear(128 * 28 * 28, n_output)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 1, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.Flatten(1),
            torch.nn.Linear(256 * 14 * 14, n_output)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 1, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.Flatten(1),
            torch.nn.Linear(512 * 7 * 7, n_output)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 1, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Flatten(1),
            torch.nn.Linear(1024 * 4 * 4, n_output)
        )
        self.resnet.fc = torch.nn.Sequential(
            self.resnet.fc,
            torch.nn.Linear(1000, n_output)
        )

    def register_hooks(self):
        def forward_hook(module, _, output):
            module.output = output

        self.handles['conv1_layer'] = self.resnet.layer1.register_forward_hook(forward_hook)
        self.handles['conv2_layer'] = self.resnet.layer2.register_forward_hook(forward_hook)
        self.handles['conv3_layer'] = self.resnet.layer3.register_forward_hook(forward_hook)
        self.handles['conv4_layer'] = self.resnet.layer4.register_forward_hook(forward_hook)
        self.handles['fc_layer1'] = self.layer1.register_forward_hook(forward_hook)
        self.handles['fc_layer2'] = self.layer2.register_forward_hook(forward_hook)
        self.handles['fc_layer3'] = self.layer3.register_forward_hook(forward_hook)
        self.handles['fc_layer4'] = self.layer4.register_forward_hook(forward_hook)
        self.handles['fc_layer'] = self.resnet.fc.register_forward_hook(forward_hook)

    def remove_hooks(self):
        assert self.handles is not None
        for k, v in self.handles.items():
            self.handles[k].remove()

    def forward(self, x: torch.tensor):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x1 = self.layer1(x)
        x = self.resnet.layer2(x)
        x2 = self.layer2(x)
        x = self.resnet.layer3(x)
        x3 = self.layer3(x)
        x = self.resnet.layer4(x)
        x4 = self.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x1, x2, x3, x4, x

    def predict(self, x: torch.tensor):
        x1, x2, x3, x4, x = self.forward(x)

        x = torch.softmax(x, dim=1)
        x1 = torch.softmax(x1, dim=1)
        x2 = torch.softmax(x2, dim=1)
        x3 = torch.softmax(x3, dim=1)
        x4 = torch.softmax(x4, dim=1)

        return torch.stack([x1.argmax(1),
                            x2.argmax(1),
                            x3.argmax(1),
                            x4.argmax(1),
                            x.argmax(1)]).mode(0)[0]


class MySimpleModel(torch.nn.Module):
    @property
    def features(self):
        assert self.handles is not None
        return self.resnet.layer4.output

    @property
    def feature_dimension(self):
        return self.resnet.layer4[-1].conv2.out_channels

    @property
    def soft_output(self):
        return self.fc.output

    @property
    def n_output(self):
        return self.fc[-1].out_features

    def __init__(self, n_output: int):
        super(MySimpleModel, self).__init__()
        self.handles = {}

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc_backup = self.resnet.fc
        self.resnet.fc = torch.nn.Sequential()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.resnet.fc_backup.in_features, int(self.resnet.fc_backup.in_features/2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.resnet.fc_backup.in_features/2), n_output)
        )

    def register_hooks(self):
        def forward_hook(module, _, output):
            module.output = output

        self.handles['conv_layer'] = self.resnet.layer4.register_forward_hook(forward_hook)
        self.handles['fc_layer'] = self.fc.register_forward_hook(forward_hook)

    def remove_hooks(self):
        assert self.handles is not None
        for k, v in self.handles.items():
            self.handles[k].remove()

    def forward(self, x: torch.tensor):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def predict(self, x: torch.tensor):
        x = self.forward(x)
        return torch.softmax(x, dim=1).argmax(1)


def get_batch(current_idx: int, n_task: int, data_loader: MyDataLoader):
    batch_data, batch_target = [], []
    while len(batch_data) < args.batch_size and \
            data_loader.get_x(item=current_idx, n_task=n_task + 1) is not None:
        data, target = data_loader.get_x_y(item=current_idx, n_task=n_task + 1)
        if data is not None:
            batch_data.append(data), batch_target.append(target)
        current_idx += 1
    if len(batch_data) == 0:
        return current_idx, 0, None, None
    else:
        return current_idx, \
               len(batch_data), \
               torch.cat(batch_data).view(len(batch_data), 3, 224, 224).to(device), \
               torch.tensor(batch_target).to(device)


def save_load_best_model(model: MySimpleModel, dataset: MyDataLoader, n_task: int, is_train=True, pbar=None):
    path = f'./state'
    state_path = f'{path}/cafoal_{args.dataset}_{n_task + 1}'

    if not exists(path):
        mkdir(path)
    if not exists(state_path):
        torch.save(model.state_dict(), state_path)
    assert exists(path)
    assert exists(state_path)

    with torch.no_grad():
        model.eval()
        idx = 0
        corrects = 0
        total_task = 0
        if is_train:
            while dataset.get_x(item=idx, n_task=n_task + 1) is not None:
                idx, n_elem, data, target = get_batch(idx, n_task, dataset)
                total_task += n_elem
                corrects += int(sum(model.predict(data) == target))
            accuracy = (corrects / total_task) if total_task > 0 else 0
            description = f'Train accuracy for base task {n_task + 1}: {accuracy * 100:.2f}% ({corrects}/{total_task})'

            if pbar is None:
                print(description)
            else:
                pbar.set_description_str(description)
            torch.save(model.state_dict(), state_path)
            model.train()
        else:
            model.load_state_dict(torch.load(state_path))
            while dataset.get_x(item=idx, n_task=n_task + 1) is not None:
                idx, n_elem, data, target = get_batch(idx, n_task, dataset)
                total_task += n_elem
                corrects += int(sum(model.predict(data) == target))
            accuracy = (corrects / total_task) if total_task > 0 else 0
            print(f'Test accuracy for base task {n_task + 1}: {accuracy * 100:.2f}% ({corrects} / {total_task})')

    return model, accuracy


def amalgamate(teachers: List[MySimpleModel], data_array: List = [], labels: List = [],
               train: CustomDataLoader = None, test: CustomDataLoader = None, epochs: int = 100) -> MySimpleModel:
    class AverageTracker(object):
        def __init__(self):
            self.book = dict()

        def reset(self, key=None):
            item = self.book.get(key, None)
            if key is None:
                self.book.clear()
            elif item is not None:
                item[0] = 0  # value
                item[1] = 0  # count

        def update(self, key, val):
            record = self.book.get(key, None)
            if record is None:
                self.book[key] = [val, 1]
            else:
                record[0] += val
                record[1] += 1

        def get(self, key):
            record = self.book.get(key, None)
            assert record is not None
            return record[0] / record[1]

    def memory_keys(all_data, model, labels):
        n_elem = (len(labels) / (train.n_classes / train.n_tasks)) * args.memory_size
        n_elem = int(n_elem)
        return get_conf_keys(all_data, model, labels, n_elem)

    def get_conf_keys(all_data, model, labels, n_elem=args.memory_size):
        batch_sample = []
        batch_idx = []
        conf = {}
        for label in labels:
            conf[label] = {}
            for idx, data in enumerate(all_data):
                if data is None:
                    break

                if len(batch_sample) < args.batch_size and data is not None:
                    if int(data['y']) == label:
                        batch_sample.append(data['x'])
                        batch_idx.append(idx)
                elif data is None and len(batch_sample) == 0:
                    break
                elif len(batch_sample) > 0 or data is None:
                    samples = torch.cat(batch_sample).view(-1, 3, 224, 224).to(device)
                    batch_sample = []

                    soft_top_2 = model(samples).topk(2)[0].tolist()
                    for i, j in enumerate(batch_idx):
                        conf[label][j] = soft_top_2[i][0] - soft_top_2[i][1]
                    batch_idx = []

        idxs = []
        for key in conf.keys():
            idxs = idxs + list(dict(sorted(conf[key].items(),
                                           key=lambda x: x[1],
                                           reverse=True)).keys())[:int(np.ceil(n_elem/len(labels)))]

        return idxs[:n_elem]

    student = MySimpleModel(sum([teacher.n_output for teacher in teachers])).to(device)
    cfl_blk = CommonFeatureBlocks(student.feature_dimension,
                                  [teachers[0].feature_dimension, teachers[0].feature_dimension],
                                  int(sum([teacher.feature_dimension for teacher in teachers])/len(teachers))).to(device)

    cfl_lr = args.lr * 10 if args.cfl_lr is None else args.cfl_lr

    params_10x = [param for name, param in student.named_parameters() if 'fc' in name]
    params_1x = [param for name, param in student.named_parameters() if 'fc' not in name]
    optimizer = torch.optim.Adam([{'params': params_1x,            'lr': args.lr},
                                  {'params': params_10x,           'lr': args.lr * 10},
                                  {'params': cfl_blk.parameters(), 'lr': cfl_lr}])

    student.train()
    [teacher.register_hooks() for teacher in teachers]
    student.register_hooks()
    average_tracker = AverageTracker()

    common_feature_learning_criterion = CommonFeatureLearningLoss().to(device)

    data_idx = []
    for idx, data in enumerate(data_array):
        if data is None:
            data_idx.append(memory_keys(train.data, teachers[idx], labels[idx]))
        else:
            data_idx.append(data)
        data_array[idx] = torch.stack([train.data[idx_]['x'] for idx_ in data_idx[idx]])

    all_data = torch.cat([data for data in data_array])
    p = torch.randperm(len(all_data))
    all_data = all_data[p]

    student.eval()
    with torch.no_grad():
        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        b_accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for cur_step, data in enumerate(test.data):
            if data is None:
                break
            elif int(data['y']) not in labels_:
                continue

            label = torch.tensor(labels_.index(data['y'])).to(device)

            sample = data['x'].view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                if label in task_labels:
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            b_accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0

    student.train()
    with tqdm(unit='Epoch', total=epochs) as pbar:
        while pbar.n < epochs:
            average_tracker.reset()
            batch_sample = []
            soft_output_mask = []
            for cur_step, data in enumerate(all_data):
                if len(batch_sample) < args.batch_size and data is not None:
                    for idx, _ in enumerate(data_array):
                        if sum([len(d) for d in data_array][:idx]) <= \
                                int(p[cur_step]) < \
                                sum([len(d) for d in data_array][:idx+1]):
                            mask = np.concatenate([np.ones(teacher.n_output)
                                                   if idx_ == idx else np.zeros(teacher.n_output)
                                                   for idx_, teacher in enumerate(teachers)])
                            soft_output_mask.append(mask)
                    batch_sample.append(data.type(torch.float))
                elif data is None and len(batch_sample) == 0:
                    break
                elif len(batch_sample) > 0 or data is None:
                    samples = torch.stack(batch_sample).to(device)
                    batch_sample = []

                    optimizer.zero_grad()
                    with torch.no_grad():
                        soft_output_mask = torch.Tensor(soft_output_mask).to(device)
                        [teacher(samples) for teacher in teachers]
                        teacher_soft = torch.cat(tuple([teacher.soft_output for teacher in teachers]), dim=1)
                        # teacher_soft = teacher_soft * soft_output_mask
                        soft_output_mask = []

                    student(samples)
                    student_soft = student.soft_output

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        cross_entropy_loss = torch.nn.functional.kl_div(torch.log_softmax(student_soft, dim=1),
                                                                        torch.softmax(teacher_soft, dim=1))

                    hs, ht, ft_ = cfl_blk(student.features, [teacher.features for teacher in teachers])
                    common_features_loss = 10 * common_feature_learning_criterion(hs, ht, ft_, [teacher.features for teacher in teachers])

                    loss = cross_entropy_loss + common_features_loss
                    loss.backward()
                    optimizer.step()

                    average_tracker.update('loss', loss.item())
                    average_tracker.update('ce', cross_entropy_loss.item())
                    average_tracker.update('cf', common_features_loss.item())

                    description = f'Amalgamating ' \
                                  f'({(cur_step + 1) / (len(all_data)) * 100:.2f}%), '\
                                  f'Loss={average_tracker.get("loss"):.2f} '\
                                  f'(cross entropy={average_tracker.get("ce"):.2f}, '\
                                  f'common features={average_tracker.get("cf"):.2f})'
                    pbar.set_description_str(description)
                    pbar.refresh()
            pbar.update()
            all_data = torch.cat([data for data in data_array])
            p = torch.randperm(len(all_data))
            all_data = all_data[p]
    [teacher.remove_hooks() for teacher in teachers]
    student.remove_hooks()

    student.eval()
    with torch.no_grad():
        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for cur_step, data in enumerate(test.data):
            if data is None:
                break
            elif int(data['y']) not in labels_:
                continue

            label = torch.tensor(labels_.index(data['y'])).to(device)
            sample = data['x'].view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                if label in task_labels:
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0

    return student, [data for d in data_idx for data in d], labels_, accuracy, b_accuracy


class BackboneCrossEntropyLoss(torch.nn.Module):
    def __init__(self, gamma: float = 1.):
        super(BackboneCrossEntropyLoss, self).__init__()
        self.gamma = torch.tensor(gamma).to(device)

    def forward(self, input, target):
        bce = 0
        for idx in range(1, len(input) + 1):
            bce += self.bce(input[:idx], target)
        return bce

    def bce(self, xs, y):
        def bce_right_side(x, y):
            out = torch.log_softmax(x, 1)
            return out[range(x.shape[0]), y]

        def bce_left_side(x, y):
            out = torch.softmax(x, 1)
            return out[range(x.shape[0]), y]

        left_side = torch.sum(torch.stack([bce_left_side(x, y) for x in xs]) ,0)
        out = torch.pow(1 - left_side / len(xs), self.gamma) * bce_right_side(xs[-1], y)
        return - torch.sum(out)/y.shape[0]


def main(args):
    path = f'./state'
    state_path = f'{path}/cafoal_{args.dataset}'
    is_training_base_model = False

    # Prepare data
    train = MyDataLoader(n_tasks=args.n_tasks)
    test = MyDataLoader(n_tasks=args.n_tasks)
    train.load_dataset(args.dataset, True)
    test.load_dataset(args.dataset, False)

    # Training base model
    if args.force_base_retraining is not None and args.force_base_retraining:
        for i in range(args.n_tasks):
            if exists(f'{state_path}_{i + 1}'):
                remove(f'{state_path}_{i + 1}')
    if exists(path):
        for i in range(args.n_tasks):
            if not exists(f'{state_path}_{i + 1}'):
                is_training_base_model = True

    if is_training_base_model:
        print('Training base model')
        for n_task in range(args.n_tasks):
            model = MySimpleModel(len(train.task_info[n_task + 1]['classes_list'])).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            with tqdm(unit='Epoch', total=args.epochs) as pbar:
                while pbar.n < args.epochs:
                    model.train()
                    idx = 0
                    while train.get_x(item=idx, n_task=n_task + 1) is not None:
                        idx, n_elem, data, target = get_batch(idx, n_task, train)
                        optimizer.zero_grad()
                        output = model(data)
                        criterion(output, target).backward()
                        optimizer.step()
                    train.shuffle()

                    save_load_best_model(model, train, n_task, pbar=pbar)
                    pbar.update()

            save_load_best_model(model, test, n_task, False)

    print('Base model performance')
    base_models = []
    for n_task in range(args.n_tasks):
        model = MySimpleModel(len(test.task_info[n_task + 1]['classes_list'])).to(device)
        model, accuracy = save_load_best_model(model, test, n_task, False)
        if n_task == 0:
            accuracy_0 = accuracy
        base_models.append(model)

    accuracies = np.zeros((args.n_tasks, args.n_tasks))
    b_accuracies = np.zeros((args.n_tasks, args.n_tasks))
    accuracies[0, 0] = accuracy_0
    if args.amalgamation_strategy == 'one_at_a_time':
        amalgamated_model, data, labels, accuracy, b_accuracy = amalgamate(teachers=[base_models[0], base_models[1]],
                                                                           data_array=[None, None],
                                                                           labels=[train.task_info[1]['classes_list'],
                                                                                   train.task_info[2]['classes_list']],
                                                                           train=train, test=test, epochs=args.amalgamation_epochs)
        accuracies[0, 1] = accuracy[0]
        accuracies[1, 1] = accuracy[1]
        b_accuracies[0, 1] = b_accuracy[0]
        b_accuracies[1, 1] = b_accuracy[1]
        if args.n_tasks > 2:
            for i in range(2, args.n_tasks):
                amalgamated_model, data, labels, accuracy, b_accuracy = amalgamate(teachers=[amalgamated_model, base_models[i]],
                                                                                   data_array=[data, None],
                                                                                   labels=[labels,
                                                                                           train.task_info[i+1]['classes_list']],
                                                                                   train=train, test=test,
                                                                                   epochs=args.amalgamation_epochs)
                accuracies[i - 1, i] = accuracy[0]
                accuracies[i, i] = accuracy[1]
                b_accuracies[i - 1, i] = b_accuracy[0]
                b_accuracies[i, i] = b_accuracy[1]
    elif args.amalgamation_strategy == 'all_together':
        for n_task in range(args.n_tasks, args.n_tasks + 1, 1):
            _, _, _, accuracy, b_accuracy = amalgamate(teachers=[base_models[idx] for idx in range(n_task)],
                                            data_array=[None] * n_task,
                                            labels=[train.task_info[idx+1]['classes_list'] for idx in range(n_task)],
                                            train=train, test=test,
                                            epochs=int(args.amalgamation_epochs / 10) if n_task < args.n_tasks else args.amalgamation_epochs)
            for i in range(len(accuracy)):
                accuracies[i, n_task - 1] = accuracy[i]
                b_accuracies[i, n_task - 1] = b_accuracy[i]

    print('accuracies')
    print(accuracies)
    print()
    print('b_accuracies (random initialization)')
    print(b_accuracies)
    print()

    acc = np.nanmean(np.where(accuracies != 0, accuracies, np.nan), 0)[-1]
    print(f'ACC: {acc * 100:.2f}%')

    if args.amalgamation_strategy == 'one_at_a_time':
        bwt = 0
        for i in range(args.n_tasks - 1):
            bwt += accuracies[i, i + 1] - accuracies[i, i]
        bwt = bwt / (args.n_tasks - 1)
        print(f'BWT: {bwt * 100:.2f}%')

        fwt = 0
        for i in range(1, args.n_tasks):
            fwt += accuracies[i - 1, i] - b_accuracies[i, i]
        fwt = fwt / (args.n_tasks - 1)
        print(f'FWT: {fwt * 100:.2f}%')
    elif args.amalgamation_strategy == 'all_together':
        bwt = 0
        for i in range(args.n_tasks - 1):
            bwt += accuracies[i, -1] - accuracies[i, i]
        bwt = bwt / (args.n_tasks - 1)
        print(f'BWT: {bwt * 100:.2f}%')

        fwt = 0
        for i in range(1, args.n_tasks):
            fwt += accuracies[i - 1, -1] - b_accuracies[i, i]
        fwt = fwt / (args.n_tasks - 1)
        print(f'FWT: {fwt * 100:.2f}%')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataset == 'omniglot' or args.dataset == 'omniglot10':
        assert args.n_tasks == 50, 'The number of tasks must be set to 50 if you select omniglot dataset'
    if args.dataset == 'cifar100':
        assert args.n_tasks == 10, 'The number of tasks must be set to 10 if you select cifar100 dataset'
    if args.dataset == 'usps' or args.dataset == 'mnist' or args.dataset == 'cifar10':
        assert args.n_tasks == 5, 'The number of tasks must be set to 5 if you select usps/mnist/cifar10 dataset'

    # Configure random seed and devices
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.autograd.set_detect_anomaly = True
    device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    print(f'Device: {device}')

    main(args)

