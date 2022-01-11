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

parser = argparse.ArgumentParser('./cafoal.py',
                                 description='Tackling catastrophic forgetting with knowledge amalgamation')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use',
                    choices=['usps', 'mnist', 'cifar10', 'cifar10', 'omniglot', 'omniglot10'])
parser.add_argument('--seed', type=int, default=None, metavar='N', help='Set a seed to compare runs')
parser.add_argument('--n_tasks', type=int, default=5, metavar='N', help='Number of tasks',
                    choices=[5, 10, 50])
parser.add_argument('--cuda', action='store_true', help='enable CUDA')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--cfl_lr', type=float, default=None, help='Common feature amalgamation learning rate')
parser.add_argument('--batch_size', type=int, default=2 ** 5, help='Batch size for base model training',
                    choices=[2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8])
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per task (base learning)')
parser.add_argument('--force_base_retraining', type=bool, default=False, help='Force base model retraining')
parser.add_argument('--amalgamation_strategy', type=str, default='all_together', help='Amalgamation Strategy',
                    choices=['all_together', 'one_at_a_time'])
parser.add_argument('--amalgamation_epochs', type=int, default=1000, help='Amalgamation epochs',
                    choices=[10, 100, 500, 1000])
parser.add_argument('--memory_size', type=int, default=500, help='Size of sample memory',
                    choices=[100, 200, 500])
parser.add_argument('--n_omniglot_augmentations', type=int, default=0,
                    help='Number of augmentation loops for omniglot dataset', choices=[0, 10])
parser.add_argument('--omniglot_test_ratio', type=float, default=0.3, help='Ratio of train to test in omniglot')


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


class CIFARDataLoader(CustomDataLoader):
    def __init__(self, dataset, transforms: T = None, n_tasks: int = 1):
        super(CIFARDataLoader, self).__init__(dataset, transforms, n_tasks)


class MyDataLoader:
    class CIFAR10(CIFARDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transforms)
            CIFARDataLoader.__init__(self, dataset, transforms, n_tasks)

    class CIFAR100(CIFARDataLoader):
        def __init__(self, transforms: T = None, train: bool = True, n_tasks: int = 1):
            dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transforms)
            CIFARDataLoader.__init__(self, dataset, transforms, n_tasks)

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

    def __load_cifar10(self, transforms: T = None, train: bool = True):
        self.data = self.CIFAR10(transforms, train, self.n_tasks)

    def __load_cifar100(self, transforms: T = None, train: bool = True):
        self.data = self.CIFAR100(transforms, train, self.n_tasks)

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


if __name__ == '__main__':
    args = parser.parse_args()

    # Configure random seed and devices
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.autograd.set_detect_anomaly = True
    device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    print(f'Device: {device}')
    dataset = MyDataLoader(5)
    dataset.load_dataset('cifar10')
    dataset.get_x(1,1)
    main(args)

