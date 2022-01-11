import copy
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets.utils import list_dir, list_files
import numpy as np
import warnings
from os.path import join
from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, RandomAffine, RandomRotation
import torch
import torchvision.transforms.functional as TF
from encoder import MySimpleModel

# ----------------------------------------------------------------------------------------------------------#

def load_saved_model(model, path, state_path, classes_per_task,dataset,device):

    net = MySimpleModel(classes_per_task)
    net.load_state_dict(torch.load(state_path, map_location=device))
    model.fcE.resnet = copy.deepcopy(net.resnet)

    return model


# ----------------------------------------------------------------------------------------------------------#


class MyRotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)

class MyTransposeTransform:
    def __call__(self, x):
        return x.transpose(0,0)

# specify available data-sets.
AVAILABLE_DATASETS = {
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'cifar': {'size': 32, 'channels': 1, 'classes': 10},
}
# ----------------------------------------------------------------------------------------------------------#

def get_dataset(name, type='train', download=True, capacity=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'cifar'
    dataset_class = AVAILABLE_DATASETS[name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        MyTransposeTransform(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        MyRotationTransform(-90),
        transforms.ToTensor()
    ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

# ----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if original_dataset.train:
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])

            if label in sub_labels:
                self.sub_indeces.append(index)    ## only add the index from the input dataset that its label in sub_labels
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

# ----------------------------------------------------------------------------------------------------------#


def get_cifar10_experiment(scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                        exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if tasks > 5:
        raise ValueError("Experiment 'CIFAR-10' cannot have more than 5 tasks!")
    # configurations
    config = DATASET_CONFIGS['cifar']
    classes_per_task = int(np.floor(10 / tasks))
    if not only_config:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
        target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
        # prepare train and test datasets with all classes
        cifar10_train = get_dataset('cifar10', type="train", dir=data_dir, target_transform=target_transform,
                                  verbose=verbose)
        cifar10_test = get_dataset('cifar10', type="test", dir=data_dir, target_transform=target_transform,
                                 verbose=verbose)

        # generate labels-per-task
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
        ]
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if scenario=='domain' else None
            train_datasets.append(SubDataset(cifar10_train, labels, target_transform=target_transform))
            test_datasets.append(SubDataset(cifar10_test, labels, target_transform=target_transform))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario == 'domain' else classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


def get_cifar100_experiment(scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                           exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if tasks > 10:
        raise ValueError("Experiment 'CIFAR-100' cannot have more than 10 tasks!")
    # configurations
    config = DATASET_CONFIGS['cifar']
    classes_per_task = int(np.floor(100 / tasks))
    if not only_config:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        permutation = np.array(list(range(100))) if exception else np.random.permutation(list(range(100)))
        target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
        # prepare train and test datasets with all classes
        cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, target_transform=target_transform,
                                    verbose=verbose)
        cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, target_transform=target_transform,
                                   verbose=verbose)

        # generate labels-per-task
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
        ]
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if scenario=='domain' else None
            train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
            test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario == 'domain' else classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)