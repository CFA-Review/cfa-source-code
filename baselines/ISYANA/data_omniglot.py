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

# ----------------------------------------------------------------------------------------------------------#

alphabets = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)',
             'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)',
             'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew',
             'Inuktitut_(Canadian_Aboriginal_Syllabics)',
             'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)',
             'Mkhedruli_(Georgian)',
             'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog',
             'Tifinagh', 'Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic',
             'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)',
             'Oriya',
             'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']

# specify available data-sets.
AVAILABLE_DATASETS = {
    'omniglot': datasets.Omniglot,
}

# # specify available transforms.
# AVAILABLE_TRANSFORMS = {
#     'omniglot': [
#         Resize(28),transforms.ToTensor(),
#     ],
# }

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'omniglot': {'size': 28, 'channels': 1, 'classes': 10},
}

# data augment parameters
np.random.seed(1)
rot = list(np.random.choice(np.arange(1, 20), 10, replace=False))
x_cor = np.random.rand(10)
y_cor = np.random.rand(10)
# save transformation vars
# print('Rot:{}\nCoordinate\nx\n{}\ny\n{}'.format(rot, x_cor, y_cor))
np.savetxt('Transformation', (rot, x_cor, y_cor))


# ----------------------------------------------------------------------------------------------------------#


def rotate_img(image, rotation):
    '''rotate the pixels of an image according to [rotation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if rotation is None:
        return image
    else:
        trans = transforms.ToPILImage()
        img = trans(image)
        img = TF.rotate(img,rotation,fill=(0,))
        trans1 = transforms.ToTensor()
        # return trans1(img)
        return img


# ----------------------------------------------------------------------------------------------------------#


def get_dataset(name, type='train', alphabet_name=None, capacity=None,
                verbose=False, target_transform=None,
                rot=None, x_cor=None, y_cor=None, task_id=0, classes_per_task=10):
    # load data-set
    dataset = [OmniglotOneAlphabetDataset(alphabet_name=alphabet_name, train=False if type == 'test' else True,
                                          task_id=task_id, classes_per_task=classes_per_task)]

    # data augment: 20 transforms
    for r in rot:
        dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: rotate_img(x, r))])
        dataset.append(OmniglotOneAlphabetDataset(alphabet_name=alphabet_name, train=False if type == 'test' else True,
                                                  transform=dataset_transform, task_id=task_id,
                                                  classes_per_task=classes_per_task
                                                  ))
    for cor in range(10):
        transform = RandomAffine(0, translate=(x_cor[cor], y_cor[cor]))
        dataset.append(OmniglotOneAlphabetDataset(alphabet_name=alphabet_name, train=False if type == 'test' else True,
                                                  transform=transform, task_id=task_id,
                                                  classes_per_task=classes_per_task
                                                  ))
    dataset_ = torch.utils.data.ConcatDataset(dataset)
    # dataset_ = torch.utils.data.DataLoader(dataset_,batch_size=32,
    #                                shuffle=True,num_workers=0,
    #                                pin_memory=True,drop_last=False)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset_)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset_) < capacity:
        dataset_copy = copy.deepcopy(dataset_)
        dataset_ = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset_


# ----------------------------------------------------------------------------------------------------------#

class OmniglotOneAlphabetDataset(datasets.Omniglot):
    alphabets = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)',
                 'Balinese',
                 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic',
                 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew',
                 'Inuktitut_(Canadian_Aboriginal_Syllabics)',
                 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)',
                 'Mkhedruli_(Georgian)',
                 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog',
                 'Tifinagh',
                 'Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic',
                 'Gurmukhi',
                 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya',
                 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']

    def __init__(
            self, alphabet_name, train=True,
            data_root="./data2/", test_ratio=0.3, target_type='int',
            transform=None, target_transform=None, task_id=0, classes_per_task=10):

        assert 0. < test_ratio and test_ratio < 1., 'test ration must be in [0, 1]'
        assert alphabet_name in self.alphabets, "alphabet '{}' not presented".format(alphabet_name)
        self.alphabet_name = alphabet_name
        background = False
        if self.alphabets.index(alphabet_name) < 30:
            background = True
        transforms_list = []
        if train and transform is not None:
            transforms_list.append(transform)
        transforms_list.extend([
            Resize((28, 28)),  # see https://arxiv.org/pdf/1606.04080.pdf
            ToTensor()])
        transform = Compose(transforms_list)
        super().__init__(
            data_root, background=background,
            transform=transform, download=False, target_transform=target_transform)
        self.alph_index = self._alphabets.index(self.alphabet_name)

        alphs_counts = [sum(
            [len(list_files(join(self.target_folder, a, c), '.png')) for c in list_dir(join(self.target_folder, a))])
            for a in self._alphabets]

        n_char_samples = 20
        self.task_id = task_id
        self.classes_per_task = classes_per_task

        self.alph_counts = alphs_counts[self.alph_index]
        assert (self.alph_counts % n_char_samples == 0)

        self.alph_start = np.cumsum(alphs_counts)[self.alph_index] - self.alph_counts
        char_indices = np.arange(self.alph_start, self.alph_start + self.alph_counts)
        train_ratio = 1. - test_ratio

        if train:
            indices = char_indices.reshape(-1, n_char_samples)[:, :int(n_char_samples * train_ratio)]
        else:
            indices = char_indices.reshape(-1, n_char_samples)[:, int(n_char_samples * train_ratio):]
        classes = np.cumsum(
            np.zeros_like(indices).astype(target_type) + 1, axis=0) - 1

        self.indices = indices.flatten()
        self.classes = classes.flatten()
        self.n_classes = self.alph_counts // n_char_samples

        if train and len(self.indices) == 0:
            raise Exception('Train dataset is empty, consider using smaller test_ratio')
        if train and len(self.indices) == self.alph_counts:
            warnings.warn("Train dataset takes all available data")
        if not train and len(self.indices) == self.alph_counts:
            warnings.warn("Test dataset takes all available data")
        if not train and len(self.indices) == 0:
            raise Exception('Test dataset is empty, consider using greater test_ratio')
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        im, target = super().__getitem__(self.indices[i])
        return torch.squeeze(im), torch.tensor(self.classes[i] + self.task_id * self.classes_per_task,dtype=torch.long)


# ----------------------------------------------------------------------------------------------------------#


def get_omni_experiment(name, scenario, tasks, only_config=False, verbose=False,
                        exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'omniglot':
        # check for number of tasks
        if tasks > 50:
            raise ValueError("Experiment 'omniglot' cannot have more than 50 tasks!")
        # configurations
        config = DATASET_CONFIGS['omniglot']
        classes_per_task = 10
        if not only_config:
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, alp in enumerate(alphabets):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x * classes_per_task
                ) if scenario in ('task', 'class') else None
                # prepare train and test datasets with all classes
                train_datasets.append(get_dataset('omniglot', type="train", alphabet_name=alp,
                                                  verbose=verbose, target_transform=target_transform,
                                                  rot=rot, x_cor=x_cor, y_cor=y_cor,
                                                  task_id=task_id, classes_per_task=classes_per_task
                                                  ))
                test_datasets.append(get_dataset('omniglot', type="test", alphabet_name=alp,
                                                 verbose=verbose, target_transform=target_transform,
                                                 rot=rot, x_cor=x_cor, y_cor=y_cor,
                                                 task_id=task_id, classes_per_task=classes_per_task
                                                 ))


    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario == 'domain' else classes_per_task * tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)