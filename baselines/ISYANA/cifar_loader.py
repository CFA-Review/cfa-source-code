from data import get_multitask_experiment
import utils
from base_model import MySimpleModel,MyDataLoader
import torchvision
import torch
from encoder import fc_layer

classifier = fc_layer(int(512/2), 10, excit_buffer=True, nl='none', drop=0)
for i in range(2):
    net = MySimpleModel(2)
    path = f'./CIFAR10'
    state_path = f'{path}/cafoal_cifar10_{i+1}'
    net.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))

    for name, param in net.fc.named_parameters():
        if '3.weight' in name:
            classifier.linear.weight.data[i*2:(i+1)*2,:] = param.data
        if '3.bias' in name:
            classifier.linear.bias.data[i*2:(i+1)*2] = param.data

torch.save(classifier.state_dict(),'./CIFAR10/cifar10_classifier')

net = torch.nn.Sequential(
    torchvision.models.resnet18(pretrained=True),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, int(512/2)),
    torch.nn.ReLU(),
    torch.nn.Linear(int(512/2), 10)
)
for i in range(1,3):
    print(i)
for n_task in range(5):
    path = f'./CIFAR10'
    state_path = f'{path}/cafoal_cifar10_{n_task + 1}'
    print(torch.load(state_path, map_location=torch.device('cpu')))
    model = MySimpleModel(2)
    model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))
    for name, param in model.named_parameters():
        print(name)
        print(param.shape)
    print(model)

# dataset = MyDataLoader.load_dataset(dataset='cifar10',train=True)
# model = MySimpleModel
# model, acc =  save_load_best_model(dataset,model)

(train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
    name='splitMNIST', scenario='task', tasks=5, data_dir='./datasets',
    verbose=True, exception=True,)

tmp = iter(utils.get_data_loader(train_datasets, 32, drop_last=True))

for batch_idx, (img, labels) in enumerate(tmp): # change dataloader Data of Task1 ***Revised
    print(img.shape)
    print(labels)

(train_datasets, test_datasets), config, classes_per_task = get_cifar10_experiment(
    scenario='task', tasks=5, size=32, verbose=False)

print(train_datasets)
for i,x in enumerate(train_datasets):
    print(x.shape)
    print(y)

print('Base model performance')
base_models = []
for n_task in range(args.n_tasks):
    model = MySimpleModel(len(test.task_info[n_task + 1]['classes_list'])).to(device)
    model, accuracy = save_load_best_model(model, test, n_task, False)
    if n_task == 0:
        accuracy_0 = accuracy
        base_models.append(model)
        for n_task in range(2, args.n_tasks + 1, 1):
            _, _, _, accuracy, b_accuracy = amalgamate(teachers=[base_models[idx] for idx in range(n_task)],
                                                       data_array=[None] * n_task,
                                                       labels=[train.task_info[idx+1]['classes_list'] for idx in range(n_task)],
                                                       train=train, test=test,
                                                       epochs=int(args.amalgamation_epochs / 10) if n_task < args.n_tasks else args.amalgamation_epochs)
            for i in range(len(accuracy)):
                accuracies[i, n_task - 1] = accuracy[i]
                b_accuracies[i, n_task - 1] = b_accuracy[i]