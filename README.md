# cfa-source-code
CFA

Setup the environment and run

## Setting up a CONDA environment


Execute line by line

```
conda create -n CFA python=3.8
conda activate CFA
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install tqdm matplotlib
pip install avalanche-lib
```

## Setting up a PIP environment

Execute line by line

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
pip install matplotlib
pip install avalanche-lib
```

## Running


For a list of commands:
```
python cfa.py --help
```

For MNIST
```
python cfa.py --dataset mnist --memory_budget 1000 --memory_strategy fixed
python cfa.py --dataset mnist --memory_budget 1000 --memory_strategy grow
```

For CIFAR10
```
python cfa.py --dataset cifar10 --memory_budget 1000 --memory_strategy fixed
python cfa.py --dataset cifar10 --memory_budget 1000 --memory_strategy grow
```

For CIFAR100

```
python cfa.py --dataset cifar100 --memory_budget 1000 --memory_strategy fixed
python cfa.py --dataset cifar100 --memory_budget 1000 --memory_strategy grow
```

For Tiny ImageNet

```
python cfa.py --dataset tiny10 --memory_budget 1000 --memory_strategy fixed
python cfa.py --dataset tiny10 --memory_budget 1000 --memory_strategy grow
```

## Tip

If you are not intersted in evaluating the BWT and FWT metrics, just the ACC, modify the line 721 from:

```python
        for n_task in range(2, n_tasks + 1, 1):
```

to
```python
        for n_task in range(n_tasks, n_tasks + 1, 1):
```

In order to calculate BWT and FWT, we need to run multiple CFA experiments, which can be time-consuming. By making this change, you force the algorithm to just run a full amalgamation of all teachers. This will give you the ACC metric, but BWT and FWT will not be valid.