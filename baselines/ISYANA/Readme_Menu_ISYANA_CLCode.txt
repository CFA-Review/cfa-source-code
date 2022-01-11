###Our work: Accumulating Experiences on The Fly via Inter-Task Synaptic Mapping

## We develop from the source code: https://github.com/GMvandeVen/continual-learning
##The authors for that soruce code are (van de Ven, Gido M and Tolias, Andreas S)

#ISYANA work (ours) is based on the the source code above. 
Please consider citing our papers if you use our code in your research.



Manual for running the code:

##The dataset contains permMNIST, splitMNIST and rotMNIST.

***Use main.py script to get results of ISYANA algorithm.  
"--epoch=N" is the parameter. The "N" can be set by yourself and this parameter means it runs N times with different random seeds each time. The command below means running 10 times.

1. splitMNIST

python main.py "--isyana" "--scenario=task" "--experiment=splitMNIST" "--tasks=5" "--fc-units=256" "--lr=0.01"  "--epoch=10" "--optimizer=sgd"


2. permMNIST

python main.py "--isyana" "--scenario=task" "--experiment=permMNIST" "--tasks=10" "--fc-units=500" "--epsilon=0.1"  "--lr=0.1"  "--epoch=10" "--optimizer=sgd"


3. rotMNIST

python main.py "--isyana"  "--scenario=task"  "--experiment=rotMNIST"  "--tasks=10"  "--fc-units=500"  "--epsilon=0.1"  "--lr=0.1", "--epoch=10"   "--optimizer=sgd"




***Use _compare.py script to get results of different algorithms including ISYANA algorithm. 


1. splitMNIST

python _compare.py  "--c=1" "--scenario=task" "--experiment=splitMNIST" "--tasks=5" "--fc-units=256" "--lr=0.01"  "--epoch=10" "--optimizer=sgd"


2. permMNIST

python _compare.py  "--c=1" "--scenario=task" "--experiment=permMNIST" "--tasks=10" "--fc-units=500" "--epsilon=0.1"  "--lr=0.1"  "--epoch=10" "--optimizer=sgd"



3. rotMNIST

python _compare.py "--c=1"  "--scenario=task"  "--experiment=rotMNIST"  "--tasks=10"  "--fc-units=500"  "--epsilon=0.1"  "--lr=0.1"  "--epoch=10"   "--optimizer=sgd"


4. omniglot 
python _compare.py "--c=1"  "--scenario=task"  "--experiment=omniglot"  "--tasks=50"  "--fc-units=500"  "--epsilon=0.1"  "--lr=0.1"  "--epoch=10"   "--optimizer=sgd"  "--batch 32"  





python main.py "--isyana" "--scenario=task" "--experiment=splitMNIST" "--tasks=5" "--fc-units=256" "--lr=0.01"  "--epoch=1" "--optimizer=sgd"


agem



1. splitMNIST

python main.py  "--replay=exemplars"  "--agem"  "--budget=2000"  "--c=1"  "--scenario=task"  "--experiment=splitMNIST"   "--tasks=5"  "--fc-units=256"    "--lr=0.01" "--epoch=10"  "--optimizer=sgd"





2. permMNIST

python main.py  "--replay=exemplars"  "--agem"  "--budget=2000" "--c=1" "--scenario=task" "--experiment=permMNIST" "--tasks=10" "--fc-units=500" "--epsilon=0.1"  "--lr=0.1"  "--epoch=10" "--optimizer=sgd"



3. rotMNIST

python main.py  "--replay=exemplars"  "--agem"  "--budget=2000"  "--c=1"  "--scenario=task"  "--experiment=rotMNIST"  "--tasks=10"  "--fc-units=500"  "--epsilon=0.1"  "--lr=0.1"  "--epoch=10"   "--optimizer=sgd"


python main.py --scenario task --experiment omniglot --tasks 50 --c 1 --fc-units 500 --seed 0 --lr 0.1 --optimizer sgd --epsilon 0.1 --batch 32 
