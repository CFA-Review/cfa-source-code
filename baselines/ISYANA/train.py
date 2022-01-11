import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
import time
from data import SubDataset, ExemplarDataset
from continual_learner import ContinualLearner

from torch import nn
from deep_clustering_network import AverageMeter, autoencoder_mnist, autoencoder_cifar, cluster
from deep_clustering_network import clusteringProcess, collectClusterList
from deep_clustering_network import taskToClassDistance, interTaskMerging
from deep_clustering_network import getClassCenterOfFirstTask
from torch.autograd import Variable
from torch.utils.data import DataLoader
import evaluate
import argparse
from data_cifar import load_saved_model

###iternumIntrainCL = 2000   ## **Revised set in line 14
##***Revised
def train_cl(model, train_datasets, test_datasets, args=None, param_stamp=None, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,  
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, eval_cbs_exemplars=list(), device='cpu'):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''


    ##Random model on the Each testing data of each task
    initialmodeAccuracyPrecVec = None   
    siiPrecVec = []
    sminusiiPrecVec = []  #si-1i
    accuracySTiPrecVec = []
    backwardTransferSTiPrecVec = []
    forwardTransferSTiPrecVec  = []

    # load best cifar model from task 1
    if args.experiment == 'cifar10':
        path = f'/home/weng/ISYANA-cifar/ISYANA-cifar/CIFAR10'
        state_path = f'{path}/cafoal_cifar10_1'
        model = load_saved_model(model, path, state_path,classes_per_task,args.experiment,device=device)
    elif args.experiment == 'cifar100':
        path = f'/home/weng/ISYANA-cifar/ISYANA-cifar//CIFAR100'
        state_path = f'{path}/cafoal_cifar100_1'
        model = load_saved_model(model, path, state_path,classes_per_task,args.experiment,device=device)

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    ##get the values under the random model    (Random model on Each testing data of each task)
    print("\n\n--> Random Initial model Evaluation - (task-incremental learning scenario):")

    ##The total number of tasks
    tasktotalnum = round(model.classes/classes_per_task)
    # Evaluate precision of random initial model on full test-set (testing data of each task)
    initialmodeAccuracyPrecVec = [evaluate.validate(
        copy.deepcopy(model), copy.deepcopy(test_datasets[i]), verbose=False, test_size=None, task=i+1, with_exemplars=False,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1)))
    ) for i in range(1, tasktotalnum)]
    print("\n Initial model Precision on test-set (softmax classification):")
    for i in range(tasktotalnum-1):
        print(" - Task {}: {:.4f}".format(i + 1, initialmodeAccuracyPrecVec[i]))



    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())


    ##If it is the inter-task synaptic mapping flow
    ##create the AutoEncode for mapping the input data to latent space L
    if args.isyana:
        ##For AutoEncoder
        # initiate network
        autoencoderModel = autoencoder_mnist()
        if model.resnet18:
            autoencoder = autoencoder_cifar()

        ##device = 'cuda'
        autoencoderModel.to(device)

        criterion = nn.MSELoss()
        autoencoderoptimizer = torch.optim.Adam(
        autoencoderModel.parameters(), lr=0.01, weight_decay=1e-5)

        num_epochs = 4   
        nClass = model.classes  ##total number of class/label

        ##in the fist task, these items for creating initial cluster. And then used for creating cluster for each task
        bakeinitialCenter = None
        bakeinitialLabels = None
        bakeinitialCounter = None

        ## record the node importance to Each class. 
        ##node important to class (only previous task)
        prevlayer1nodeImportantoClassMatrix = torch.Tensor().float().to(device) 
        prevlayer2nodeImportantoClassMatrix = torch.Tensor().float().to(device)    

        ##current node important to class (current task and + previous task) matrix
        layer1nodeImportantoClassMatrix = torch.Tensor().float().to(device) 
        layer2nodeImportantoClassMatrix = torch.Tensor().float().to(device)    

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):  ### Task index starts from 1

        if(3 == task):
            print("bug comes out task %d\n", task)

        # # load model for cifar
        # if task>1:
        #     if args.experiment == 'cifar10':
        #         path = f'./CIFAR10'
        #         state_path = f'{path}/cafoal_cifar10_{task}'
        #         model = load_saved_model(model, state_path)
        #     elif args.experiment == 'cifar100':
        #         path = f'./CIFAR10'
        #         state_path = f'{path}/cafoal_cifar10_{task}'
        #         model = load_saved_model(model, state_path)

        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task>1:
            # ---------- ADHOC SOLUTION: permMNIST needs transform to tensor, while splitMNIST does not ---------- #
            if len(train_datasets)>6:
                target_transform = (lambda y, x=classes_per_task: torch.tensor(y%x)) if (
                        scenario=="domain"
                ) else (lambda y: torch.tensor(y))
            else:
                target_transform = (lambda y, x=classes_per_task: y%x) if scenario=="domain" else None
            # ---------------------------------------------------------------------------------------------------- #
            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        ###**Revised
        outputlayer1 = None
        outputlayer2 = None
        y_hat = None   ##self(x)

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            active_classes = list(range(classes_per_task * task))

        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task


        # Loop over all iterations
        ###do not use the iters. just use the length of batches for each Task ***Revised
        tmpdata_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
        #print("\ntmpdata_loader length", len(tmpdata_loader))
        iters = len(tmpdata_loader)
        gen_iters = iters
        iters_to_use = iters

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))


        ##Initial a good network of AutoEncoder, and create a inital cluster  
        ##for calculate TT setting
        # print("\nTask ", task)  ##**Revised

        ##Inter-task synaptic mapping flow
        if args.isyana:
            ##only for the first task, (before creating cluster) need to warm up to train and build a good network. For other tasks, no need
            if 1 == task:
                warmUp = 2
                myClusterPerTask = None  ##initial
            else:
                warmUp = 0    


            if 1 == task:
                ##only task 1 needs to Train AutoEncoder Network in advance. 
                ##Each task needs to create an initial cluster. Currently, the initial cluster is the same 
                for epoch in range(num_epochs):  ##iteration for each Task  . Use the Task 1 dataset to train the network            
                    tmpdata_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))                
                    isFinishInitCluster = 0 ###
                    for batch_idx, (img, labels) in enumerate(tmpdata_loader): # change dataloader Data of Task1 ***Revised
                        # Set the gradients to zeros
                        autoencoderoptimizer.zero_grad()

                        if model.resnet18:
                            img = model.resnet(img.to(device))

                        img = img.view(img.size(0), -1)
                        img = Variable(img).cpu()
                        img.requires_grad_()
                        img = img.to(device)

                        # ===================forward=====================
                        output, _ = autoencoderModel(img)

                        # ==================Create an Initial clustering===================
                        if epoch == warmUp and batch_idx == 0:  ##only first task,need to prepare initial cluster
                            totaltask = round(nClass/classes_per_task)
                            tmptotaltraindataset = ConcatDataset(train_datasets[:totaltask])   
                            #print(len(tmptotaltraindataset))                     
                            tmpdataloader = DataLoader(tmptotaltraindataset, len(tmptotaltraindataset), shuffle=False)  ##all the data. Just need labels
                            #print("total data number ", len(tmpdataloader))
                            for tmpbatch_idx, (tmpimg, tmplabels) in enumerate(tmpdataloader):   
                                with torch.no_grad():
                                    tmpimg = tmpimg.view(tmpimg.size(0), -1)
                                    tmpimg = Variable(tmpimg).cpu()
                                    tmpimg.requires_grad_()
                                    _, tmphr = autoencoderModel(tmpimg)
                                
                                # calculate the value of initial cluster
                                initialIndex = torch.Tensor().long()

                                for iClass in range(0,nClass):   ##make sure. tmplabels contains all the class 
                                    selectedLabel = torch.eq(tmplabels,iClass)
                                    selectedIndex = torch.nonzero(selectedLabel)[0]
                                    initialIndex = torch.cat((initialIndex,selectedIndex),0)

                                initialCenter = tmphr.index_select(0,initialIndex.to(device)).detach()
                                initialLabels = tmplabels.index_select(0,initialIndex).detach().to(device)

                                ##bake these three item. For other tasks creating their intiial clusters respectively
                                bakeinitialCenter = copy.deepcopy(initialCenter)
                                bakeinitialLabels = copy.deepcopy(initialLabels)
                                break
                            

                            # initiate cluster for every task
                            myClusterPerTask = cluster(initialCenter, nClass, initialLabels, task, classes_per_task)
                            isFinishInitCluster = 1 ##flag. finish creating initial cluster
                    
                        if epoch <= warmUp:
                            # forward network
                            finalLoss = criterion(output, img.detach())
                            reconsLossPrint = finalLoss.detach().item()                   

                        # ===================backward====================
                        finalLoss.backward()
                        autoencoderoptimizer.step()
                        if 1 == isFinishInitCluster:  ##already created initial cluster. Done
                            break    

                    # ===================log========================
                    #print('Epoch: {}'.format(epoch))
                    #print('Reconstruction Loss: {}'.format(reconsLossPrint))
                    if epoch == warmUp:
                        break   ##No need to run
            else:
                # initiate cluster for every task
                myClusterPerTask = cluster(bakeinitialCenter, nClass, bakeinitialLabels, task, classes_per_task)            

            ###get current task's all labels
            tmpindexbegin = classes_per_task * (task -1)
            tmpindexend = classes_per_task * task 
            tmpsegmentindex = torch.arange(tmpindexbegin,tmpindexend)
            currentTaskLabels = bakeinitialLabels.index_select(0,tmpsegmentindex.to(device)).detach()               

            #print("Finish warming up AutoEncode and creating an initial cluster")
            #print("Entering Main loop (all Tasks)")

        for batch_index in range(1, iters_to_use+1):
            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                #print("len dataloader ", len(data_loader))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact:
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                x, y = next(data_loader)                                    #--> sample training data of current task
                

                ##Inter-task synaptic mapping flow
                if args.isyana:
                    ##copy x and y for calculating TT **Revised
                    imgx = copy.deepcopy(x)  #** For TT
                    labely = copy.deepcopy(y)


                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                
                if 'cifar' in args.experiment:
                    with torch.no_grad():
                        x = model.resnet(x)
                        
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = (previous_model(x))[2][:, :(classes_per_task * (task - 1))]
                else:
                    scores = None


                ##Inter-task synaptic mapping flow
                if args.isyana:
                ###For calculating TT. 
                    #xxxfor batch_idx, (img, labels) in enumerate(dataloader): # change dataloader
                    # Set the gradients to zeros
                    autoencoderoptimizer.zero_grad()

                    imgx = imgx.view(imgx.size(0), -1)
                    imgx = Variable(imgx).cpu()
                    imgx.requires_grad_()

                    # ===================forward=====================
                    output, hr = autoencoderModel(imgx)

                    # forward network and conduct clustering

                    # forward network
                    reconsLoss = criterion(output, imgx.detach())
                    finalLoss = reconsLoss   

                    reconsLossPrint = reconsLoss.detach().item()
                    
                    ##
                    if task == 1:   ##Task index starts from 1
                        tmpclassClusterList = collectClusterList(copy.deepcopy(myClusterPerTask), nClass)

                        # calculate TT
                        taskToClassDistanceTemp = taskToClassDistance(myClusterPerTask,myClusterPerTask,hr,labely,nClass,tmpclassClusterList,currentTaskLabels,device,task)
                    else:
                        tmpclassClusterList = collectClusterList(globalCluster, nClass)
                        # calculate TT
                        taskToClassDistanceTemp = taskToClassDistance(globalCluster,myClusterPerTask,hr,labely,nClass,tmpclassClusterList,currentTaskLabels,device,task)
                        
                    # print("\nTask to Class Distance: TT. Size ", len(taskToClassDistanceTemp))
                    # print(taskToClassDistanceTemp)

                    # ===================backward====================
                    finalLoss.backward()
                    autoencoderoptimizer.step()
                    

            # ===================log========================
            #print('Current Task: {}'.format(task))
            #print('Batch_index: {}'.format(batch_index))            
            #print('Reconstruction Loss: {}'.format(reconsLossPrint))
                

            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)[2]
                        scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        if 'cifar' in args.experiment:
                            with torch.no_grad():
                                x_temp = model.resnet(x_temp.to(device))
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])[2]
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)[2]
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)[2]
                        if scenario=="domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None


            #---> Train MAIN MODEL
            if batch_index <= iters:

                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, args=args, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)

                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c>0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

               # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)



                ###Inter-Task Synaptic Mapping Flow
                if args.isyana:
                    #### **Revised  After each batch. update
                    outputlayer1, outputlayer2, y_hat = copy.deepcopy(model)(x)

                    # -if needed, remove predictions for classes not in current task
                    if active_classes is not None:
                        class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                        y_hat = y_hat[:, class_entries]                


                    # ISYANA: calculate and update the normalized path integral
                    if isinstance(model, ContinualLearner) and (model.si_c>0):
                        ##ISYANA method
                        flagSetSIPreTaskValue = 0 ##false         
                        if batch_index == iters: 
                            flagSetSIPreTaskValue = 1 ##Need to set the previous task value (record previous task's parameters) 
                        
                        layer1nodeImportantoClassMatrix, layer2nodeImportantoClassMatrix = model.isyanaUpdate_omega(W, model.epsilon, outputlayer1, outputlayer2, y_hat, task, classes_per_task, taskToClassDistanceTemp, prevlayer1nodeImportantoClassMatrix, prevlayer2nodeImportantoClassMatrix, flagSetSIPreTaskValue)
                    
                        ##already it is the last batch of this task. record the node importance to class matrix
                        if batch_index == iters: 
                            prevlayer1nodeImportantoClassMatrix = copy.deepcopy(layer1nodeImportantoClassMatrix)
                            prevlayer2nodeImportantoClassMatrix = copy.deepcopy(layer2nodeImportantoClassMatrix)
            
            
            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1./task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)



        ##After each task
        if args.isyana:
        ##after finishing the current Task. Then store  
        #print("\nThe number of clusters in Current Cluster", myClusterPerTask.noOfCluster)   
            # create global cluster
            if task == 1:
                ###For KL divergence.              
                kltmpdataloader = DataLoader(training_dataset, len(training_dataset), shuffle=False)  ##all the data. Just need labels
                #print("total data number ", len(kltmpdataloader))
                for kltmpbatch_idx, (kltmpimg, kltmplabels) in enumerate(kltmpdataloader):   
                    with torch.no_grad():
                        kltmpimg = kltmpimg.view(kltmpimg.size(0), -1)
                        kltmpimg = Variable(kltmpimg).cpu()
                        kltmpimg.requires_grad_()
                        _, kltmphr = autoencoderModel(kltmpimg)           

                getClassCenterOfFirstTask(myClusterPerTask,kltmphr,kltmplabels,currentTaskLabels,device)  

                globalCluster = copy.deepcopy(myClusterPerTask)
            else:
                # Combine cluster. Just only add clusters of each task to global clusters (No need merge)
                globalCluster = interTaskMerging(globalCluster,myClusterPerTask)           
            #print("\nThe number of clusters in globalCluster", globalCluster.noOfCluster)



        #print("Current model evaluates current task and next task")
        # Evaluate precision ---  current model on current task
        ### Notice. in this function. task index starts from 1. so need to notice.
        if task < tasktotalnum:
            tmpsiipre = evaluate.validate(
                model, test_datasets[task-1], verbose=False, test_size=None, task=task, with_exemplars=False,
                allowed_classes=list(range(classes_per_task*(task-1), classes_per_task*(task)))
            ) 
            siiPrecVec.append(tmpsiipre)

        # Evaluate precision -- current model on next task
            tmpsminusii = evaluate.validate(
                model, test_datasets[task], verbose=False, test_size=None, task=task+1, with_exemplars=False,
                allowed_classes=list(range(classes_per_task*task, classes_per_task*(task+1)))
            ) 
            sminusiiPrecVec.append(tmpsminusii)


        ##----------> UPON FINISHING EACH TASK...
        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario=="task" else (list(range(classes_per_task*task)) if scenario=="class" else None)
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(training_dataset, allowed_classes=allowed_classes, resnet=model.resnet if model.resnet18 else None)

        # # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            ##original method
            if args.si:   ##SI flow
                model.update_omega(W, model.epsilon)
                    

        # EXEMPLARS: update exemplar sets
        if (add_exemplars or use_exemplars) or replay_mode=="exemplars":
            exemplars_per_class = int(np.floor(model.memory_budget / (classes_per_task*task)))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(classes_per_task)) if scenario=="domain" else list(range(classes_per_task*(task-1),
                                                                                              classes_per_task*task))
            for class_id in new_classes:
                start = time.time()
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
                print("Constructed exemplar-set for class {}: {} seconds".format(class_id, round(time.time()-start)))
            model.compute_means = True
            # evaluate this way of classifying on test set
            for eval_cb in eval_cbs_exemplars:
                if eval_cb is not None:
                    eval_cb(model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]




    print("\n\n--> Final model Evaluation - (task-incremental learning scenario):")
    # Evaluate precision of final model on Each Task -- full test-set
    accuracySTiPrecVec = [evaluate.validate(
        copy.deepcopy(model), copy.deepcopy(test_datasets[i]), verbose=False, test_size=None, task=i+1, with_exemplars=False,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1)))
    ) for i in range(tasktotalnum)]
    print("\n Precision on test-set (softmax classification):")
    for i in range(tasktotalnum):
        print(" - Task {}: {:.6f}".format(i + 1, accuracySTiPrecVec[i]))
    avgAccuracySTiPrec = sum(accuracySTiPrecVec) / tasktotalnum
    # print('=> Final model-average precision over all {} tasks: {:.4f}'.format(tasktotalnum, avgAccuracySTiPrec))

    ## Backward Transfer Prec  And Foward Transfer Prec
    for i in range(tasktotalnum-1):
        tmpbackwardtransfer = accuracySTiPrecVec[i] - siiPrecVec[i]
        backwardTransferSTiPrecVec.append(tmpbackwardtransfer)


        tmpforwardtransfer = sminusiiPrecVec[i] - initialmodeAccuracyPrecVec[i]
        forwardTransferSTiPrecVec.append(tmpforwardtransfer)
    
    backwardTransfer = sum(backwardTransferSTiPrecVec) / (tasktotalnum - 1)
    forwardTransfer = sum(forwardTransferSTiPrecVec) / (tasktotalnum - 1)

    print("Average accuracy") 
    print(avgAccuracySTiPrec)
    print("backwardTransfer")
    print(backwardTransfer)
    print("forwardTransfer")
    print(forwardTransfer)


    # save test results to file
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'a')
    output_file.write('AverageAccuracy:\n{}\n'.format(avgAccuracySTiPrec))
    output_file.write('ForwardTransfer:\n{}\n'.format(forwardTransfer))
    output_file.write('BackwardTransfer:\n{}\n'.format(backwardTransfer))
    output_file.close()

    return avgAccuracySTiPrec, backwardTransfer, forwardTransfer