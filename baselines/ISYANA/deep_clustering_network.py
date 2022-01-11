#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import pdb
import numpy as np
import copy
import time




#device = 'cuda'
device = 'cpu'

#for self-adaptive clustering approach.  Take the referece paper: Towards K-means -friendly spaces: simultaneous deep learning and clustering
###for each Task it has a class: averageMeter
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
       Modified by Arasnet team
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.avg_old = 0
        self.std = 0.01
        self.std_old = 0.01
        self.count = 0
        self.miu_min = 100
        self.std_min = 100

    def updateMiuStd(self, val, n=1):
        self.value   = val
        self.avg_old = copy.deepcopy(self.avg)
        self.std_old = copy.deepcopy(self.std)
        self.count  += n
        self.avg     = self.avg_old + np.divide((val-self.avg_old),self.count)
        self.std     = np.sqrt(self.std_old**2 + self.avg_old**2 - self.avg**2 + ((val**2 - self.std_old**2 - self.avg_old**2)/self.count))

        
    def reset_min(self):
        self.miu_min = copy.deepcopy(self.avg)
        self.std_min = copy.deepcopy(self.std)
        
    def update_min(self):
        if self.avg < self.miu_min:
            self.miu_min = copy.deepcopy(self.avg)
        if self.std < self.std_min:
            self.std_min = copy.deepcopy(self.std)



def deleteTensor(x,index):
    x = x[torch.arange(x.size(0))!=index] 
    return x



def extractDigits(lst): 
    res = [] 
    for el in lst: 
        sub = el
        res.append([sub]) 
      
    return(res) 


tmpepsilon = 0.001   ###when the variance is 0 
class autoencoder_mnist(nn.Module):
    def __init__(self):
        super(autoencoder_mnist, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())


    def forward(self, x):
        hiddenx = self.encoder(x)
        h = self.relu(hiddenx.clone()) + tmpepsilon
        outputx = self.decoder(hiddenx)
        return outputx, h


class autoencoder_cifar(nn.Module):
    def __init__(self):
        super(autoencoder_cifar, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Linear(1000, 250),
            nn.ReLU(True),
            nn.Linear(250, 100),
            nn.ReLU(True), 
            nn.Linear(100, 3), 
            nn.ReLU(True), 
            # nn.Linear(125, 50),
            )
        self.decoder = nn.Sequential(
            # nn.Linear(50, 125),
            # nn.ReLU(True),
            nn.Linear(3, 100),
            nn.ReLU(True),
            nn.Linear(100, 250),
            nn.ReLU(True), nn.Linear(250, 1000), nn.Tanh())


    def forward(self, x):
        hiddenx = self.encoder(x)
        h = self.relu(hiddenx.clone()) + tmpepsilon
        outputx = self.decoder(hiddenx)
        return outputx, h

def clusteringProcess(x,labels,curcluster,criterion,miuMinDis,device):
    accumulatedLoss = torch.Tensor().float().to(device)
           
    # loss calculation
    for i in range(0,x.shape[0]): # loop over data
        distanceXtoClust = torch.Tensor().float().to(device) 
        ###first dive x into several parts according to the labels. and then get miu and std. then use kl to calculate 
        for (cluster_idx,center) in enumerate(curcluster.clusterCenter): # loop over cluster
            currCenter = center.clone().detach()
            currCenter.requires_grad_()
            currCenter.to(device)
            distanceXtoCurrClust = criterion(x[i], currCenter) # calculate loss    according to label update miu
            distanceXtoClust = torch.cat((distanceXtoClust,torch.unsqueeze(distanceXtoCurrClust, dim = -1)),0)
        
        loss, winnerCluster = torch.min(distanceXtoClust,0) #get loss and winner cluster
        accumulatedLoss = torch.cat((accumulatedLoss,torch.unsqueeze(loss, dim = -1)),0) 
        
        # calculate average of minimum distance
        minimumDistance = loss.clone().detach().cpu().item()
        miuMinDis.updateMiuStd(minimumDistance)
        
        # grow cluster condition
        Lclus = 0.5*torch.norm(x[i].clone().detach() - curcluster.clusterCenter[winnerCluster])
        k_sigma = 2*torch.exp(-Lclus**2).item() + 1
        threshold = miuMinDis.avg + k_sigma*miuMinDis.std
        if minimumDistance > threshold:
            # grow cluster
            curcluster.addCluster(x[i].clone().detach().unsqueeze(0),labels[i],torch.ones(1))
            # print('number of cluster: ',curcluster.clusterCenter.shape[0])            
                    
        # update cluster
        inputCluster = x[i].detach()
        curcluster.updateCluster(inputCluster,labels[i],winnerCluster)
    
    averageAccLoss = torch.mean(accumulatedLoss)
    return averageAccLoss, curcluster, miuMinDis



class cluster(object):
    def __init__(self,initialCenter, nClass, initialLabels = -1, taskNo = -1, classes_per_task = -1):
        indexbegin = classes_per_task * (taskNo -1)
        indexend = classes_per_task * taskNo 
        segmentindex = torch.arange(indexbegin,indexend)

        self.clusterCenter = initialCenter[indexbegin:indexend]
        self.clustersLabel = initialLabels.index_select(0,segmentindex.to(device)).detach()  ##initialLabels.index_select(indexbegin,indexend) ##   initialLabels[indexbegin,indexend]
        self.noOfCluster = self.clusterCenter.shape[0]
        self.clusterClassCounter = (torch.eye(len(initialLabels),nClass))[indexbegin:indexend]
        self.nClass = nClass

        
    def addCluster(self,newClusterCenter,newLabel,newCounter):
        self.clusterCenter = torch.cat((self.clusterCenter,newClusterCenter),0)
        self.clusterCounter = torch.cat((self.clusterCounter,newCounter),0)
        newClusterClassCounter = torch.zeros(1,self.nClass)
        newClusterClassCounter[0][newLabel] = 1
        self.clusterClassCounter = torch.cat((self.clusterClassCounter,newClusterClassCounter),0)
        self.clustersLabel = torch.cat((self.clustersLabel, newLabel.unsqueeze(0).to(device)),0)
        self.noOfCluster = self.clusterCenter.shape[0]

        

    def addClusterToGlobalCluster(self, newCluster):
        self.clusterCenter = torch.cat((self.clusterCenter, newCluster.clusterCenter),0)
        self.clusterClassCounter = torch.cat((self.clusterClassCounter, newCluster.clusterClassCounter),0)
        self.clustersLabel = torch.cat((self.clustersLabel, newCluster.clustersLabel),0)
        self.noOfCluster = self.clusterCenter.shape[0]


    def updateCluster(self,x,label,winner_idx):
        self.clusterCounter[winner_idx] = self.clusterCounter[winner_idx] + torch.tensor(1)
        self.clusterCenter[winner_idx] = (self.clusterCenter[winner_idx] - 
                                          (torch.tensor(1.0)/self.clusterCounter[winner_idx]*(self.clusterCenter[winner_idx]-x)))
        self.clusterClassCounter[winner_idx][label] = self.clusterClassCounter[winner_idx][label] + 1
        _,self.clustersLabel[winner_idx] = torch.max(self.clusterClassCounter[winner_idx],0)
        
    ##Intra cluster merging. Need to update the number of clusters etc.
    def mergeWith(self, mergeFromcluster, mergeFromIdx, mergeToIdx):
        self.clusterCenter[mergeToIdx] = ((self.clusterCounter[mergeToIdx]*self.clusterCenter[mergeToIdx] + 
                                           mergeFromcluster.clusterCounter[mergeFromIdx]*mergeFromcluster.clusterCenter[mergeFromIdx])/
                                          (self.clusterCounter[mergeToIdx] + mergeFromcluster.clusterCounter[mergeFromIdx]))
        self.clusterCounter[mergeToIdx] = self.clusterCounter[mergeToIdx] + mergeFromcluster.clusterCounter[mergeFromIdx]
        self.clusterClassCounter[mergeToIdx] = (self.clusterClassCounter[mergeToIdx] + 
                                                mergeFromcluster.clusterClassCounter[mergeFromIdx])
        _,self.clustersLabel[mergeToIdx] = torch.max(self.clusterClassCounter[mergeToIdx],0)
        
        # delete
        self.clusterCenter = deleteTensor(self.clusterCenter,mergeFromIdx)
        self.clusterCounter = deleteTensor(self.clusterCounter,mergeFromIdx)
        self.clusterClassCounter = deleteTensor(self.clusterClassCounter,mergeFromIdx)
        self.clustersLabel = deleteTensor(self.clustersLabel,mergeFromIdx)
        
        self.noOfCluster = self.clusterCenter.shape[0]
    

    def taskToTaskDistance(self,x):
        # x will be hidden representation of the current task
        self.interTaskDistance = torch.max(torch.exp(-(torch.norm(self.clusterCenter - x.clone().detach(),dim=1))**2))
        return self.interTaskDistance
    

    def intraTaskMerging(self,threshold):
        intraClusterB = copy.deepcopy(self)
        intraClusterCenterB = intraClusterB.clusterCenter
        afterMergeNeedToRedo = 1 ##True
        intraclusterB_indx_mergePosition = 0
        while(afterMergeNeedToRedo):
            for (intraClusterA_idx,clusterACenter) in enumerate(self.clusterCenter.clone().detach()):        
                clusterBeMergedFlag = 0 ##False. variable to check whether cluster is merged successfully
                for (intraClusterB_idx,clusterBCenter) in enumerate(intraClusterCenterB):
                    if intraClusterB_idx > intraClusterA_idx and intraClusterB_idx > intraclusterB_indx_mergePosition:
                        clusterDistance = torch.norm(clusterACenter - clusterBCenter)

    #                     ******#PROBLEM: MIUMINDIS IS NOT CORRECT ******************************          
                        if clusterDistance.detach().item() < threshold.avg + 0.5*threshold.std:
                            # calculate new center
                            self.mergeWith(intraClusterB, intraClusterB_idx,intraClusterA_idx)
                            intraclusterB_indx_mergePosition = intraClusterB_idx ##update mergePosition. record the current position
                            clusterBeMergedFlag = 1 ##True  Merged successfully
                            break

                if intraclusterB_indx_mergePosition == len(intraClusterCenterB) - 1:  ##finish merged all the intra clusters
                    afterMergeNeedToRedo = 0
                    break

                if 1 == clusterBeMergedFlag:
                    break  ##after merging. redo-merge from the beginning.
                else:
                    afterMergeNeedToRedo = 0 ## finish merging --already traverse all the clusters




def collectClusterList(globalCluster, nClass):   ##add nclass
    classClusterList = [ [] for i in range(nClass)]
    for iCluster in range(0,globalCluster.noOfCluster):
        _,iClusBelongToClass = torch.max(globalCluster.clusterClassCounter[iCluster],0)
        classClusterList[iClusBelongToClass].append(iCluster)
    
    ###in classclusterlist, a class may do not have cluster. For example, if a class has small samples, and in a cluster, this class is not the main class. The clusterLable will be marked as other class
    return classClusterList



def isvalueExistInTensor(tensorlist, value):
    return len((tensorlist == value).nonzero())
    


def getClassCenterOfFirstTask(curcluster,inputX,labely,currentTaskLabels,device):     
    for curclass in range(0, len(currentTaskLabels)):
        curpickIndex = torch.Tensor().long()
        curselectedlabel = torch.eq(labely, currentTaskLabels[curclass])
        tmpcurselectedIndex = torch.nonzero(curselectedlabel)
        curselectedIndex = torch.squeeze(tmpcurselectedIndex) 
        if (len(curselectedIndex) == 0):
            curcluster.clusterCenter[curclass ] = torch.zeros(inputX.shape[1])
            continue

        curpickIndex = torch.cat((curpickIndex, curselectedIndex),0)
        curgroup = inputX.index_select(0,curpickIndex.to(device)).detach()
        meancurgroup = torch.mean(curgroup, dim=0)
        curcluster.clusterCenter[curclass] = copy.deepcopy(meancurgroup)



##get TT value
def taskToClassDistance(cluster,curcluster,inputX,labely,nClass,classClusterList,currentTaskLabels,device,taskno = -1):
    taskToAllClassDistance = torch.Tensor().float().to(device) 
    
    numclassofinputX = 0    
    if 1 != taskno:   ##For KL divergence. First task should do special
        for curclass in range(0, len(currentTaskLabels)):
            curpickIndex = torch.Tensor().long()
            curselectedlabel = torch.eq(labely, currentTaskLabels[curclass])
            tmpcurselectedIndex = torch.nonzero(curselectedlabel)
            curselectedIndex = torch.squeeze(tmpcurselectedIndex) 
            if (curselectedIndex.dim() == 0 or len(curselectedIndex) == 0):
                curcluster.clusterCenter[curclass] = torch.zeros(inputX.shape[1])
                continue

            curpickIndex = torch.cat((curpickIndex, curselectedIndex),0)
            curgroup = inputX.index_select(0,curpickIndex.to(device)).detach()
            meancurgroup = torch.mean(curgroup, dim=0)
            curcluster.clusterCenter[curclass] = copy.deepcopy(meancurgroup)
            numclassofinputX = numclassofinputX + 1


    for iClass in range(0,nClass):
        taskToEachClassDistance = torch.Tensor().float().to(device)

        #if iClass in currentTaskLabels:
        if isvalueExistInTensor(currentTaskLabels, iClass): ##open TT   ST*TT
            ##it means the item is very relevant to current tasks/labels. give the maximum value 1.0. original range [0, 1]
            taskToEachClassDistance = torch.tensor([1.0]).to(device)
        else:
            if 1 != taskno:  ##if taskno is 1, just skip
                TaskToClassDistanceTemp = 0.0
                isEnterForloopflag = 0  ##only has two values: 0 and 1 
                for _,iCluster in enumerate(classClusterList[iClass]):
                    for curclass in range(0, len(currentTaskLabels)):
                        isEnterForloopflag = 1
                        TaskToClassDistanceTemp = TaskToClassDistanceTemp + 1.0 / torch.abs(klDiv(cluster.clusterCenter[iCluster], curcluster.clusterCenter[curclass]))
                
                if 1 == isEnterForloopflag:
                    meanTaskToClassDistanceTemp = TaskToClassDistanceTemp * 1.0 / numclassofinputX                                
                    taskToEachClassDistance = torch.cat((taskToEachClassDistance,meanTaskToClassDistanceTemp.unsqueeze(0)),0)


            if len(taskToEachClassDistance) ==  0:  ##if no element exists. give the minimum value 0. the value range is [0,1]
                taskToEachClassDistance = torch.tensor([0.0]).to(device)
        
        taskToAllClassDistance = torch.cat((taskToAllClassDistance,torch.max(taskToEachClassDistance).unsqueeze(0)),0)
        
    return taskToAllClassDistance


def klDiv(classDistribution1,classDistribution2):
    tmpepsilon = 0.001

    ##torch.where((classDistribution1 < 1e-8 or classDistribution1 > -1e-8), classDistribution1 + tmpepsilon, classDistribution1)

    #if(classDistribution1 < 1e-8) is not None and (classDistribution1 > -1e-8) is not None:
    ##does not contain 0
    if(not (classDistribution1.squeeze().size() == classDistribution1.nonzero().squeeze().size())):    
        classDistribution1 = classDistribution1 + tmpepsilon

    if(not (classDistribution2.squeeze().size() == classDistribution2.nonzero().squeeze().size())):  
        classDistribution2 = classDistribution2 + tmpepsilon       

    # if(torch.numel((classDistribution1 < 1e-8 or classDistribution1 > -1e-8).nonzero())):
    #     classDistribution1 = classDistribution1 + tmpepsilon

    ##torch.where((classDistribution2 < 1e-8 or classDistribution2 > -1e-8)>0, classDistribution2 + tmpepsilon, classDistribution2)
    # if(torch.numel((classDistribution2 < 1e-8 or classDistribution2 > -1e-8).nonzero())):
    #     classDistribution2 = classDistribution2 + tmpepsilon

    tmpdivvalue = (classDistribution1 * 1.0 / classDistribution2)
    tmplogvalue = torch.log(tmpdivvalue) ##torch.log(classDistribution1 * 1.0 / classDistribution2)
    
    klDistance = torch.sum(classDistribution1*tmplogvalue)
    
    ##the division cannot be 0
    if torch.abs(klDistance) < torch.tensor(1e-8):  
        klDistance = torch.tensor(tmpepsilon)

    return klDistance



def interTaskMerging(globalCluster,currCluster):
    # combine task cluster to global cluster
    globalCluster.addClusterToGlobalCluster(currCluster) 
    return globalCluster

