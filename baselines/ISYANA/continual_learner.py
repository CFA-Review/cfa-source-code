import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr  ### **Revised
import math
import utils
from torch import optim

class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.

    Adds methods for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC) and
    "synaptic intelligence" (SI) to its subclasses.'''

    def __init__(self):
        super().__init__()

        # XdG:
        self.mask_dict = None        # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # -SI:
        self.si_c = 0           #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # -EWC:
        self.ewc_lambda = 0     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None    #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False     #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass


    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, task):
        '''Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [task]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a task-specific mask has been specified
        for i,excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[task][i]] = 0.      # -> find task-specifc mask
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this mask

    def reset_XdGmask(self):
        '''Remove task-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this unit mask


    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None, resnet=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()
        # Create data-loader to give batches of size 1
        data_loader = utils.get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            if resnet:
                with torch.no_grad():
                    x = resnet(x)
            output = (self(x))[2] if allowed_classes is None else (self(x))[2][:, allowed_classes]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())



    #------------- "Inter-task Synaptic mapping"-specifc functions -------------#
    def isyanaUpdate_omega(self, W, epsilon, outputlayer1, outputlayer2, y_hat, task, classes_per_task, taskToClassDistanceTemp, prevlayer1nodeImportantoClassMatrix, prevlayer2nodeImportantoClassMatrix, flagSetSIPreTaskValue):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        tmpdevice = self._device()
        ##current node important to class (current task and + previous task) matrix
        tmplayer1nodeImportantoClassMatrix = torch.Tensor().float().to(tmpdevice) 
        tmplayer2nodeImportantoClassMatrix = torch.Tensor().float().to(tmpdevice)    

        ##According the current task No. Find the valid class/label   [0,...,m]  m = nClass 
        indexbegin = 0  ###classes_per_task * (task -1)
        indexbegin = classes_per_task * (task-1)
        indexend = classes_per_task * task 
        segmentindex = torch.arange(indexbegin,indexend)
        validTaskToClassDistance = taskToClassDistanceTemp.index_select(0,segmentindex.to(tmpdevice)).detach()  ##initialLabels.index_select(indexbegin,indexend) ##   initialLabels[indexbegin,indexend]


        tmpvarmatrixfinalout = torch.std(y_hat.data, dim=0)     
        tmpepsilon = 0.0001   ###when the variance is 0 in outputofeachneuron
        tmpadditionepsilon = 0.01
        omega_add = 0 ##**Revised add. initial value
        tmpi = 1   
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                
                ###**Revised
                if tmpi == 1:
                    ###layer1output, wsoutput 
                    tmpoutlayer1rowandcol = outputlayer1.size()
                    tmpoutputlayer1row = tmpoutlayer1rowandcol[0]
                    tmpoutputlayer1col = tmpoutlayer1rowandcol[1]

                    tmpyhatrowandcol = y_hat.size()
                    tmpyhatrow = tmpyhatrowandcol[0]
                    tmpyhatcol = tmpyhatrowandcol[1]

                    ###calculate variance for each neuron in hidden layer 1 --along column  eg. row 128, col 400
                    tmpvarmatrixlayer1out = torch.std(outputlayer1.data, dim=0)    
                    #print(tmpvarmatrixlayer1out.shape)                        
        
                    importantMatrixlayer1 = torch.zeros(tmpoutputlayer1col, tmpyhatcol)
                    for i in range(tmpyhatcol):
                        for j in range (tmpoutputlayer1col):   
                            tmpneuronvairance = tmpvarmatrixlayer1out[j]
                            if tmpneuronvairance == 0:
                                tmpneuronvairance = tmpepsilon
                  
                            ##Note: log is natural logarithm
                            neuronha = (1+math.log(2*(math.pi)*math.e*(tmpneuronvairance**2))) / 2.0
                            if neuronha < 0:
                                neuronha = 0   

                            tmpclassvairance = tmpvarmatrixfinalout[i]
                            if tmpclassvairance == 0:
                                tmpclassvairance = tmpepsilon
                            classhb = (1+math.log(2*(math.pi)*(tmpclassvairance**2))) / 2.0
                            if classhb < 0:
                                classhb = 0

                            ##correlation(A,B)
                            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html


                            corrab, p_value = pearsonr(outputlayer1.detach()[:,j].cpu(), y_hat.detach()[:,i].cpu())
                            if np.isnan(corrab):
                                corrab = 0
                            
                            if 1 == corrab or -1 == corrab:
                                Rab = 1
                            else:    
                                Iab = -math.log(1-corrab**2) / 2.0
                                Rab = 2.0 * Iab / (neuronha + classhb + tmpadditionepsilon)
                            
                            if Rab > 1:
                                Rab = 1 
                            importantMatrixlayer1[j][i] = Rab   ##inverse just to make the value small (loss is small). in order to learn

                    tmplayer1nodeImportantoClassMatrix = torch.cat((prevlayer1nodeImportantoClassMatrix,importantMatrixlayer1.to(tmpdevice)),1) ##along column connect
                    omega_add = torch.exp(-torch.mm(tmplayer1nodeImportantoClassMatrix, validTaskToClassDistance.unsqueeze(1)))  
                elif tmpi == 2: ##layer1 bias
                    omega_add = p.detach().clone().zero_()
                elif tmpi == 3:
                    ###layer2output, wsoutput
                    tmpoutlayer2rowandcol = outputlayer2.size()
                    tmpoutputlayer2row = tmpoutlayer2rowandcol[0]
                    tmpoutputlayer2col = tmpoutlayer2rowandcol[1]

                    tmpyhatrowandcol = y_hat.size()
                    tmpyhatrow = tmpyhatrowandcol[0]
                    tmpyhatcol = tmpyhatrowandcol[1]

                    ###calculate variance for each neuron in hidden layer 1 --along column  eg. row 128, col 400
                    tmpvarmatrixlayer2out = torch.std(outputlayer2.data, dim=0)                   
                    importantMatrixlayer2 = torch.zeros(tmpoutputlayer2col, tmpyhatcol)
                    for i in range(tmpyhatcol):
                        for j in range (tmpoutputlayer2col):
                            ##importantMatrixlayer2[400, 2]         
                            tmpneuronvairance = tmpvarmatrixlayer2out[j]
                            if tmpneuronvairance == 0:
                                tmpneuronvairance = tmpepsilon
                            neuronha = (1+math.log(2*(math.pi)*(tmpneuronvairance**2))) / 2.0
                            if neuronha < 0:
                                neuronha = 0

                            tmpclassvairance = tmpvarmatrixfinalout[i]
                            if tmpclassvairance == 0:
                                tmpclassvairance = tmpepsilon
                            classhb = (1+math.log(2*(math.pi)*(tmpclassvairance**2))) / 2.0
                            if classhb < 0:
                                classhb = 0

                            corrab, p_value = pearsonr(outputlayer2.detach()[:,j].cpu(), y_hat.detach()[:,i].cpu())
                            if np.isnan(corrab):
                                corrab = 0

                            if 1 == corrab or -1 == corrab:
                                Rab = 1
                            else:        
                                Iab = -math.log(1-corrab**2) / 2.0
                                Rab = 2.0 * Iab / (neuronha + classhb + tmpadditionepsilon) 
                            
                            if Rab > 1:
                                Rab = 1
                            importantMatrixlayer2[j][i] = Rab


                    tmplayer2nodeImportantoClassMatrix = torch.cat((prevlayer2nodeImportantoClassMatrix,importantMatrixlayer2.to(tmpdevice)),1) ##along column connect
                    omega_add = torch.exp(-torch.mm(tmplayer2nodeImportantoClassMatrix, validTaskToClassDistance.unsqueeze(1))) 
                elif tmpi == 4:
                    omega_add = omega_add = p.detach().clone().zero_()  
                else: ###for final output. the importance. skip
                    omega_add = p.detach().clone().zero_()                    

                tmpi = tmpi + 1 


                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega_add     
            

                # Store these new values in the model
                if 1 == flagSetSIPreTaskValue:   ##flagSetSIPreTaskValue  only has 0 or 1
                    self.register_buffer('{}_SI_prev_task'.format(n), p_current)

                self.register_buffer('{}_SI_omega'.format(n), omega_new)
        
        return tmplayer1nodeImportantoClassMatrix, tmplayer2nodeImportantoClassMatrix



    ##Synaptic mapping
    def isyanaSurrogate_loss(self):
        '''Calculate ISYANA's surrogate loss.'''
        try:
            losses = []
            optim_list = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))

                    # adaptive learning rate
                    if 'bias' not in n and 'classifier' not in n:
                        s = p.shape
                        zt = omega.repeat(1,s[1])
                    else:
                        zt = omega.squeeze()

                    alp = 1 * torch.exp(- zt - 0.1)

                    if 'classifier' in n:
                        alp = 0.01
                    if 'bias' in n:
                        alp = 0.01

                    optim_list = optim_list + [{'params': p, 'lr': alp}]


                    ##Notice "*" means dot/mul (element-wise multiple)
                    # Calculate synaptic mapping's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())  ### original.

            self.optimizer = optim.SGD(optim_list)

            # print("\nlosses\n", losses)
            return sum(losses)
        except AttributeError:
            # ISYANA-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())




    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add  

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)


    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())