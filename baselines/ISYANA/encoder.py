import torch
from torch.nn import functional as F
from linear_nets import MLP,fc_layer,cifar_net
from exemplars import ExemplarHandler
from continual_learner import ContinualLearner
from replayer import Replayer
import utils
import torchvision

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

class Classifier(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for classifying images, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", gated=False,
                 bias=True, excitability=False, excit_buffer=False, binaryCE=False, binaryCE_distill=False, AGEM=False,
                 data_name='splitMNIST'):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers

        # settings for training
        self.binaryCE = binaryCE
        self.binaryCE_distill = binaryCE_distill
        self.AGEM = AGEM  #-> use gradient of replayed data as inequality constraint for (instead of adding it to)
                          #   the gradient of the current data (as in A-GEM, see Chaudry et al., 2019; ICLR)

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        if 'cifar' not in data_name:
            self.resnet18 = False
            self.fcE = MLP(input_size=image_channels*image_size**2, output_size=fc_units, layers=fc_layers-1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=gated)
            mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2
            # classifier
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop)
        else:
            self.resnet = torchvision.models.resnet18(pretrained=True)
            self.resnet18 = True
            self.fcE = MLP(input_size=self.resnet.fc.out_features, output_size=fc_units, layers=fc_layers-1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=gated)
            mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2
            # classifier
            self.classifier = fc_layer(fc_units, classes, excit_buffer=True, nl='none', drop=fc_drop)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        return "{}_c{}".format(self.fcE.name, self.classes)


    def forward(self, x):     ##**Revised
        if not self.resnet18:
            x = self.flatten(x)
        prelayer_features, final_features = self.fcE(x)
        return prelayer_features, final_features, self.classifier(final_features)

    def feature_extractor(self, images):
        return self.fcE(self.flatten(images))

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.optimizer.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)    

                if momentum != 0:
                    param_state = self.optimizer.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if type(group['lr']) is float:
                    # print('lr',group['lr']) 
                    p.data.add_(-group['lr'], d_p)                  
                else:
                    p.data = p.data - group['lr'] * d_p

        return loss


    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, args=None, scores_=None, rnt=0.5, active_classes=None, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()


        # Should gradient be computed separately for each task? (needed when a task-mask is combined with replay)
        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False



        ##--(2)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self(x_)[2]

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id+1)
                    y_hat_all = self(x_temp_)[2]

                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    if self.binaryCE:
                        binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                        predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                            input=y_hat, target=binary_targets_, reduction='none'
                        ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                    else:
                        predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='elementwise_mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If task-specific mask, backward pass needs to be performed before next task-mask is applied
                if gradient_per_task:   ###self.mask_dict is not None:
                    #weighted_replay_loss_this_task = (1 - rnt) * loss_replay[replay_id] / n_replays
                    weight = 1 if self.AGEM else (1 - rnt)
                    weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        #loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)


        # If using A-GEM, calculate and store averaged gradient of replayed data
        if self.AGEM and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_task:
                loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()



        ##--(1)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model **Revised
            outputlayer1, outputlayer2, y_hat = self(x)
            #y_hat = self(x)          
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                if self.binaryCE_distill and (scores is not None):
                    classes_per_task = int(y_hat.size(1) / task)
                    binary_targets = binary_targets[:, -(classes_per_task):]
                    binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                predL = None if y is None else F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()     #--> sum over classes, then average over batch
            else:
                # -multiclass prediction loss
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y.type(torch.LongTensor).to(y.device), reduction='elementwise_mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If XdG is combined with replay, backward-pass needs to be done before new task-mask is applied
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt*loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [i.e., for offline with incremental task learning]



        # Combine loss from current and replayed batch
        if x_ is None or self.AGEM:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay


#########


        ##--(3)-- ALLOCATION LOSSES --##

        ### Inter-Task Synaptic Mapping Flow
        if args.isyana:
            ##ISYANA method
            surrogate_loss = self.isyanaSurrogate_loss()
        else:
            ##original method
            # Add SI-loss (Zenke et al., 2017)
            surrogate_loss = self.surrogate_loss()
            
        ##for ISYANA flow, it does not use the si_c and surrogate_loss
        if (self.si_c>0) and (args.si):
            print(self.si_c,surrogate_loss)
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss

        # print(self.optimizer)
        # Backpropagate errors (if not yet done)
        if (self.mask_dict is None) or (x_ is None):
            loss_total.backward()

        # If using A-GEM, potentially change gradient:
        if self.AGEM and x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur*grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep*grad_rep).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad and p.grad is not None:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        if args.isyana:
            # print(self.optimizer)
            self.step()
        else:
            self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }

