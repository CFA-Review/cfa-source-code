#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main
import random


###This script is only for running two algorithms: EWC and O-EWC. Finally, it will get results of the two algorithms

description = 'Compare performance of CL strategies on each scenario of permuted or split MNIST.'
parser = argparse.ArgumentParser('./_compare.py', description=description)

###original seed value is 1. Different from the original code seed 0
##parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--seed', type=int, default=0, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
#parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")  ##original
parser.add_argument('--cuda', type=int, default=0, help="don't use GPUs")  ## ***Revised

parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST', 'rotMNIST'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, default=5, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")

# model architecture parameters
model_params = parser.add_argument_group('Parameters Main Model')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, default=400, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, default=2000, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')
train_params.add_argument('--epoch', type=int, default=1, help="training epoches")


# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--lambda', type=float, default=5000.,dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--o-lambda', type=float, default=5000., help="--> online EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--gamma', type=float, default=1., help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--c', type=float, default=0.1, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', type=float, default=0.8, dest="xdg",help="XdG: prop neurons per layer to gate")

# iCaRL parameters
icarl_params = parser.add_argument_group('iCaRL Parameters')
icarl_params.add_argument('--budget', type=int, default=2000, dest="budget", help="how many exemplars can be stored?")
icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")
icarl_params.add_argument('--use-exemplars', action='store_true', help="use stored exemplars for classification?")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")



def get_prec(args, ext="", curepoch=0):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)


    # -check whether already run; if not do so

    ##Revise. always execute.
    ##if not os.path.isfile('{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)):
    print(" ...running: ... ")
    main.run(args, curepoch)

    ##eg. time-splitMNIST5-task--MLP([784, 256, 256])_c10--i2000-lr0.015-b128-sgd--ISYANA0.05.txt
    #1 -get average precision
    fileName = '{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)
    file = open(fileName)

    ###---Revised.
    ###After revised the file for storing the result. Modify this part of code for reading.
    tmpcount = 1
    while tmpcount <= 2: ####+ curepoch*6:   ##an = a1 + (n-1)*6
        tmpeachline = file.readline()
        tmpcount = tmpcount + 1

    ave = float(tmpeachline)
    ##ave = float(file.readline())

    file.close()


    #2 -get forward transfer
    fileName = '{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)
    file = open(fileName)

    tmpcount = 1
    while tmpcount <= 4: ###+ curepoch*6:   ##an = a1 + (n-1)*6
        tmpeachline = file.readline()
        tmpcount = tmpcount + 1

    forwardtransfer = float(tmpeachline)
    file.close()


    #3 -get backward transfer 
    fileName = '{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)
    file = open(fileName)

    tmpcount = 1
    while tmpcount <= 6: ###+ curepoch*6:   ##an = a1 + (n-1)*6
        tmpeachline = file.readline()
        tmpcount = tmpcount + 1

    backwardtransfer = float(tmpeachline)
    ##ave = float(file.readline())

    file.close()

    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision
    return ave, forwardtransfer, backwardtransfer


def collect_all(method_dict, seed_list, args, ext="", name=None, curepoch=0):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed], forwardtransfer, backwardtransfer = get_prec(args, ext=ext, curepoch=curepoch)
    # -return updated dictionary with results
    return method_dict, forwardtransfer, backwardtransfer



if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    # -set default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)



    ###Revise. clear result file. at the beginning (eg. history data)
    # name for plot
    plot_name = "summary-{}{}-{}-{}".format(args.experiment, args.tasks, args.scenario, args.optimizer)    
    output_file = open("{}/{}.txt".format(args.p_dir, plot_name), 'w')        
    output_file.close()


    ###first 
    ##first clean forward file
    fwdplot_name = "summary-{}{}-{}-{}-{}".format(args.experiment, args.tasks, args.scenario, args.optimizer,"fwd")    
    fwdoutput_file = open("{}/{}.txt".format(args.p_dir, fwdplot_name), 'w')        
    fwdoutput_file.close()
    ##re-open forward file
    fwdoutput_file = open("{}/{}.txt".format(args.p_dir, fwdplot_name), 'a') 


    ###clean backward file
    bwdplot_name = "summary-{}{}-{}-{}-{}".format(args.experiment, args.tasks, args.scenario, args.optimizer,"bwd")    
    bwdoutput_file = open("{}/{}.txt".format(args.p_dir, bwdplot_name), 'w')        
    bwdoutput_file.close()
    ##re-open backward file
    bwdoutput_file = open("{}/{}.txt".format(args.p_dir, bwdplot_name), 'a')  


    ##run several time   ---Revise
    for tmpepoch in range(args.epoch):

        ## Add non-optional input argument that will be the same for all runs
        args.feedback = False
        args.log_per_task = True

        ## Add input arguments that will be different for different runs
        args.distill = False
        args.ewc = False
        args.online = False
        args.si = False
        args.isyana = False
        args.gating_prop = 0.
        args.add_exemplars = False
        args.bce_distill= False
        args.icarl = False
        # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#
        seed_list = list(range(args.seed, args.seed+args.n_seeds))


        ## None
        args.replay = "none"


        ###----"REGULARIZATION"----####

        ## EWC
        args.ewc = True
        EWC = {}
        EWC, forwardtransfer, backwardtransfer = collect_all(EWC, seed_list, args, name="EWC", curepoch=tmpepoch)

        ##record forward result
        fwdoutput_file.write("{:12s} {:.6f}  ".format("EWC", forwardtransfer))

        #record backward result
        bwdoutput_file.write("{:12s} {:.6f}  ".format("EWC", backwardtransfer))

        ## online EWC
        args.online = True
        args.ewc_lambda = args.o_lambda
        OEWC = {}
        OEWC, forwardtransfer, backwardtransfer = collect_all(OEWC, seed_list, args, name="Online EWC", curepoch=tmpepoch)
        args.ewc = False
        args.online = False

        ##record forward result
        fwdoutput_file.write("{:12s} {:.6f}  ".format("o-EWC", forwardtransfer))

        #record backward result
        bwdoutput_file.write("{:12s} {:.6f}  ".format("o-EWC", backwardtransfer))


        ##after one loop
        fwdoutput_file.write('\n')
        bwdoutput_file.write('\n')

        ##close forward result file 
        if (tmpepoch == args.epoch-1):  
            fwdoutput_file.close()

        #close backward result file 
        if (tmpepoch == args.epoch-1):
            bwdoutput_file.close()

        #-------------------------------------------------------------------------------------------------#

        #---------------------------#
        #----- COLLECT RESULTS -----#
        #---------------------------#

        ave_prec = {}
        for seed in seed_list:
            ave_prec[seed] = [EWC[seed], OEWC[seed]]




        #-------------------------------------------------------------------------------------------------#

        #--------------------#
        #----- PLOTTING -----#
        #--------------------#

        # name for plot
        plot_name = "summary-{}{}-{}-{}".format(args.experiment, args.tasks, args.scenario, args.optimizer)
        scheme = "{}-incremental learning".format(args.scenario)
        title = "{}  -  {}".format(args.experiment, scheme)




        names = ["EWC"]
        colors = ["grey"]
        # select names / colors / ids
        ids = [0]


        names += ["o-EWC"]
        colors += ["deepskyblue"]
        ids += [1] ###---Revised



        # open pdf
        pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
        figure_list = []


        # bar-plot
        means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
        if len(seed_list)>1:
            sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
            cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                    title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
        figure_list.append(figure)

        # print results to screen
        print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
        for i,name in enumerate(names):
            if len(seed_list) > 1:
                print("{:12s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
            else:
                print("{:12s} {:.2f}".format(name, 100*means[i]))
        print("#"*60)


        #save results to file
        output_file = open("{}/{}.txt".format(args.p_dir, plot_name), 'a')        
        for i,name in enumerate(names):
            if len(seed_list) > 1:
                output_file.write("{:12s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
            else:
                output_file.write("{:12s} {:.2f}  ".format(name, 100*means[i]))

        output_file.write('\n')
        output_file.close()


        # add all figures to pdf
        for figure in figure_list:
            pp.savefig(figure)

        # close the pdf
        pp.close()

        # Print name of generated plot on screen
        print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
        
        ##if run several times.
        args.seed = random.randint(0, 10000)  