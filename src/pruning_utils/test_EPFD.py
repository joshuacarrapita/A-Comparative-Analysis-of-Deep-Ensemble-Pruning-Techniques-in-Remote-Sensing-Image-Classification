# coding: utf-8
# import system libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
import sys

# import 3rd party libraries
from sklearn import datasets
from sklearn import neighbors
import torch
import lightning.pytorch as pl
import numpy as np
from pathlib import Path

# import local libraries
from pyensemble.classify import BaggingEnsembleAlgorithm
from epfd.data_distributed import (distributed_single_pruning,
                              distributed_pruning_methods)
from epfd.data_distributed import COMEP_Pruning, DOMEP_Pruning


def define_params(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-cls", type=int, default=17,
        help='Number of individual classifiers in the ensemble')
    parser.add_argument("--nb-pru", type=int, default=5,
        help='Number of members in the pruned sub-ensemble')

    parser.add_argument("--name-pru", type=str, default='COMEP',
        choices=['ES', 'KP', 'KL', 'RE', 'OO',
                 'DREP', 'SEP', 'OEP', 'PEP',
                 'GMA', 'LCS', 'COMEP', 'DOMEP'],
        help='Name of the expected ensemble pruning method')

    parser.add_argument("--distributed", action="store_true",
        help='Whether to use EPFD (framework)')

    parser.add_argument("--lam", type=float, default=0.5,
        help="lambda")
    parser.add_argument("--m", type=int, default=2,
        help='Number of Machines')
    parser.add_argument('--seed', type=int, default='666', help='Random seed for RNG.')
    parser.add_argument('--results', type=str, required=True, help='Input path for ensemble inference results')
    parser.add_argument('--targets', type=str, required=True, help='Input path for ensemble inference targets')

    return parser.parse_args(args)


def main(args):
    nb_cls = args.nb_cls # No. classifiers in the ensemble
    nb_pru = args.nb_pru # No. classifiers in selected sub-ensemble
    name_pru = args.name_pru # Name of pruning method
    distributed = args.distributed # Used distributed EFPD / Pruning framework
    lam = args.lam # Value of lambda
    m = args.m # Number of machines
    seed = args.seed
    pl.seed_everything(seed=seed)

    probas = torch.load(Path(args.results).resolve()).float().numpy()
    results = np.argmax(probas, axis=2)
    targets = torch.load(Path(args.targets).resolve()).float().numpy()
    rho = nb_pru / nb_cls

    if name_pru not in ['COMEP', 'DOMEP']:
        since = time.time()
        Pc = distributed_single_pruning(name_pru, targets, results,
                nb_cls, nb_pru, rho=rho)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

        if distributed:
            since = time.time()
            Pd = distributed_pruning_methods(targets, results,
                nb_pru, m, name_pru, rho=rho)
            Td = time.time() - since
            print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

    elif name_pru == 'COMEP':
        since = time.time()
        Pc = COMEP_Pruning(np.array(results).T.tolist(), nb_pru, targets, lam)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

    elif name_pru == 'DOMEP':
        since = time.time()
        Pd = DOMEP_Pruning(np.array(results).T.tolist(), nb_pru, m, targets, lam)
        Td = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

    else:
        raise ValueError("Please check the `name_pru`.")


if __name__ == "__main__":
    args = define_params(sys.argv[1:])
    main(args)

