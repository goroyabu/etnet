#!/usr/bin/env python3
# coding: UTF-8

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

from torch.utils.data import DataLoader

from etnet.dataset import EtrackDataset
from etnet.neuralnet import EtrackNet


def usage():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s [options] FILE...",
        add_help=False,
    )

    parser.add_argument(
        "file", metavar="FILE", type=str, nargs="+", help="CSV file name"
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=2,
        metavar="Ne",
        help="number of epochs to train",
    )

    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=0.001,
        metavar="L",
        help="learning rate",
    )

    parser.add_argument(
        "--mu",
        "-m",
        type=float,
        default=0.0,
        metavar="M",
        help="mu for smoothing term",
    )

    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=1.0,
        metavar="B",
        help="beta for sparce term",
    )

    parser.add_argument(
        "--batchsize",
        "-bs",
        type=int,
        default=100,
        metavar="Nb",
        help="mini batch size",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="model",
        metavar="DIR",
        help="name of output directory",
    )

    parser.add_argument("--use-gpu", action="store_true", help="enable GPU")

    parser.add_argument(
        "--cuda",
        type=str,
        default='"cuda:0"',
        metavar="C",
        help="name of cuda",
    )

    parser.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    return parser


def outname(input, args):
    base = os.path.basename(input)
    name, ext = os.path.splitext(base)
    outname = f"{name}_lr{args.learning_rate}_beta{args.beta}_mu{args.mu}"
    return outname


def main(data_path, args):

    log_file_name = args.outdir + "/log_" + outname(data_path, args) + ".txt"
    losstrend_file = open(log_file_name, "w")

    train_data = EtrackDataset(data_path=data_path, train=True)
    test_data = EtrackDataset(data_path=data_path, train=False)
    num_traindata = len(train_data)
    num_testdata = len(test_data)
    print("EtrackDataset loaded")

    batch_size = args.batchsize
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_data, batch_size=num_testdata, shuffle=False)

    net = EtrackNet()
    print("EtrackNet generated")

    if args.use_gpu:
        net = net.to(args.cuda)
    optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate)

    inputs_test = None
    targets_test = None
    for i, (inputs, targets) in enumerate(test_loader):
        if i == 0:
            inputs_test = inputs.float()
            targets_test = targets.float()
            if args.use_gpu:
                inputs_test = inputs_test.to(args.cuda)
                targets_test = targets_test.to(args.cuda)
            break

    net.train()
    
    print("")
    print("Begin Epoch Loop...")
    for epoch in range(args.epochs):

        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.float()
            targets = targets.float()
            if args.use_gpu:
                inputs = inputs.to(args.cuda)
                targets = targets.to(args.cuda)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss, _, _ = EtrackDataset.custom_loss(
                outputs, targets, args.beta, args.mu
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 != 99:
                continue

            net.train = False

            with torch.no_grad():
                optimizer.zero_grad()
                outputs_test = net(inputs_test)
                (
                    loss_test,
                    cross_entropy_pos,
                    cross_entropy_phi,
                ) = EtrackDataset.custom_loss(
                    outputs_test, targets_test, 1.0, 0.0
                )

            print(
                f"[{epoch+1:d}, {i+1:d}] ",
                f"loss (train): {running_loss/100:.15f} ",
                f"loss (test ): {loss_test.item():.15f} ",
                f"{cross_entropy_pos.item():.15f} ",
                f"{cross_entropy_phi.item():.15f} ",
                sep=",",
            )

            print(
                epoch * (num_traindata / batch_size) + i + 1,
                running_loss / 100,
                loss_test.item(),
                cross_entropy_pos.item(),
                cross_entropy_phi.item(),
                file=losstrend_file,
            )

            running_loss = 0.0
            net.train = True

        model_path = (
            args.outdir
            + "/model_"
            + outname(data_path, args)
            + f"_epoch{epoch+1}.pth"
        )
        torch.save(net.state_dict(), model_path)

    print("Finished Training")
    print("")


if __name__ == "__main__":

    parser = usage()
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for file in args.file:
        # print(file)

        start_time = time.time()
        main(file, args)
        print(f"elapsed time: {time.time() - start_time:.3f} [sec]")
