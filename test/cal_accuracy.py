#!/usr/bin/env python3

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy
from ROOT import TFile, TH1D, TH2D, TCanvas
from ROOT import gROOT, gPad, gStyle
import ROOT

from etnet.dataset import EtrackDataset
from etnet.neuralnet import EtrackNet

def usage():

    parser = argparse.ArgumentParser(
        description="Accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        usage="%(prog)s [options] FILE ...",
    )

    parser.add_argument(
        "file", metavar="FILE", type=str, nargs="+", help=".pth file name"
    )

    parser.add_argument("--noutput", metavar="N", type=int, default=-1, help="")

    parser.add_argument("--csv", metavar="DATA", type=str, help="csv file")

    parser.add_argument(
        "--outdir", type=str, default="ana", help="directory name to output"
    )

    parser.add_argument("--use-gpu", action="store_true", help="enable GPU")

    parser.add_argument("--cuda", type=str, default='"cuda:0"', help="name of cuda")

    parser.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    return parser


def estimation(model_path, outfilename, data_path, args):

    testdata = EtrackDataset(data_path=data_path, train=False)
    nevents = len(testdata)
    if args.noutput > 0 and args.noutput < nevents:
        nevents = args.noutput
    print("Input:", data_path, "is loaded")
    print("Number of test data :", nevents)

    net = EtrackNet(train=False)
    if args.use_gpu:
        net = net.to(args.cuda)
    net.load_state_dict(torch.load(model_path))

    outfile = TFile(outfilename, "recreate")
    print(f"Output: {outfilename} is created")

    # th2_diff_pos = TH2D(
    #     "th2_diff_pos", "diff_pos;#Delta Pixel;Counts", 128, -4, 4, 128, -4, 4
    # )
    th2_diff_pos = TH2D(
        "th2_diff_pos", "Distance True and Estimation;RelativeX [-1,1];RelativeY", 
        EtrackDataset.N_PIXELS_1D*10, -1, 1,
        EtrackDataset.N_PIXELS_1D*10, -1, 1
    )
    th1_diff_phi = TH1D(
        "th1_diff_phi", "diff_phi;#Delta Phi (deg.)", 
        EtrackDataset.N_LABELS_PHI*10, -180, 180
    )
    th1_diff_phi_nice = th1_diff_phi.Clone()
    th1_diff_phi_nice.SetNameTitle("th1_diff_phi_nice", "diff_phi_nice")

    # diff_cos_beta = ROOT.TH1D("diff_cos_beta", "diff_cos_beta;#Delta cos #beta;Counts", 128, -1, 1)
    # diff_phi_2d = ROOT.TH2D("diff_phi_2d", "diff_phi_2d;#Delta Phi (deg.);cos #beta;Counts", 128, -180, 180, 6, 0, 1)
    # diff_cos_beta_2d = ROOT.TH2D("diff_cos_beta_2d", "diff_cos_beta_2d;#Delta cos #beta;cos #beta;Counts", 128, -1, 1, 6, 0, 1)
    # corr_cos_beta_numpixnonzero = ROOT.TH2D( "corr_cos_beta_numpixnonzero",\
    #     "corr_cos_beta_numpixnonzero;cos #beta;NumPixNonZero;Counts", 1, 0, 1, 64, 0, 64)

    th1_maxlike_phi = TH1D(
        "th1_maxlike_phi",
        "Likelihood Map of Phi",
        EtrackDataset.N_LABELS_PHI,
        -0.5,
        EtrackDataset.N_LABELS_PHI - 0.5,
    )
    th1_dist_like_phi = TH1D(
        "th1_dist_like_phi", "Distribution of Likelihood of Phi", 100, 0.0, 1.0
    )
    th2_dphi_vs_maxlike = TH2D(
        "th2_dphi_vs_maxlike",
        ";#Delta#phi[deg];Maximum of Likelihood Map",
        128,
        -180.0,
        180.0,
        10,
        0.0,
        1.0,
    )
    th2_dphi_vs_prom_like = TH2D(
        "th2_dphi_vs_prom_like",
        ";#Delta#phi[deg];Dev(Maximum Likelihood)/#sigma",
        128,
        -180.0,
        180.0,
        20,
        -0.25,
        10.0 - 0.25,
    )
    th2_dphi_vs_rms_of_map = TH2D(
        "th2_dphi_vs_rms_of_map",
        ";#Delta#phi[deg];RMS of Likelihood Map",
        128,
        -180.0,
        180.0,
        20,
        -0.25,
        10.0 - 0.25,
    )

    true_pos_map = TH2D(
        "true_pos_map",
        "true_pos_map;X;Y",
        EtrackDataset.N_PIXELS_1D,
        0,
        EtrackDataset.N_PIXELS_1D,
        EtrackDataset.N_PIXELS_1D,
        0,
        EtrackDataset.N_PIXELS_1D,
    )
    est_pos_map = TH2D(
        "est_pos_map",
        "est_pos_map;X;Y",
        EtrackDataset.N_PIXELS_1D,
        0,
        EtrackDataset.N_PIXELS_1D,
        EtrackDataset.N_PIXELS_1D,
        0,
        EtrackDataset.N_PIXELS_1D,
    )
    true_phi_map = TH1D(
        "true_phi_map",
        "true_phi_map;Phi (deg.);Counts",
        EtrackDataset.N_LABELS_PHI,
        -180,
        180,
    )
    est_phi_map = TH1D(
        "est_phi_map",
        "est_phi_map;Phi (deg.);Counts",
        EtrackDataset.N_LABELS_PHI,
        -180,
        180,
    )
    # true_cos_beta_map = ROOT.TH1D( "true_cos_beta_map", "true_cos_beta_map;cos #beta;Counts", 1, 0, 1 )
    # est_cos_beta_map = ROOT.TH1D( "est_cos_beta_map", "est_cos_beta_map;cos #beta;Counts", 1, 0, 1 )

    th2_image = TH2D("th2_image", "th2_image", 32, -0.5, 32 - 0.5, 32, -0.5, 32 - 0.5)

    gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")
    c = TCanvas("c", "c", 800, 400)
    c.Divide(2, 1)
    gStyle.SetOptStat("miren")
    pdfname = outfilename.replace(".root", ".pdf")

    with torch.no_grad():

        for ievent in range(nevents):

            if ievent % 10 == 0:
                print(f"\r{ievent}/{nevents}", end="")

            image, label = testdata[ievent]
            inputs = (
                torch.tensor(image)
                .view(-1, 1, EtrackDataset.N_PIXELS_1D, EtrackDataset.N_PIXELS_1D)
                .float()
            )

            if args.use_gpu:
                inputs = inputs.to(args.cuda)

            true_phi_deg = testdata.getPhi(ievent)
            true_ini_pos_x, true_ini_pos_y = testdata.getIniPos(ievent)
            # true_cos_beta = testdata.getCosBeta(ievent)

            outputs = net(inputs)
            outputs_numpy = outputs.to("cpu").detach().numpy().copy()

            pos_map, phi_map = EtrackDataset.split(outputs_numpy[0])
            pos_map = pos_map.reshape(EtrackDataset.N_PIXELS_SHAPE)
            phi_map = phi_map.reshape(EtrackDataset.N_LABELS_SHAPE)

            max_likeli_pos = numpy.max(pos_map)
            max_likeli_phi = numpy.max(phi_map)
            max_prob_pos_idx = numpy.unravel_index(numpy.argmax(pos_map), pos_map.shape)
            max_prob_phi_idx = numpy.unravel_index(numpy.argmax(phi_map), phi_map.shape)
            # max_prob_phi_cos_beta_idx = numpy.unravel_index(numpy.argmax(phi_cos_beta_map), phi_cos_beta_map.shape)

            max_prob_pos = pos_map[max_prob_pos_idx]
            max_prob_phi = phi_map[max_prob_phi_idx]
            # max_prob_phi_cos_beta = phi_cos_beta_map[max_prob_phi_cos_beta_idx]

            est_ini_pos_x = EtrackDataset.index_to_pos(max_prob_pos_idx[1])
            est_ini_pos_y = EtrackDataset.index_to_pos(max_prob_pos_idx[0])
            est_phi_deg = EtrackDataset.index_to_phi_deg(max_prob_phi_idx[0])

            diff_pos_x = est_ini_pos_x - true_ini_pos_x
            diff_pos_y = est_ini_pos_y - true_ini_pos_y
            diff_phi = est_phi_deg - true_phi_deg

            th2_diff_pos.Fill(diff_pos_x, diff_pos_y)
            th1_diff_phi.Fill(diff_phi)

            th2_dphi_vs_maxlike.Fill(diff_phi, max_likeli_phi)
            # diff_cos_beta.Fill( est_cos_beta - true_cos_beta)
            # diff_phi_2d.Fill( est_phi_deg - true_phi_deg, true_cos_beta)
            # diff_cos_beta_2d.Fill( est_cos_beta - true_cos_beta, true_cos_beta)

            th1_maxlike_phi.Reset()
            th1_dist_like_phi.Reset()
            for index in range(len(phi_map)):
                likeli = phi_map[index]
                th1_maxlike_phi.Fill(index, likeli)
                th1_dist_like_phi.Fill(likeli)

            rms = th1_dist_like_phi.GetRMS()
            mean = th1_dist_like_phi.GetMean()
            div = abs(max_likeli_phi - mean)
            rms_of_map = th1_maxlike_phi.GetRMS()

            th1_maxlike_phi.SetTitle(f"Likelihood Map of Phi: dx/rms={div/rms:6.3f}")
            th1_dist_like_phi.SetTitle(
                f"Distribution of Likelihood of Phi: dx/rms={div:5.3f}/{rms:5.3f}"
            )
            th2_dphi_vs_prom_like.Fill(diff_phi, div / rms)
            th2_dphi_vs_rms_of_map.Fill(diff_phi, rms_of_map)

            if div <= 4 * rms:
                th1_diff_phi_nice.Fill(diff_phi)

            th2_image.Reset()
            n_filled = 0

            for i in range(32):
                if numpy.all(image[0][i] == 0.0):
                    continue
                for j in range(32):
                    if image[0][i][j] == 0.0:
                        continue
                    th2_image.Fill(i, j, image[0][i][j])
                    n_filled += 1

            if n_filled < 5 and ievent != nevents - 1 and ievent != 0:
                continue

            c.cd(1)
            th1_maxlike_phi.Draw("hist")
            gStyle.SetOptStat("miren")
            c.cd(2)
            th2_image.Draw("colz")
            th2_image.GetXaxis().SetRangeUser(10, 25)
            th2_image.GetYaxis().SetRangeUser(10, 25)
            # th1_dist_like_phi.Draw( 'hist' )

            if ievent == 0 and 1 < nevents:
                c.SaveAs(pdfname + "(")
            elif ievent == nevents - 1 and 1 < nevents:
                c.SaveAs(pdfname + ")")
            else:
                c.SaveAs(pdfname)

            true_pos_map.Fill(true_ini_pos_x, true_ini_pos_y)
            est_pos_map.Fill(est_ini_pos_x, est_ini_pos_y)
            true_phi_map.Fill(true_phi_deg)
            est_phi_map.Fill(est_phi_deg)
            # true_cos_beta_map.Fill(true_cos_beta)
            # est_cos_beta_map.Fill(est_cos_beta)

            # corr_cos_beta_numpixnonzero.Fill(est_cos_beta, testdata.getNumNonZeroPix(ievent))

        print("\r")

    outfile.cd()
    th2_diff_pos.Write()
    th1_diff_phi.Write()

    th1_diff_phi_nice.Write()
    th2_dphi_vs_maxlike.Write()
    th2_dphi_vs_prom_like.Write()
    th2_dphi_vs_rms_of_map.Write()

    # diff_cos_beta.Write()
    # diff_phi_2d.Write()
    # diff_cos_beta_2d.Write()
    true_pos_map.Write()
    est_pos_map.Write()
    true_phi_map.Write()
    est_phi_map.Write()
    # true_cos_beta_map.Write()
    # est_cos_beta_map.Write()
    # corr_cos_beta_numpixnonzero.Write()
    outfile.Close()


def main(input_file_name, args):

    base = os.path.basename(input_file_name)
    name, ext = os.path.splitext(base)

    if ext != ".pth":
        print("Input file is not .pth file")
        return

    output_file_name = f"{args.outdir}/acc_{name}.root"
    os.makedirs(args.outdir, exist_ok=True)

    estimation(input_file_name, output_file_name, args.csv, args)


if __name__ == "__main__":

    parser = usage()
    args = parser.parse_args()

    for file in args.file:
        main(file, args)

    # main()
