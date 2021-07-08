#!/usr/bin/env python3

import os
import argparse
import math

import numpy
import ROOT

from etnet.dataset import EtrackDataset


def usage():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options] FILE ...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "file",
        metavar="FILE",
        type=str,
        nargs="+",
        help="name of g4tree file",
    )

    parser.add_argument(
        "--outdir",
        default="csv",
    )

    parser.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    return parser


def analysis(filename, args):

    infile = ROOT.TFile(filename, "read")
    if infile is None or infile.IsZombie():
        print(filename, "is None")
        return -1

    intree = infile.Get("g4tree")
    if intree is None:
        print("g4tree is None")
        return -1

    outname = args.outdir + "/" + os.path.basename(filename)
    outname = outname.replace(".root", ".csv")
    outfile = open(outname, "w")
    # print( outname )

    intree.GetEntry(0)
    nentries = intree.GetEntries()
    # print(nentries)

    n_pixels_1d = EtrackDataset.N_PIXELS_1D  # or 32
    pixel_pitch = 0.02  # mm, fixed
    image_width = n_pixels_1d * pixel_pitch

    current_nimages = 0

    pixel_index = range(n_pixels_1d * n_pixels_1d)
    str_index = "eventID"
    for index in pixel_index:
        str_index += f",index{index}"
    str_index += "," + ",".join(
        [
            "ini_pos_x_norm",
            "ini_pos_y_norm",
            "ini_phi_norm",
            "ini_cos_beta",
            "ini_sin_beta",
            "ini_beta_norm",
        ]
    )
    print(str_index, file=outfile)

    for entry in range(nentries):
        intree.GetEntry(entry)
        if entry % 1000 == 0:
            print(f"\r{entry}/{nentries}", end="")

        if image_width <= intree.max_cmos_detx - intree.min_cmos_detx:
            continue  # image width < x-width of track
        if image_width <= intree.max_cmos_dety - intree.max_cmos_dety:
            continue  # image width < y-width of track

        ini_pos_x_norm = intree.init_pos_cmos_detx - intree.min_cmos_detx
        ini_pos_y_norm = intree.init_pos_cmos_dety - intree.min_cmos_dety
        ini_pos_x_norm /= image_width  # [0,1]
        ini_pos_y_norm /= image_width  # [0,1]

        ini_phi_norm = intree.phi / math.pi * 180.0 - 180.0  # [0,2pi]->[-180,180]
        ini_beta_norm = intree.beta / math.pi * 180.0  # [-pi/2,pi/2]->[-90,90]

        image_array = numpy.zeros([n_pixels_1d, n_pixels_1d])
        image_tree = intree.image

        xbin_begin = image_tree.GetXaxis().FindBin(intree.min_cmos_detx)
        xbin_end = image_tree.GetXaxis().FindBin(intree.max_cmos_detx)
        ybin_begin = image_tree.GetYaxis().FindBin(intree.min_cmos_dety)
        ybin_end = image_tree.GetYaxis().FindBin(intree.max_cmos_dety)

        if n_pixels_1d < xbin_end - xbin_begin:
            continue
        if n_pixels_1d < ybin_end - ybin_begin:
            continue

        for xbin in range(xbin_begin, xbin_end):
            for ybin in range(ybin_begin, ybin_end):
                gbin = image_tree.GetBin(xbin, ybin)

                if image_tree.IsBinOverflow(gbin):
                    continue
                if image_tree.IsBinUnderflow(gbin):
                    continue
                content = image_tree.GetBinContent(gbin)
                if content <= 0.0:
                    continue

                image_array[xbin - xbin_begin][ybin - ybin_begin] = content

        print(
            current_nimages,
            *image_array.flatten(),
            ini_pos_x_norm,
            ini_pos_y_norm,
            ini_phi_norm,
            ini_beta_norm,
            sep=",",
            file=outfile,
        )
        current_nimages += 1

    print(f"\r{nentries}/{nentries}")


if __name__ == "__main__":

    parser = usage()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for file in args.file:
        analysis(file, args)
