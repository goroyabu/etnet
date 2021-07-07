#!/usr/bin/env python3

import numpy
import pandas
import torch


class EtrackDataset(torch.utils.data.Dataset):

    # DATA format
    N_PIXELS_1D = 32
    N_PIXELS_2D = N_PIXELS_1D * N_PIXELS_1D
    N_PIXELS_SHAPE = [N_PIXELS_1D, N_PIXELS_1D]

    # TARGET format
    N_LABELS_PHI = 36
    N_LABELS_SHAPE = [N_LABELS_PHI]

    # parameters
    RATE_FOR_TEST = 0.3

    def __init__(self, data_path=None, train=True, transform=None, sigmazero=False):

        self.data_path = data_path
        self.is_for_training = train
        self.transform = transform
        self.sigma_zero = sigmazero

        if self.is_for_training == True:
            print("- EtrackDataset for Training-")
        else :
            print("- EtrackDataset for Test -")
        print("Data format   :", EtrackDataset.N_PIXELS_SHAPE)
        print("Target format :", EtrackDataset.N_LABELS_SHAPE)

        self.data = pandas.read_csv(self.data_path)
        self.data_num = len(self.data)
        print("Number of all data in", self.data_path, "is", self.data_num)

        test_index = int(self.data_num * EtrackDataset.RATE_FOR_TEST)
        if self.is_for_training == True:
            self.data = self.data[test_index : self.data_num]
            self.data_num = len(self.data)
            print("Number of data for training :", self.data_num)
        else:
            self.data = self.data[0:test_index]
            self.data_num = len(self.data)
            print("Number of data for test :", self.data_num)
        print("")

        # pixel_array = ["index{}".format(i) for i in range(EtrackDataset.N_PIXELS_2D)]
        
        self.eventID = self.data[["eventID"]].values
        self.image = self.data[
            ["index{}".format(i) for i in range(EtrackDataset.N_PIXELS_2D)]
        ].values
        self.ini_pos_norm = self.data[["ini_pos_x_norm", "ini_pos_y_norm"]].values
        self.ini_phi_norm = self.data[["ini_phi_norm"]].values

        self.ini_pos_map = numpy.zeros(
            [
                self.data_num,
                EtrackDataset.N_PIXELS_1D,
                EtrackDataset.N_PIXELS_1D,
            ]
        )
        self.ini_phi_map = numpy.zeros([self.data_num, EtrackDataset.N_LABELS_PHI])

        # self.ini_cos_beta = self.data[['ini_cos_beta']].values
        # self.ini_sin_beta = self.data[['ini_sin_beta']].values
        # self.ini_beta_norm = self.data[['ini_beta_norm']].values
        # self.ini_phi_cos_beta_map = numpy.zeros([self.data_num, nBinPhi, nBinCos])

        for i_num in range(self.data_num):

            ini_pos_x_norm = self.ini_pos_norm[i_num][0]
            ini_pos_y_norm = self.ini_pos_norm[i_num][1]

            index_pos_map_x = EtrackDataset.pos_norm_to_index(ini_pos_x_norm)
            index_pos_map_y = EtrackDataset.pos_norm_to_index(ini_pos_y_norm)
            self.ini_pos_map[i_num][index_pos_map_y][index_pos_map_x] = 1.0

            # phi_normalized = 0.5 * (self.ini_phi_norm[i_num] + 1)  # [-1,1] -> [0,1]
            # index_phi = EtrackDataset.phi_norm_to_index(phi_normalized)
            ini_phi_norm = self.ini_phi_norm[i_num]
            index_phi = EtrackDataset.phi_norm_to_index(ini_phi_norm)
            self.ini_phi_map[i_num][index_phi] = 1.0

        self.ini_pos_map = self.ini_pos_map.reshape(-1, EtrackDataset.N_PIXELS_2D)
        self.ini_phi_map = self.ini_phi_map.reshape(-1, EtrackDataset.N_LABELS_PHI)
        self.conc_map = numpy.concatenate([self.ini_pos_map, self.ini_phi_map], 1)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.image[idx].reshape(
            -1, EtrackDataset.N_PIXELS_1D, EtrackDataset.N_PIXELS_1D
        )
        out_label = self.conc_map[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

    def getPhi(self, idx):
        return self.ini_phi_norm[idx][0]

    # def getPhiDeg(self, idx):
    #     return self.ini_phi_norm[idx][0] * 180

    # def getCosBeta(self, idx):
    #     return self.ini_cos_beta[idx][0]
    #
    # def getSinBeta(self, idx):
    #     return self.ini_sin_beta[idx][0]
    #
    # def getBetaDeg(self, idx):
    #     return self.ini_beta_norm[idx][0] * 90

    def getIniPos(self, idx):
        return self.ini_pos_norm[idx][0], self.ini_pos_norm[idx][1]

    # def getIniPosPix(self, idx):
    #     # return self.ini_pos_norm[idx][0] * nPix, self.ini_pos_norm[idx][1] * nPix
    #     x = self.ini_pos_norm[idx][0] * EtrackDataset.N_PIXELS_1D
    #     y = self.ini_pos_norm[idx][1] * EtrackDataset.N_PIXELS_1D
    #     return x, y

    # def getNumNonZeroPix(self, idx):
    #     return numpy.count_nonzero(self.image[idx] > 0)

    # def getEventID(self, idx):
    #     return self.eventID[idx][0]

    @classmethod
    def index_to_pos(cls, index):  # [0,N_PIXELS_1D-1] -> [-1,1]
        if cls.N_PIXELS_1D - 1 <= index:
            index = cls.N_PIXELS_1D - 1
        elif index < 0:
            index = 0
        pos = float((index + 0.5) / cls.N_PIXELS_1D) * 2.0 - 1.0
        return pos
        # return index + 0.5

    @classmethod
    def pos_norm_to_index(cls, pos):  # [-1,1] -> [0,N_PIXELS_1D-1]
        index = int((pos + 1) * 0.5 * (cls.N_PIXELS_1D))
        # index = int(pos * cls.N_PIXELS_1D)
        if cls.N_PIXELS_1D - 1 <= index:
            return cls.N_PIXELS_1D - 1
        elif index < 0:
            return 0
        return index

    @classmethod
    def index_to_phi_deg(cls, index):  # [0,N_LABELS_PHI-1] -> [-180.,180.]
        if cls.N_LABELS_PHI - 1 <= index:
            index = cls.N_LABELS_PHI - 1
        elif index < 0:
            index = 0
        return (0.5 + index) * 360.0 / cls.N_LABELS_PHI - 180.0

    @classmethod
    def phi_norm_to_index(cls, phi):  # [-180.,180.] -> [0,N_LABELS_PHI-1]
        index = int((phi / 180.0 + 1) * 0.5 * cls.N_LABELS_PHI)
        # index = int(phi * cls.N_LABELS_PHI)
        if cls.N_LABELS_PHI - 1 <= index:
            return cls.N_LABELS_PHI - 1
        elif index < 0:
            return 0
        return index

    @classmethod
    def custom_loss(cls, outputs, targets, beta, mu):

        outputs = outputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        outputs_pos_map, outputs_phi_map = cls.split(outputs)
        targets_pos_map, targets_phi_map = cls.split(targets)

        zero = 1e-20
        cross_entropy_pos = -cls.N_PIXELS_2D * torch.mean(
            targets_pos_map * torch.log(outputs_pos_map + zero)
        )
        cross_entropy_phi = -cls.N_LABELS_PHI * torch.mean(
            targets_phi_map * torch.log(outputs_phi_map + zero)
        )

        loss = cross_entropy_pos + cross_entropy_phi
        return loss, cross_entropy_pos, cross_entropy_phi

    @classmethod
    def split(cls, data):
        pos = data[0 : cls.N_PIXELS_2D]
        phi = data[cls.N_PIXELS_2D : cls.N_PIXELS_2D + cls.N_LABELS_PHI]
        return pos, phi
