#!/usr/bin/env python
# coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as functional
from .dataset import EtrackDataset


class EtrackNet(nn.Module):
    def __init__(self, train=True):
        super(EtrackNet, self).__init__()

        self.is_for_training = train
        self.p = 0.5  # ratio of dropout

        self.conv1a = nn.Conv2d(1, 16, 5, padding=2)
        self.conv1b = nn.Conv2d(16, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2)

        self.conv2a = nn.Conv2d(16, 32, 5, padding=2)
        self.conv2b = nn.Conv2d(32, 32, 5, padding=2)

        self.conv3a = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3b = nn.Conv2d(64, 64, 5, padding=2)

        self.fc1 = nn.Linear(int(EtrackDataset.N_PIXELS_1D / 8) ** 2 * 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3_pos_map = nn.Linear(512, EtrackDataset.N_PIXELS_2D)
        self.fc3_phi_map = nn.Linear(512, EtrackDataset.N_LABELS_PHI)

        self.dropout = torch.nn.Dropout(p=self.p)

        self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3_pos_map.weight)
        nn.init.kaiming_normal_(self.fc3_phi_map.weight)
        # nn.init.kaiming_normal_(self.fc3_phi_cos_beta_map.weight)

    def forward(self, x):

        x = functional.relu(self.conv1a(x))
        x = functional.relu(self.conv1b(x))
        x = self.pool(x)

        x = functional.relu(self.conv2a(x))
        x = functional.relu(self.conv2b(x))
        x = self.pool(x)

        x = functional.relu(self.conv3a(x))
        x = functional.relu(self.conv3b(x))
        x = self.pool(x)

        x = x.view(x.size()[0], -1)

        x = functional.relu(self.fc1(x))
        if self.train and self.p > 0:
            x = self.dropout(x)

        x = functional.relu(self.fc2(x))
        if self.train and self.p > 0:
            x = self.dropout(x)

        x_pos_map = self.softmax(self.fc3_pos_map(x))
        x_phi_map = self.softmax(self.fc3_phi_map(x))
        x = torch.cat([x_pos_map, x_phi_map], dim=1)
        # x_phi_cos_beta_map = self.softmax( self.fc3_phi_cos_beta_map(x) )
        # x = torch.cat([x_pos_map, x_phi_cos_beta_map], dim=1)
        return x
