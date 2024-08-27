import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.lst_input = sorted([f for f in os.listdir(data_dir) if 'input' in f])
        self.lst_label = sorted([f for f in os.listdir(data_dir) if 'label' in f])

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화
        label = label / 255.0
        inputs = inputs / 255.0

        # 데이터 타입 설정
        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)

        # (H, W) 형태에서 채널 축 추가 -> (H, W, C)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        # transform 적용 (PIL 이미지 또는 numpy 배열을 처리할 수 있도록 설정)
        if self.transform:
            inputs = self.transform(inputs)  # ToTensor는 (H, W, C) -> (C, H, W) 자동 변경
            label = self.transform(label)

        data = {'input': inputs, 'label': label}
        return data



class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpy와 tensor의 배열 차원 순서가 다르다.
        # numpy : (행, 열, 채널)
        # tensor : (채널, 행, 열)
        # 따라서 위 순서에 맞춰 transpose

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # 이후 np를 tensor로 바꾸는 코드는 다음과 같이 간단하다.
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data