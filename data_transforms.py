import numpy as np
from safetensors import torch


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