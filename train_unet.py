import numpy as np
import os
from curses.ascii import isdigit

from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet_modeling import UNet
from dataset_preprocessing import Dataset

lr = 1e-3  # learning rate
batch_size = 4
num_epoch = 100

data_dir = '/Users/user/Desktop/UNet_segmentation/data'
ckpt_dir = '/Users/user/Desktop/UNet_segmentation/checkpoint'
log_dir = '/Users/user/Desktop/UNet_segmentation/log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform 적용해서 데이터 셋 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),  # numpy 배열을 torch 텐서로 변환
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
])

# Dataset 정의: __getitem__ 메서드가 수정된 Dataset 클래스 사용
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)

# CustomDataset 클래스의 인스턴스 생성
dataset = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)

# 인덱스 0의 데이터 샘플 가져오기
sample = dataset[0]
# print(sample)

# 데이터 형태 출력
# print("Sample input shape:", sample['input'].shape)  # 예: (1, 512, 512)
# print("Sample label shape:", sample['label'].shape)  # 예: (1, 512, 512)

# DataLoader 정의
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

# 데이터 로드 확인
data_iter = iter(loader_train)
sample_batch = next(data_iter)

input_image = sample_batch['input']
label_image = sample_batch['label']
#
# print("After Transform - Input shape:", input_image.shape)  # 예상: torch.Size([4, 1, 512, 512])
# print("After Transform - Label shape:", label_image.shape)  # 예상: torch.Size([4, 1, 512, 512])
# print("After Transform - Input dtype:", input_image.dtype)  # 예상: torch.float32
# print("After Transform - Label dtype:", label_image.dtype)  # 예상: torch.float32

# 네트워크 불러오기
net = UNet().to(device)  # device : cpu or gpu

# loss function 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 데이터셋 크기 계산
num_train = len(dataset_train)
num_val = len(dataset_val)

# 에폭 당 배치 수 계산
num_train_for_epoch = np.ceil(num_train / batch_size)
num_val_for_epoch = np.ceil(num_val / batch_size)

# 기타 function 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_classifier = lambda x: 1.0 * (x > 0.5)

# Tensorboard 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):  # 디렉토리가 없는 경우
        os.makedirs(ckpt_dir)
        print(f"Checkpoint directory '{ckpt_dir}' created.")
        return net, optim, 0  # 초기 에폭으로 반환

    ckpt_lst = os.listdir(ckpt_dir)

    if len(ckpt_lst) == 0:  # 체크포인트 파일이 없는 경우
        print(f"No checkpoint found in '{ckpt_dir}'. Starting from scratch.")
        return net, optim, 0

    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), weights_only=True)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


# 네트워크 학습시키기
start_epoch = 0
net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(start_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        inputs = data['input'].to(device)

        # 모델에 전달하기 전 데이터 모양 출력
        # print(f"Epoch: {epoch}, Batch: {batch}")
        # print(f"Input shape: {inputs.shape}, Label shape: {label.shape}")

        output = net(inputs)

        # 첫 번째 배치에서만 출력 확인
        # if batch == 1:
        #     print(f"Model output (first batch): {output[0]}")  # 첫 번째 샘플 출력

        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        # 학습 중 손실 값 확인
        if batch % 10 == 0:  # 매 10번째 배치마다 출력
            print(f"Epoch {epoch}, Batch {batch} - Training Loss: {loss.item()}")

    # 에폭이 끝날 때 평균 손실 값 출력
    mean_loss = np.mean(loss_arr)
    print(f"Epoch {epoch} - Training Loss: {mean_loss}")
    writer_train.add_scalar('Training Loss', mean_loss, epoch)  # TensorBoard에 기록

    # validation
    with torch.no_grad():
        net.eval()
        val_loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            loss = fn_loss(output, label)
            val_loss_arr.append(loss.item())  # 검증 손실 값을 리스트에 추가

            # print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04f' % (
            #     epoch, num_epoch, batch, num_val_for_epoch, np.mean(loss_arr)))

        mean_val_loss = np.mean(val_loss_arr)
        print(f"Epoch {epoch} - Validation Loss: {mean_val_loss}")
        writer_val.add_scalar('Validation Loss', mean_val_loss, epoch)  # TensorBoard에 기록

    # 체크포인트 저장
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()
