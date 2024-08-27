import numpy as np
import os
from curses.ascii import isdigit

from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from unet_modeling import UNet
from dataset_preprocessing import Dataset

# wandb 초기화
wandb.init(project="UNet_segmentation")

lr = 1e-3  # learning rate
batch_size = 4
num_epoch = 100

data_dir = '/Users/user/Desktop/UNet_segmentation/data'
ckpt_dir = '/Users/user/Desktop/UNet_segmentation/checkpoint'
test_dir = '/Users/user/Desktop/UNet_segmentation/data/test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb 설정
wandb.config.update({
    "learning_rate": lr,
    "batch_size": batch_size,
    "num_epoch": num_epoch
}, allow_val_change=True)

# transform 적용해서 데이터 셋 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),  # numpy 배열을 torch 텐서로 변환
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
])

# Dataset 정의
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
dataset_test = Dataset(data_dir=test_dir, transform=transform)

# DataLoader 정의
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

# 네트워크 불러오기
net = UNet().to(device)  # device : cpu or gpu

# loss function 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, f'{ckpt_dir}/model_epoch{epoch}.pth')

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
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# 성능 평가
def dice_coefficient(pred, target, smooth=1e-6):
    # 예측 마스크와 실제 마스크를 이진화
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def mean_iou(pred, target, threshold=0.5, smooth=1e-6):

    # 예측 마스크를 이진화
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()  # 교집합
    union = pred.sum() + target.sum() - intersection  # 합집합

    iou = (intersection + smooth) / (union + smooth)
    return iou  # tensor에서 숫자형 값으로 변환


start_epoch = 0
net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# 에포크에 대한 step을 초기화
global_step = start_epoch

for epoch in range(start_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []
    iou_arr = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        inputs = data['input'].to(device)

        output = net(inputs)
        output_sigmoid = torch.sigmoid(output)

        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr.append(loss.item())
        iou = mean_iou(output_sigmoid, label)
        iou_arr.append(iou)

    mean_loss = np.mean(loss_arr)
    mean_iou_value = np.mean(iou_arr)
    print(f"Epoch {epoch} - Training Loss: {mean_loss} - Mean IoU: {mean_iou_value}")

    with torch.no_grad():
        net.eval()
        val_loss_arr = []
        val_iou_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)
            output_sigmoid = torch.sigmoid(output)

            loss = fn_loss(output, label)
            val_loss_arr.append(loss.item())
            iou = mean_iou(output_sigmoid, label)
            val_iou_arr.append(iou)

        mean_val_loss = np.mean(val_loss_arr)
        mean_val_iou = np.mean(val_iou_arr)
        print(f"Epoch {epoch} - Validation Loss: {mean_val_loss} - Mean IoU: {mean_val_iou}")

    # `wandb.log`에 `step`을 `global_step`으로 설정
    wandb.log({
        "Training Loss": mean_loss,
        "Validation Loss": mean_val_loss,
        "Training Mean IoU": mean_iou_value,
        "Validation Mean IoU": mean_val_iou,
    }, step=global_step)

    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    # 에포크가 끝날 때마다 step 증가
    global_step += 1


print("Evaluating on test set...")
net.eval()
with torch.no_grad():
    test_input_images = []
    test_true_masks = []
    test_predicted_masks = []
    test_iou_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        inputs = data['input'].to(device)
        output = net(inputs)
        output_sigmoid = torch.sigmoid(output)

        input_image = inputs[0].cpu().detach().numpy().transpose(1, 2, 0)
        true_mask = label[0].cpu().detach().numpy().transpose(1, 2, 0)
        predicted_mask = output_sigmoid[0].cpu().detach().numpy().transpose(1, 2, 0)

        # 시각화를 위해 0-1 범위로 클리핑
        input_image = np.clip(input_image, 0, 1)
        predicted_mask = np.clip(predicted_mask, 0, 1)

        # 테스트 이미지 저장
        test_input_images.append(input_image)
        test_true_masks.append(true_mask)
        test_predicted_masks.append(predicted_mask)

        # Mean IoU 계산
        iou = mean_iou(output_sigmoid, label)
        test_iou_arr.append(iou.item())

    mean_test_iou = np.mean(test_iou_arr)
    print(f"Test Mean IoU: {mean_test_iou}")

    # 테스트 결과 wandb에 기록
    wandb.log({
        "Test Input Image": [wandb.Image(img, caption="Test Input Image") for img in test_input_images],
        "Test True Mask": [wandb.Image(mask, caption="Test True Mask") for mask in test_true_masks],
        "Test Predicted Mask": [wandb.Image(mask, caption="Test Predicted Mask") for mask in test_predicted_masks],
        "Test Mean IoU": mean_test_iou
    })

wandb.finish()
print("done!")
