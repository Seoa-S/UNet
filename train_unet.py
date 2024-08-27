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

# 네트워크 학습시키기
start_epoch = 0
net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(start_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    # 에폭마다 시각화할 이미지를 저장할 리스트 초기화
    input_images = []
    true_masks = []
    predicted_masks = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        inputs = data['input'].to(device)

        output = net(inputs)

        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr.append(loss.item())

        # 첫 번째 배치의 이미지를 저장
        if batch == 1:
            input_image = inputs[0].cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            true_mask = label[0].cpu().detach().numpy().transpose(1, 2, 0)
            predicted_mask = torch.sigmoid(output[0]).cpu().detach().numpy().transpose(1, 2, 0)

            # 시각화를 위해 0-1 범위로 클리핑
            input_image = np.clip(input_image, 0, 1)
            predicted_mask = np.clip(predicted_mask, 0, 1)

            # 이미지를 리스트에 추가
            input_images.append(input_image)
            true_masks.append(true_mask)
            predicted_masks.append(predicted_mask)

    mean_loss = np.mean(loss_arr)
    print(f"Epoch {epoch} - Training Loss: {mean_loss}")

    # Validation
    with torch.no_grad():
        net.eval()
        val_loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            loss = fn_loss(output, label)
            val_loss_arr.append(loss.item())

        mean_val_loss = np.mean(val_loss_arr)
        print(f"Epoch {epoch} - Validation Loss: {mean_val_loss}")

    # 에폭이 끝날 때 wandb에 한 번 로그 기록
    wandb.log({
        "Training Loss": mean_loss,
        "Validation Loss": mean_val_loss,
        "Input Image": [wandb.Image(img, caption="Input Image") for img in input_images],
        "True Mask": [wandb.Image(mask, caption="True Mask") for mask in true_masks],
        "Predicted Mask": [wandb.Image(mask, caption="Predicted Mask") for mask in predicted_masks]
    }, step=epoch)

    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

print("Evaluating on test set...")
net.eval()
with torch.no_grad():
    test_input_images = []
    test_true_masks = []
    test_predicted_masks = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        inputs = data['input'].to(device)
        output = net(inputs)

        input_image = inputs[0].cpu().detach().numpy().transpose(1, 2, 0)
        true_mask = label[0].cpu().detach().numpy().transpose(1, 2, 0)
        predicted_mask = torch.sigmoid(output[0]).cpu().detach().numpy().transpose(1, 2, 0)

        # 시각화를 위해 0-1 범위로 클리핑
        input_image = np.clip(input_image, 0, 1)
        predicted_mask = np.clip(predicted_mask, 0, 1)

        # 테스트 이미지 저장
        test_input_images.append(input_image)
        test_true_masks.append(true_mask)
        test_predicted_masks.append(predicted_mask)

    # 테스트 결과 wandb에 기록
    wandb.log({
        "Test Input Image": [wandb.Image(img, caption="Test Input Image") for img in test_input_images],
        "Test True Mask": [wandb.Image(mask, caption="Test True Mask") for mask in test_true_masks],
        "Test Predicted Mask": [wandb.Image(mask, caption="Test Predicted Mask") for mask in test_predicted_masks]
    })


wandb.finish()
print("done!")