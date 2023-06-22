import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
import torchmetrics

from model.mobilenet import mobilenetv3
from data.data_utils import *


def parser_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--log_name", type=str, default="log")
    parser.add_argument("--print_freq", type=int, default=10)
    return parser


def train_one_epoch(model: nn.Module, 
                    data_loader: DataLoader, 
                    optimizer: Optimizer, 
                    criterion: nn.Module, 
                    device: torch.device, 
                    epoch: int,
                    print_freq: int = 10):

    model.train()
    optimizer.zero_grad()

    time_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    end = time.time()
    iters = len(data_loader)
    for batch_id, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        acc = (outputs.argmax(dim=1) == labels).sum().item() / batch_size
        time_meter.update(time.time() - end)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        if batch_id % print_freq == 0:
            print(f"[epoch:{epoch}] {batch_id + 1}/{iters} acc:{acc:.4f} loss:{loss.item():.4f} time:{(time.time() - end):.2f}")
        end = time.time()

    return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    recall = torchmetrics.Recall(task='binary', average=None, num_classes=2).to(device)
    precision = torchmetrics.Precision(task='binary', average=None, num_classes=2).to(device)

    for imgs, labels in data_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        labels_pred = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        acc = (labels_pred == labels).sum().item() / batch_size
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        recall(labels_pred, labels)
        precision(labels_pred, labels)

    recall = recall.compute()
    precision = precision.compute()
    print("recall", recall)
    print("precision", precision)

    return acc_meter.avg, loss_meter.avg


if __name__ == '__main__':
    parser = parser_option()
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    print("Building data loader...")
    data_loader_train = get_dataloader(args.data_dir, "data/public_train.json", args.batch_size, is_train=True)
    data_loader_valid = get_dataloader(args.data_dir, "data/public_valid.json", args.batch_size, is_train=False)
    data_loader_test = get_dataloader(args.data_dir, "data/public_test.json", args.batch_size, is_train=False)

    print("Creating model...")
    model = mobilenetv3()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=os.path.join('runs', args.log_name), comment=args.log_name)

    for epoch in range(args.epochs):
        train_acc, train_loss = train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, args.print_freq)
        valid_acc, valid_loss = evaluate(model, data_loader_valid, criterion, device)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/acc", valid_acc, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)