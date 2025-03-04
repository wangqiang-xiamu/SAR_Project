# import torch
# print(torch.cuda.is_available())  # 如果返回 False，说明 PyTorch 不支持 CUDA
# print(torch.version.cuda)         # 查看 CUDA 版本
#
# import torch
# print(torch.backends.mps.is_available())  # 返回 True 表示 MPS 可用
# print(torch.backends.mps.is_built())      # 返回 True 表示 PyTorch 支持 MPS

#
# from torch.utils.tensorboard import SummaryWriter
# import os
#
# # 设置日志目录
# log_dir = "results/cifar10@4000.5"
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir)
#
# # 写入数据
# for epoch in range(10):
#     writer.add_scalar('train/loss', 0.1 * epoch, epoch)
#     print(f"Logged train loss: {0.1 * epoch} at epoch {epoch}")
#
# # 关闭 writer
# writer.close()

import os
from tensorboard.backend.event_processing import event_file_loader

log_dir = "/Users/xiamu/PycharmProjects/SAR_Project/moudle/FixMatch-pytorch-master/results/cifar10@4000.5"

for file_name in os.listdir(log_dir):
    if file_name.startswith("events.out.tfevents"):
        event_file_path = os.path.join(log_dir, file_name)
        print(f"Checking event file: {event_file_path}")
        try:
            for event in event_file_loader.EventFileLoader(event_file_path).Load():
                print(event)
        except Exception as e:
            print(f"Error loading event file {event_file_path}: {e}")