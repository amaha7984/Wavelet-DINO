import os
import torch
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world_size)

def save_checkpoint(model, path, epoch, optimizer, best_acc):
    torch.save({
        "epoch": epoch + 1,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
    }, path)
