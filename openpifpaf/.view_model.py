import torch
import openpifpaf.network as network

checkpoint = "resnet50"

network.Factory.checkpoint = checkpoint

model_cpu, _ = network.Factory().factory(head_metas=None)


torch.save(model_cpu.head_nets[0].conv.state_dict(), "wat.pth")
torch.save(model_cpu.base_net.state_dict(), "resnet_wat.pth")

print("Ok")