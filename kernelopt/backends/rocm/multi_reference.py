import torch
class Model(torch.nn.Module):
 def forward(self,x): return (x+1)+1
def get_inputs(): return [torch.randn(256)]
def get_init_inputs(): return []
