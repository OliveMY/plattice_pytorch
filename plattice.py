import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import PLOO
from torch.nn import Module

class PermutoLatticeFunction(Function):
    @staticmethod
    def forward(ctx, feature, values):
        weight, out = PLOO.forward(feature, values)
        ctx.save_for_backward(weight, feature)
        return out
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        weight, feature = ctx.saved_tensors
        grad_values = PLOO.backward(feature, grad_output.contiguous(), weight)
        return None, grad_values # no need to back propogate features

class PermutoLattice(Module):
    def forward(self, feature, values):
        return PermutoLatticeFunction.apply(feature, values)

if __name__ == "__main__":
    # check the gradient of the function
    ft = torch.randn(5,5,3,dtype=torch.float32,requires_grad=False).cuda() 
    v = torch.randn(5,5, 3,dtype=torch.float32,requires_grad=True).cuda()
    v.retain_grad()
    v_dv = torch.randn(5,5,3, dtype=torch.float32,requires_grad=True).cuda()
    v_dv.retain_grad()
    # weight, out = PLOO.forward(ft, v)
    # __, g_out = PLOO.forward(ft, v_dv)
    # grad_v = PLOO.backward(ft, torch.ones_like(out), weight)
    # _, out2 = PLOO.forward(ft, v+v_dv)
    # grad_cha = grad_v * g_out
    # real_cha = out2 - out
    # cha = grad_cha - real_cha
    # print(torch.mean(cha))
    
    plattice = PermutoLatticeFunction.apply
    c = v+v_dv
    c.retain_grad()
    a = plattice(ft, c)
    loss = 1 -  torch.sum(a)
    loss.backward()
    print(c.grad)

    

