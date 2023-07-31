#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import z_values

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    zval: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features,2),**factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight[:,0].data.uniform_(-1,1)
        self.weight[:,1].data.uniform_(0,1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, data_inputs: Tensor) -> Tensor:
        torch.absolute(self.weight[:,1])
        r1 = data_inputs.T
        mu1 = self.weight[:,0]
        mu1 = mu1.unsqueeze(0)
        sigma1 = self.weight[:,1]
        sigma1 = sigma1.unsqueeze(0)
        l1 = self.in_features
        l2 = self.out_features
        if torch.cuda.is_available():
            zval=[]
            zval=torch.tensor(torch.zeros(l2), device=torch.device("cuda"))
            
        else:
            zval=torch.tensor(torch.zeros(l2))
            
        for i in range (1,l2+1):
            b = round(i/l2,2)
            b1 = math.trunc(b*10)/10
            r = int(b1 * 10)
            c = int((b - b1) * 100)
            zval[i-1] = z_snd[r][c]
        zval1 = zval[np.newaxis, :]
        r1 = torch.matmul((torch.matmul(sigma1.T,zval1) + mu1.T).T,r1).T
        return r1
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

