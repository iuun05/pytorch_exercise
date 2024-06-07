import torch

input = [3, 4, 5, 6, 7,
         8, 9, 0, 1, 2,
         3, 5, 7, 9, 1,
         3, 5, 7, 9, 0,
         1, 2, 1, 7, 9]

# 参数分别为 B C H W
input = torch.Tensor(input).view(1, 1, 5, 5)

# stride 步长
conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False, stride=2)

# 参数分为为output channel, input channel, weight, height
kernel =torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data

output = conv_layer(input)

print(output)