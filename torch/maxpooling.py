import torch

input = [3, 4, 5, 6,
         8, 9, 0, 1,
         3, 5, 7, 9,
         3, 5, 7, 9]

# 参数分别为 B C H W
input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)

print(output)
