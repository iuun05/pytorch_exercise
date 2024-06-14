import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layer = 1

cell = torch.nn.RNN(input_size, hidden_size,num_layers=num_layer)

dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layer, batch_size, hidden_size)

print(hidden)
print(dataset)

out, hidden = cell(dataset, hidden)


print('output size: ', out.shape)
print(out)
print('output size: ', hidden.shape)
print(hidden)


# for index, input in enumerate(dataset):
#     print('='*20, index, '='*20)

#
#     hidden = cell(input, hidden)
#
