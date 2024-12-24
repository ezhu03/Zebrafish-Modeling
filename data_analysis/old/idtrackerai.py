import numpy as np
import torch

int_array = np.zeros((1, 2), int)
print("Default NumPy integer type:", int_array.dtype)
print(f"{torch.__version__ = }, {np.__version__ = }")

loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)

target = torch.tensor(np.random.randint(0, 3, (3), int))
output = loss(input, target)