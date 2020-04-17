from typing import Callable, Union

import numpy as np
import torch

LinearOperator = Callable[[torch.Tensor], torch.Tensor]

Matrix = Union[torch.Tensor, np.array]
