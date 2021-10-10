# +
import torch
import torch.nn as nn

class MAE_FILTERED(nn.Module):
    """
    Directly optimizes the competition metric
    """
    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae



# -

if __name__ == '__main__':
    MAE_FILTERED()
