from typing import Tuple, Optional

import torch
import torch.distributed as dist
from HAPiCLR.utils.misc import FilterInfNNan

### Gather Object from Pytorch Lightning 
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

### Pytorch Lightning nt_xent_loss 
def nt_xent_loss( out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("Distributed_multi_GPUs")
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)

        else:
            out_1_dist = out_1
            out_2_dist = out_2


        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
        
        out_dist=FilterInfNNan(out_dist)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
       
     
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
        
        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)


        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / (neg + eps)).mean()
        
        return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.
    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR
    
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        print("Gradient out backprob", grad_out.shape)
        return grad_out

def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0

def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1

def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherLayer.apply(input)


def eye_rank(n: int, device: Optional[torch.device]=None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the rank
    of this process set to 1.
    Example output where n=3, the current process has rank 1, and there are 
    4 processes in total:
        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0
    Equivalent to torch.eye for undistributed settings or if world size == 1.
    Args:
        n:
            Size of the square matrix on a single process.
        device:
            Device on which the matrix should be created.
    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask