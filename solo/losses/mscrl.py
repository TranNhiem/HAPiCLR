# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank
import math



def simclr_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).
    Return:
        torch.Tensor: SimCLR loss.
    """
    # z = F.normalize(z, dim=-1)
    # gathered_z = z
    # indexes = indexes.unsqueeze(0)
    # gathered_indexes = indexes


    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
        gathered_z = gather(z)
        gathered_z = F.normalize(gathered_z, dim=-1)
        z = F.normalize(z, dim=-1)
        gathered_indexes = gather(indexes)
        indexes = indexes.unsqueeze(0)
        gathered_indexes = gathered_indexes.unsqueeze(0)
    else:
        z = F.normalize(z, dim=-1)
        gathered_z = z
        indexes = indexes.unsqueeze(0)
        gathered_indexes = indexes

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)#4096*4096

    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank():].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    # print("simclr pos",pos.shape)
    # print("simclr neg", neg.shape)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss, torch.mean(pos), torch.mean(neg)

def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = gather(out_1)
        out_2_dist = gather(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    out_1_dist = F.normalize(out_1_dist, dim=-1)
    out_2_dist = F.normalize(out_2_dist, dim=-1)
    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss , torch.mean(pos), torch.mean(neg)


def mscrl_loss_func_V1(
    z: torch.Tensor, f: torch.Tensor, b: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    f = F.normalize(f, dim=-1)
    b = F.normalize(b, dim=-1)

    #gathered_z = gather(z)#除了本身之外其它的output(同一個batch下)
    gathered_f = gather(f)
    gathered_b = gather(b)
    #sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    sim_f_f = torch.exp(torch.einsum("if, jf -> ij", f, gathered_f) / temperature)
    temp_sim_f_b = torch.einsum("if, jf -> ij", f, gathered_b)
    sim_f_b = torch.exp(temp_sim_f_b / temperature)#經過推算後發現會有crash


    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)#增加一個維度
    gathered_indexes = gathered_indexes.unsqueeze(0)#增加一個維度
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)


    # negatives
    #neg_mask = indexes.t() != gathered_indexes

    # print(pos_mask.shape)
    # print(pos_mask)
    # print(neg_mask)
    #sim_f_f 1024 *1024
    pos = torch.sum(sim_f_f * pos_mask, 1)
    neg = torch.sum(sim_f_b , 1)

    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    #loss =-(torch.mean(torch.log(pos_f / (pos_f + neg_f))))
    #loss = -torch.mean((torch.log(pos))-(torch.log(1-neg)))
    #loss = -(torch.mean(torch.log(pos) + torch.log(1-neg)))
    #目前只將neg改為前景和背景的相似度，其餘沒動
    return loss, torch.mean(pos), torch.mean(neg)

def mscrl_loss_func_V2(
    z: torch.Tensor, f: torch.Tensor, b: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    f = F.normalize(f, dim=-1)
    b = F.normalize(b, dim=-1)

    gathered_z = gather(z)#除了本身之外其它的output(同一個batch下)
    #gathered_f = gather(f)
    gathered_b = gather(b)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    sim_f_b = torch.exp(torch.einsum("if, jf -> ij", f, gathered_b) / temperature)
    #remove the 0 point
    # print(sim_f_b.type)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)#增加一個維度


    gathered_indexes = gathered_indexes.unsqueeze(0)#增加一個維度
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)


    # negatives
    #neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim_f_b, 1)
    loss = -(torch.mean(torch.log(pos / pos+neg)))
    #目前只將neg改為前景和背景的相似度，其餘沒動
    return loss, torch.mean(pos), torch.mean(neg)

def mscrl_loss_func_V3(
    z: torch.Tensor, f: torch.Tensor, b: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1, alpha: float = 0.5
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    #z = F.normalize(z, dim=-1)
    f = F.normalize(f, dim=-1)
    b = F.normalize(b, dim=-1)

    #gathered_z = gather(z)#除了本身之外其它的output(同一個batch下)
    gathered_f = gather(f)
    gathered_b = gather(b)

    sim_f_f = torch.exp(torch.einsum("if, jf -> ij", f, gathered_f) / temperature)
    sim_b_b = torch.exp(torch.einsum("if, jf -> ij", b, gathered_b) / temperature)
    sim_f_b = torch.exp(torch.einsum("if, jf -> ij", f, gathered_b) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)#增加一個維度
    gathered_indexes = gathered_indexes.unsqueeze(0)#增加一個維度
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)


    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = alpha*torch.sum(sim_f_f * pos_mask, 1) + (1-alpha)*torch.sum(sim_b_b * pos_mask, 1)
    neg = torch.sum(sim_f_b * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    #目前只將neg改為前景和背景的相似度，其餘沒動
    return loss, torch.mean(pos), torch.mean(neg)

def mscrl_loss_func_V4(
    z: torch.Tensor, f: torch.Tensor, b: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    f = F.normalize(f, dim=-1)
    # b = F.normalize(b, dim=-1)

    gathered_f = gather(f)
    sim_f_f = torch.exp(torch.einsum("if, jf -> ij", f, gathered_f) / temperature)



    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)#增加一個維度
    gathered_indexes = gathered_indexes.unsqueeze(0)#增加一個維度
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)


    # negatives
    neg_mask = indexes.t() != gathered_indexes

    # print(pos_mask.shape)
    # print(pos_mask)
    # print(neg_mask)
    #sim_f_f 1024 *1024
    pos = torch.sum(sim_f_f * pos_mask, 1)
    neg = torch.sum(sim_f_f * neg_mask, 1)


    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    #loss = -torch.mean((torch.log(pos))-(torch.log(1-neg)))
    #loss = -(torch.mean(torch.log(pos) + torch.log(1-neg)))
    #目前只將neg改為前景和背景的相似度，其餘沒動
    return loss, torch.mean(pos), torch.mean(neg)

def pixel_lavel_ontrastive(
    z: torch.Tensor, pixel_levels: list, pixel_mask: list, indexes: torch.Tensor, pixel_mask_b: list = None,
        temperature: float = 0.2
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    #z = F.normalize(z, dim=-1)
    pixel_levels = [F.normalize(pixel_level, dim=1) for pixel_level in pixel_levels]


    sim_pixel = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[1]) / temperature)

    # print("sim_pixel",sim_pixel.shape)

    # sim_pixel_v1 = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[0]) / temperature)
    # sim_pixel_v2 = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[1], pixel_levels[1]) / temperature)
    # pixel_neg_mask_V1 = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask_b[0])
    # pixel_neg_mask_V2 = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[1], pixel_mask_b[1])
    # pixel_neg_mask_V1 = pixel_neg_mask_V1 > 0
    # pixel_neg_mask_V2 = pixel_neg_mask_V2 > 0
    # pixel_neg_V1 = sim_pixel_v1.masked_select(pixel_neg_mask_V1).mean()
    # pixel_neg_V2 = sim_pixel_v2.masked_select(pixel_neg_mask_V2).mean()
    # #sim_pixel = torch.exp(F.cosine_similarity(pixel_levels[0][..., :, None], pixel_levels[1][..., None, :],dim=1) / temperature)
    #
    pixel_pos_mask = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask[1])
    # pixel_pos_mask = pixel_pos_mask > 0

    if pixel_mask_b != None:
        pixel_pos_mask += torch.einsum("...fi, ...fj -> ...ij", pixel_mask_b[0], pixel_mask_b[1])


    pixel_pos_mask = pixel_pos_mask > 0

    # print("pixel_pos_mask", pixel_pos_mask.shape)
    # pixel_pos = sim_pixel.masked_select(pixel_pos_mask)
    # pixel_neg = sim_pixel.masked_select(torch.logical_not(pixel_pos_mask))
    pixel_pos = sim_pixel * pixel_pos_mask
    pixel_neg = sim_pixel * torch.logical_not(pixel_pos_mask)
    # print("pixel_pos", pixel_pos[0])
    # print("pixel_neg", pixel_neg[0])
    # pixel_pos = pixel_pos
    # pixel_neg = pixel_neg.sum()
    # print("pixel_pos_sum", pixel_pos)
    # print("pixel_neg_sum", pixel_neg)
    # pixel_pos = pixel_pos.mean()
    # pixel_neg = pixel_neg.mean()
    pixel_pos = torch.sum(pixel_pos, (1,2))+1
    pixel_neg = torch.sum(pixel_neg, (1,2))+1
    # print("pixel_pos_sum", pixel_pos.shape)
    # print("pixel_neg_sum", pixel_neg.shape)


    pixel_levels_loss = -(torch.log(pixel_pos / (pixel_pos + pixel_neg)))
    # print(pixel_levels_loss.shape)
    loss = pixel_levels_loss.mean()
    # print(loss)
    return loss, pixel_pos.mean(), pixel_neg.mean()


def pixel_lavel_ontrastive_new(
        z: torch.Tensor, pixel_levels: list, pixel_mask: list, indexes: torch.Tensor, pixel_mask_b: list = None,
        temperature: float = 0.2
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    # z = F.normalize(z, dim=-1)
    pixel_levels = [F.normalize(pixel_level, dim=1) for pixel_level in pixel_levels]

    sim_pixel = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[1]) / temperature)

    # if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
    #     print("distributed")
    #     sim_pixel_gather = gather(sim_pixel)
    # else:
    #     sim_pixel_gather = sim_pixel
    #
    # sim_pixel_gather = sim_pixel_gather

    pixel_pos_mask = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask[1])
    #
    if pixel_mask_b != None:
        pixel_pos_mask += torch.einsum("...fi, ...fj -> ...ij", pixel_mask_b[0], pixel_mask_b[1])

    pixel_pos_mask = pixel_pos_mask > 0

    # pixel_ff = sim_pixel.masked_select(pixel_pos_mask).sum()
    # pixel_fb_bb = sim_pixel.masked_select(torch.logical_not(pixel_pos_mask)).sum()

    pixel_pos = sim_pixel * pixel_pos_mask #512*49*49
    pixel_neg = sim_pixel * torch.logical_not(pixel_pos_mask)

    pixel_pos = torch.sum(pixel_pos, (1, 2)) + 1e-5 #[batch * 1]
    total_sum = sim_pixel.sum() #[1]
    # print("pixel_pos", pixel_pos.shape)
    # print("total_sum", total_sum.shape)

    pixel_hapiclr = -(torch.log(pixel_pos / (total_sum)))

    #pixel_levels_loss = -(torch.log(pixel_pos / (pixel_pos + pixel_neg)))
    loss = pixel_hapiclr.mean()
    return loss, pixel_pos.mean(), pixel_neg.mean()

def pixel_lavel_ontrastive_new_V2(
        z: torch.Tensor, pixel_levels: list, pixel_mask: list, indexes: torch.Tensor, pixel_mask_b: list = None,
        temperature: float = 0.2
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    # z = F.normalize(z, dim=-1)
    pixel_levels = [F.normalize(pixel_level, dim=1) for pixel_level in pixel_levels]

    sim_pixel = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[1]) / temperature)

    # if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
    #     print("distributed")
    #     sim_pixel_gather = gather(sim_pixel)
    # else:
    #     sim_pixel_gather = sim_pixel
    #
    # sim_pixel_gather = sim_pixel_gather

    pixel_pos_mask = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask[1])
    #
    if pixel_mask_b != None:
        pixel_pos_mask += torch.einsum("...fi, ...fj -> ...ij", pixel_mask_b[0], pixel_mask_b[1])

    pixel_pos_mask = pixel_pos_mask > 0

    pixel_ff = sim_pixel.masked_select(pixel_pos_mask)
    pixel_neg = sim_pixel.masked_select(torch.logical_not(pixel_pos_mask))

    pixel_hapiclr = -torch.mean(torch.log(pixel_ff/torch.mean(sim_pixel)))

    # pixel_pos = sim_pixel * pixel_pos_mask #512*49*49
    # pixel_neg = sim_pixel * torch.logical_not(pixel_pos_mask)

    # pixel_pos = torch.sum(pixel_pos, (1, 2)) + 1e-5 #[batch * 1]
    # total_sum = torch.sum(sim_pixel) #[1*1.2M]
    # print("pixel_pos", pixel_pos.shape)
    # print("total_sum", total_sum.shape)


    #pixel_levels_loss = -(torch.log(pixel_pos / (pixel_pos + pixel_neg)))
    loss = pixel_hapiclr.mean()
    return loss, pixel_ff.mean(), pixel_neg.mean()


def pixel_lavel_ontrastive_DCL(
    z: torch.Tensor, pixel_levels: list, pixel_mask: list, indexes: torch.Tensor, pixel_mask_b: list = None,
        temperature: float = 0.2
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    1024 * 128

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    #z = F.normalize(z, dim=-1)
    pixel_levels = [F.normalize(pixel_level, dim=1) for pixel_level in pixel_levels]

    sim_pixel = torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[1]) / temperature
    # sim_pixel_v1 = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[0], pixel_levels[0]) / temperature)
    # sim_pixel_v2 = torch.exp(torch.einsum("...fi, ...fj -> ...ij", pixel_levels[1], pixel_levels[1]) / temperature)
    # pixel_neg_mask_V1 = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask_b[0])
    # pixel_neg_mask_V2 = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[1], pixel_mask_b[1])
    # pixel_neg_mask_V1 = pixel_neg_mask_V1 > 0
    # pixel_neg_mask_V2 = pixel_neg_mask_V2 > 0
    # pixel_neg_V1 = sim_pixel_v1.masked_select(pixel_neg_mask_V1).mean()
    # pixel_neg_V2 = sim_pixel_v2.masked_select(pixel_neg_mask_V2).mean()
    # #sim_pixel = torch.exp(F.cosine_similarity(pixel_levels[0][..., :, None], pixel_levels[1][..., None, :],dim=1) / temperature)
    #
    pixel_pos_mask = torch.einsum("...fi, ...fj -> ...ij", pixel_mask[0], pixel_mask[1])
    # pixel_pos_mask = pixel_pos_mask > 0

    if pixel_mask_b != None:
        pixel_pos_mask += torch.einsum("...fi, ...fj -> ...ij", pixel_mask_b[0], pixel_mask_b[1])


    pixel_pos_mask = pixel_pos_mask > 0

    pixel_pos = - sim_pixel.masked_select(pixel_pos_mask).mean()
    pixel_neg = torch.exp(sim_pixel.masked_select(torch.logical_not(pixel_pos_mask)).mean())


    pixel_levels_loss = pixel_pos+torch.log(pixel_neg)
    loss = pixel_levels_loss
    return loss, pixel_pos, pixel_neg

def DCL_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.
    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).
    Return:
        torch.Tensor: SimCLR loss.
    """
    # z = F.normalize(z, dim=-1)
    # gathered_z = z
    # indexes = indexes.unsqueeze(0)
    # gathered_indexes = indexes
    z = F.normalize(z, dim=-1)
    gathered_z = z
    indexes = indexes.unsqueeze(0)
    gathered_indexes = indexes

    sim = torch.einsum("if, jf -> ij", z, gathered_z) / temperature

    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank():].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = -torch.sum(sim * pos_mask, 1)
    neg = torch.sum(torch.exp(sim) * neg_mask, 1)
    loss = torch.mean(pos+torch.log(neg))
    return loss, torch.mean(pos), torch.mean(neg)
