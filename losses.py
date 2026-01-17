# losses.py
import torch
import torch.nn.functional as F


# -----------------------
# Reconstruction loss
# -----------------------
def reconstruction_loss(recon, x):
    """
    VAE reconstruction loss
    """
    return F.mse_loss(recon, x)


# -----------------------
# KL divergence
# -----------------------
def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma) and N(0, 1)
    """
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )


# -----------------------
# Dice loss
# -----------------------
def dice_loss(pred, target, eps=1e-6):
    """
    Soft Dice loss for binary segmentation
    """
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

    num = 2.0 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1.0 - num / den


# -----------------------
# Focal loss
# -----------------------
def class_balanced_focal_loss(
    pred,
    target,
    beta=0.999,
    gamma=2.0,
    eps=1e-6
):
    """
    Class-balanced focal loss (Cui et al.)
    """
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)


    # class frequencies
    pos = target.sum()
    neg = (1 - target).sum()

    effective_pos = (1 - beta ** pos) / (1 - beta + eps)
    effective_neg = (1 - beta ** neg) / (1 - beta + eps)

    w_pos = 1.0 / effective_pos
    w_neg = 1.0 / effective_neg

    weights = w_pos * target + w_neg * (1 - target)

    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)

    loss = weights * (1 - pt).pow(gamma) * bce
    return loss.mean()


def soft_skeletonize(x, iters=10):
    for _ in range(iters):
        min_pool = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(min_pool - x)
        x = F.relu(x - contour)
    return x


def cldice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)

    skel_pred = soft_skeletonize(pred)
    skel_gt = soft_skeletonize(target)

    tprec = (skel_pred * target).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt * pred).sum() / (skel_gt.sum() + eps)

    cldice = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cldice

#--------------Segmentation Loss-----------------
def segmentation_loss(
    pred,
    target,
    use_dice=True,
    use_focal=True,
    use_cldice=False,
    dice_w=1.0,
    focal_w=1.0,
    cldice_w=1.0,
):
    loss = 0.0

    if use_dice:
        loss += dice_w * dice_loss(pred, target)

    if use_focal:
        loss += focal_w * class_balanced_focal_loss(pred, target)

    if use_cldice:
        loss += cldice_w * cldice_loss(pred, target)

    return loss
