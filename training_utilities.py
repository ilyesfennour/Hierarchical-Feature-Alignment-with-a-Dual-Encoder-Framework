import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))

l2_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def perceptual_loss(model, prediction, target):
    """ VGG perceptual loss """
    with torch.no_grad():
        clean_vgg = model.get_vgg_features(target)
    out_vgg = model.get_vgg_features(prediction)
    
    return l2_loss(out_vgg, clean_vgg)

def tap_loss(model, enc_clean_feats, dec_feats):
    """ TAP loss """
    tap_loss = 0
    for i in range(min(3, model.num_layers-1)):
        enc_feat = model.enc_proj_layers[i](enc_clean_feats[i])
        dec_feat = model.dec_proj_layers[i](dec_feats[-(i+1)])
        if enc_feat.shape != dec_feat.shape:
            enc_feat = F.adaptive_avg_pool2d(enc_feat, dec_feat.shape[-2:])
        tap_loss += l2_loss(enc_feat, dec_feat)
    return tap_loss

def pairwise_distances(x):
    return torch.cdist(x, x, p=2)

def latent_distance_loss(z, x_ref):
    
    Dz = pairwise_distances(z)                          # latent distances
    Dx = pairwise_distances(x_ref.flatten(1))           # image-space distances

    Dz = Dz / (Dz.mean() + 1e-8)
    Dx = Dx / (Dx.mean() + 1e-8)

    return F.mse_loss(Dz, Dx)

def ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) *
                (sigma_x + sigma_y + C2))

    return 1 - ssim_map.mean()

def edge_loss(pred, target):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=pred.device, dtype=pred.dtype
    ).view(1, 1, 3, 3)

    sobel_y = sobel_x.transpose(2, 3)

    def gradient(img):
        gx = F.conv2d(img, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        gy = F.conv2d(img, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    return F.l1_loss(gradient(pred), gradient(target))

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim).to(device)
        )

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = (features - centers_batch).pow(2).sum() / (2.0 * batch_size)

        return loss

def batch_hard_triplet_loss(embeddings, labels, margin=1.0):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T)
    negative_mask = (labels != labels.T)

    diag_mask = torch.eye(len(labels), device=labels.device).bool()
    positive_mask[diag_mask] = False
    
    hardest_positive = (dist_matrix * positive_mask).max(dim=1)[0]
    max_dist = dist_matrix.max().item()
    masked_neg_dist = dist_matrix.clone()
    masked_neg_dist[~negative_mask] = max_dist + 1
    hardest_negative = masked_neg_dist.min(dim=1)[0]

    loss = F.relu(hardest_positive - hardest_negative + margin)

    return loss.mean()

def compute_total_loss(model, losses_config, prediction, target, enc_clean_feats, dec_feats, z=None, labels=None):
    """ Compute total loss """

    total_loss = 0

    if 'l1' in losses_config:
        total_loss += losses_config['l1']['lambda'] * l1_loss(prediction, target)

    if 'l2' in losses_config:
        total_loss += losses_config['l2']['lambda'] * l2_loss(prediction, target)

    if 'perceptual' in losses_config:
        total_loss += losses_config['perceptual']['lambda'] * perceptual_loss(model, prediction, target)

    if 'tap' in losses_config:
        total_loss += losses_config['tap']['lambda'] * tap_loss(model, enc_clean_feats, dec_feats)

    if 'latent' in losses_config:
        total_loss += losses_config['latent']['lambda'] * latent_distance_loss(z, target)

    if 'ssim' in losses_config:
        total_loss += losses_config['ssim']['lambda'] * ssim_loss(prediction, target)

    if 'edge' in losses_config:
        total_loss += losses_config['edge']['lambda'] * edge_loss(prediction, target)

    if 'triplet' in losses_config:
        total_loss += losses_config['triplet']['lambda'] * batch_hard_triplet_loss(z, labels)

    return total_loss