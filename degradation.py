import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Precompute GPU blur kernel
k = 15
kernel = torch.zeros(k, k, device=device)
kernel[k//2,:] = 1.0/k
blur_kernel = kernel.view(1,1,k,k).repeat(3,1,1,1)

# image enhancment
def add_gaussian_noise(img, sigma=75/255.0):
    """Add Gaussian noise to image"""
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0.0, 1.0)

def add_motion_blur(img):
    """Add motion blur to image"""
    x = img.unsqueeze(0).to(device)
    pad = F.pad(x, (7,7,7,7), mode='reflect')
    out = F.conv2d(pad, blur_kernel, groups=3)
    return out.squeeze(0).cpu()

def apply_super_resolution(img, scale_factor=4, noise_sigma=10/255.0):
    """downscale then upscale with poor quality"""

    # Get dimensions
    _, h, w = img.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # 1. Downsample to low-resolution (clean downsampling)
    low_res = F.interpolate(
        img.unsqueeze(0),
        size=(lr_h, lr_w),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)
    
    # 2. Add slight noise to simulate low-quality capture
    low_res = add_gaussian_noise(low_res, sigma=noise_sigma)
    
    # 3. Upscale with poor quality (simulating bilinear/bicubic upscaling)
    degraded = F.interpolate(
        low_res.unsqueeze(0),
        size=(h, w),
        mode='bilinear',  # Poor quality upsampling
        align_corners=False
    ).squeeze(0)
    
    # 4. Add compression artifacts noise
    degraded = add_gaussian_noise(degraded, sigma=noise_sigma/3)
    return degraded

# recovering missing data
def apply_missing_data(img, mask_ratio=0.3, mask_type='Central_block'):
    """Simulate missing/corrupted data by masking parts of image"""
    _, h, w = img.shape
    mask = torch.ones_like(img)
    
    if mask_type == 'random':
        # Random pixels
        num_missing = int(h * w * mask_ratio)
        indices = torch.randperm(h * w)[:num_missing]
        mask = mask.view(3, -1)
        mask[:, indices] = 0
        mask = mask.view(3, h, w)
        
    elif mask_type == 'block':
        # Block missing region
        block_h, block_w = int(h * 0.3), int(w * 0.3)
        start_h, start_w = torch.randint(0, h-block_h, (1,)), torch.randint(0, w-block_w, (1,))
        mask[:, start_h:start_h+block_h, start_w:start_w+block_w] = 0
    
    elif mask_type == 'Central_block':
        mask[:, h//4:3*h//4, w//4:3*w//4] = 0
    
    # Apply mask
    degraded = img * mask
    mask = 1 - mask
    
    return degraded.clamp(0, 1), mask

def apply_degradation(clean_img, task=None, sigma=75/255.0, scale_factor=4):
    """Apply degradation based on task"""
    mask = clean_img
    if task == 'denoising':
        # Target is clean, degraded is noisy
        target = clean_img
        degraded = add_gaussian_noise(clean_img, sigma)
                
    elif task == 'super_resolution':
        # Target is high-resolution, degraded is low-resolution
        target = clean_img
        degraded = apply_super_resolution(clean_img, scale_factor=scale_factor)

    elif task == 'missing_data':
        target = clean_img
        degraded, mask = apply_missing_data(clean_img)

    else:
        raise ValueError(f"Unknown task: '{task}'")
    
    return degraded, target, mask