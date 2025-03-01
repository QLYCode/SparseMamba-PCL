import torch
import torch.nn.functional as F
import cv2
import numpy as np

def get_image_pseudo_edge(images, scribbles, sobel_threshold=0.15, k=15, kernel_sizes=[3, 7, 15], ignore_index=4):
    edges = compute_image_edges(images, sobel_threshold)
    return generate_pseudo_edges(scribbles, edges, k, kernel_sizes, ignore_index=ignore_index)

def compute_image_edges(images, sobel_threshold=0.1):
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    edges = np.zeros((images.shape[0], 1, images.shape[1], images.shape[2]))
    for b in range(images.shape[0]):
        image_np = images[b]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        edge = (grad_mag > sobel_threshold).astype(np.float32)
        edges[b][0] = edge
    return torch.Tensor(edges)

def generate_pseudo_edges(scribbles, edge_maps, k, kernel_sizes, ignore_index=None):
    """
    Generate pseudo-edge maps based on scribbles and edge maps using dynamic kernel dilation.
    
    Args:
        scribbles (Tensor): Scribble masks of shape (B, C, H, W).
        edge_maps (Tensor): Edge maps of shape (B, 1, H, W).
        k (int): Maximum number of times a pixel can be added.
        kernel_sizes (list of int): List of odd kernel sizes in increasing order (e.g., [3,5,7]).
    
    Returns:
        Tensor: Pseudo-edge maps of shape (B, C, H, W).
    """
    B, C, H, W = scribbles.shape
    device = scribbles.device
    pseudo_edge_maps = torch.zeros_like(scribbles)
    count = torch.zeros_like(scribbles)  # Tracks additions per pixel per class
    
    for kernel_size in kernel_sizes:
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        
        for c in range(C):
            if c == ignore_index:
                continue
            # Extract scribble centers for class c and dilate
            scribble_c = scribbles[:, c].unsqueeze(1)  # Shape: (B, 1, H, W)
            dilated = F.conv2d(scribble_c.float(), kernel, padding=pad) > 0  # Boolean tensor
           
            
            # Determine eligible pixels for current class and kernel size
            current_count = count[:, c].unsqueeze(1)  # Shape: (B, 1, H, W)
            eligible = (edge_maps > 0) & (current_count < k) & dilated  # Boolean tensor
            
            # Update pseudo-edge maps with max value between current and eligible edges
            pseudo_edge_maps[:, c:c+1] = torch.maximum(pseudo_edge_maps[:, c:c+1], edge_maps * eligible)
            
            # Increment count for eligible pixels, clamping at k
            count[:, c:c+1] += eligible
            count[:, c:c+1] = torch.clamp(count[:, c:c+1], max=k)
    return pseudo_edge_maps.float()