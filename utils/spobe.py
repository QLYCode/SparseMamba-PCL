import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectPseudoBoundaryGenerator(nn.Module):
    def __init__(self, k=25, kernel_sizes=[7, 13, 25], sobel_threshold=0.4, ignore_index=None, device="cuda"):
        super(ObjectPseudoBoundaryGenerator, self).__init__()
        self.k = k
        self.kernel_sizes = kernel_sizes
        self.sobel_threshold = sobel_threshold
        self.ignore_index = ignore_index
        self.device = device

    def compute_gt_edges(self, images):
        images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
        edges = np.zeros((images.shape[0], 1, images.shape[1], images.shape[2]))
        for b in range(images.shape[0]):
            image_np = images[b]
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_mag = grad_mag / (grad_mag.max() + 1e-8)
            edge = (grad_mag > self.sobel_threshold).astype(np.float32)
            edges[b][0] = edge
        return torch.Tensor(edges).to(self.device)

    def get_natural_pseudo_edge(self, images, scribbles):
        edges = self.compute_gt_edges(images)
        return self.generate_pseudo_edges(scribbles, edges)

    def generate_pseudo_edges(self, scribbles, edge_maps):
        _, C, _, _ = scribbles.shape
        pseudo_edge_maps = torch.zeros_like(scribbles).to(self.device)
        count = torch.zeros_like(scribbles).to(self.device)

        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, "Kernel size must be odd"
            pad = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
            for c in range(C):
                if c == self.ignore_index:
                    continue
                scribble_c = scribbles[:, c].unsqueeze(1)
                dilated = F.conv2d(scribble_c.float(), kernel, padding=pad) > 0
                current_count = count[:, c].unsqueeze(1)
                eligible = (edge_maps > 0) & (current_count < self.k) & dilated

                pseudo_edge_maps[:, c:c + 1] = torch.maximum(pseudo_edge_maps[:, c:c + 1], edge_maps * eligible)
                count[:, c:c + 1] += eligible
                count[:, c:c + 1] = torch.clamp(count[:, c:c + 1], max=self.k)
        return pseudo_edge_maps.float()

    def forward(self, images, scribbles):
        return self.get_natural_pseudo_edge(images, scribbles)

