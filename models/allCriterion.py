import torch.nn.functional as F
import torch.nn as nn

import torch
class ImprovedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, T=2.0):
        super(ImprovedDistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_latent, teacher_latent, student_depth, depth):
        latent_loss = F.kl_div(
            F.log_softmax(student_latent / self.T, dim=1),
            F.softmax(teacher_latent / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T * self.T)
        depth_loss = F.smooth_l1_loss(student_depth, depth)
        return self.alpha * latent_loss + (1 - self.alpha) * depth_loss
    
class LogDepthLoss(nn.Module):
    def __init__(self):
        super(LogDepthLoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.log(torch.abs(pred - target) + 1))