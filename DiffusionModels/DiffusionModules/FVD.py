import torch
import torch.nn as nn
import scipy
import numpy as np
from DiffusionModules.Util import open_url

# https://github.com/universome/fvd-comparison/blob/master/compare_models.py

class FVDLoss(nn.Module):
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)
    
    def __init__(self, device):
        super(FVDLoss, self).__init__()
        with open_url(FVDLoss.detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
        
    def forward(self, videos_fake, targets):
        feats_fake = self.detector(videos_fake, **FVDLoss.detector_kwargs).cpu().detach().numpy()
        feats_real = self.detector(targets, **FVDLoss.detector_kwargs).cpu().detach().numpy()
        
        return FVDLoss.compute_fvd(feats_fake, feats_real)
    
    @staticmethod
    def compute_fvd(feats_fake, feats_real):
        mu_gen, sigma_gen = FVDLoss.compute_stats(feats_fake)
        mu_real, sigma_real = FVDLoss.compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

        return float(fid)

    @staticmethod
    def compute_stats(feats):
        mu = feats.mean(axis=0) # [d]
        sigma = np.cov(feats, rowvar=False) # [d, d]

        return mu, sigma
    
    

    