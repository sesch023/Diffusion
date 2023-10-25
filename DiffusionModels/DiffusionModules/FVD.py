import torch
import torch.nn as nn
import scipy
import numpy as np
from DiffusionModules.Util import open_url


class FVDLoss(nn.Module):
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)
    
    def __init__(self, device):
        """
        Intializes the FVD loss as defined in https://github.com/universome/fvd-comparison/blob/master/compare_models.py.
        This uses a pre-trained I3D model.

        :param device: Device to use.
        """        
        super(FVDLoss, self).__init__()
        with open_url(FVDLoss.detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
        
    def forward(self, videos_fake, targets):
        """
        Computes the FVD loss for the fake videos and the targets.

        :param videos_fake: Fake videos. 
        :param targets: Targets.
        :return: FVD loss.
        """        
        feats_fake = self.detector(videos_fake, **FVDLoss.detector_kwargs).cpu().detach().numpy()
        feats_real = self.detector(targets, **FVDLoss.detector_kwargs).cpu().detach().numpy()
        
        return FVDLoss.compute_fvd(feats_fake, feats_real)
    
    @staticmethod
    def compute_fvd(feats_fake, feats_real):
        """
        Computes the FVD loss for the fake and real features.

        :param feats_fake: Fake features.
        :param feats_real: Real features.
        :return: FVD loss.
        """        
        mu_gen, sigma_gen = FVDLoss.compute_stats(feats_fake)
        mu_real, sigma_real = FVDLoss.compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

        return float(fid)

    @staticmethod
    def compute_stats(feats):
        """
        Computes the mean and covariance for the features.

        :param feats: Features.
        :return: Mean and covariance.
        """        
        mu = feats.mean(axis=0) # [d]
        sigma = np.cov(feats, rowvar=False) 

        return mu, sigma
    
    

    