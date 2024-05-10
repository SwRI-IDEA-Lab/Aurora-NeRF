import numpy as np
import torch

class syntheticAurora:
    """

    """
    def __init__(self,
                 centerPoint:torch.Tensor,
                 pointDirection:torch.Tensor,
                 intensityDecay:float):
        self.centerPoint = centerPoint
        self.pointDirection = pointDirection/torch.linalg.norm(pointDirection)
        self.intensityDecay = intensityDecay

    def volumetricIntensityGenerator(self, 
                                     x: torch.Tensor, 
                                     y: torch.Tensor, 
                                     z: torch.Tensor):
        
        p = torch.cat((x[:,None], y[:,None], z[:,None]), dim=-1) 
        a = p - self.centerPoint[None,:]
        distance = a - (((p - self.centerPoint[None,:])*self.pointDirection[None,:]).sum(dim=1))[:,None]*self.pointDirection[None,:]
        distance = torch.norm(distance, dim=1)

        return 1/(distance*self.intensityDecay)
