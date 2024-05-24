    # Everything related to rendering. 
import torch
class RenderingAurora:

    def __init__(self,
                 bottomSouthwestCorner:torch.Tensor,
                 topNortheastCorner:torch.Tensor):
        
        self.bottomSouthwestCorner = bottomSouthwestCorner
        self.topNortheastCorner = topNortheastCorner
        
        #return 