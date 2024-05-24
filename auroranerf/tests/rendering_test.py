import unittest
import os
import numpy as np
import torch 

from auroranerf.synthetic_aurora import syntheticAurora
from auroranerf.rendering import RenderingAurora

class RenderingAuroraTest(unittest.TestCase):

    def setUp(self):
        self.synthetic_Aurora = syntheticAurora(centerPoint = torch.Tensor([0,0,100]),
                                                pointDirection = torch.Tensor([2, 0, 0]),
                                                intensityDecay = 1)
        
        self.rendering_Aurora = RenderingAurora(bottomSouthwestCorner = torch.Tensor([-60,-60, 0]),
                                                topNortheastCorner = torch.Tensor([60,60,300]))
    

    def test_classExists(self):
        self.assertIsNotNone(self.synthetic_Aurora)
        self.assertIsNotNone(self.rendering_Aurora)
 

    def test_volume(self):
        self.assertIsNotNone(self.rendering_Aurora.bottomSouthwestCorner)
        self.assertIsNotNone(self.rendering_Aurora.topNortheastCorner)

    def test_camera(self):
        self.assertIsNotNone(self.rendering_Aurora.syntheticCamera)

    
if __name__ == "__main__":
    unittest.main()
