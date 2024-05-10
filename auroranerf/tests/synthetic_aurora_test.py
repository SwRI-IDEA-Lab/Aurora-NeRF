import unittest
import os
import numpy as np
import torch 

from auroranerf.synthetic_aurora import syntheticAurora

class SyntheticAuroraTest(unittest.TestCase):

    def setUp(self):
        self.synthetic_Aurora = syntheticAurora(centerPoint = torch.Tensor([0,0,100]),
                                                pointDirection = torch.Tensor([2, 0, 0]),
                                                intensityDecay = 1)

    def test_classExists(self):
        self.assertIsNotNone(self.synthetic_Aurora)

    def test_auroraParams(self):
        self.assertIsNotNone(self.synthetic_Aurora.centerPoint)
        self.assertIsNotNone(self.synthetic_Aurora.pointDirection)
        self.assertIsNotNone(self.synthetic_Aurora.intensityDecay)
        self.assertEqual(np.linalg.norm(self.synthetic_Aurora.pointDirection),1)
        self.assertGreater(self.synthetic_Aurora.centerPoint[2], 0)
        self.assertEqual(self.synthetic_Aurora.centerPoint.shape[0], 3)
        self.assertEqual(self.synthetic_Aurora.pointDirection.shape[0], 3)

    def test_volumetricIntensity(self):
        x = torch.arange(5)*20
        y = torch.arange(5)*18
        z = torch.ones(y.shape)*100
        
        self.assertIsNotNone(self.synthetic_Aurora.volumetricIntensityGenerator(x,y,z))
        intensity = self.synthetic_Aurora.volumetricIntensityGenerator(x, y, z)
        self.assertEqual(intensity.shape[0], y.shape[0])
        self.assertTrue((intensity>=0).all())

if __name__ == "__main__":
    unittest.main()
