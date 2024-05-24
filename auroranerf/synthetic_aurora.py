import numpy as np
import torch

class syntheticAurora:
    """

    Class to initialize synthetic aurora. In this moment only tubular aurora is implemented. 

    Parameters
    ----------
    centerPoint : torch.Tensor
        The center point of the tubular aurora. Takes a three dimensional tensor [x, y, z]. 
        Expected units are Km. The ground is located at z = 0.
    pointDirection : torch.Tensor
        This is a vector that indicates the direction of the tube. This vector will be normalized
        into a unit vector. It has three components [x, y, z]. Units are 
        not important, but the relative magnitude of each component determines the direction of
        the tube. 
        The direction for positive x is East and negative x is West. 
        The direction for positive y is North and negative y is South.
        The direction for positive z is up.
    intensityDecay : float
        Value that describes intensity decay in 1/Km perpendicularly to the axis of the tube
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
                                     z: torch.Tensor)->torch.Tensor:
        """
        
        Method to evaluate the intensity per unit length for arbitrary set of points in space.

        Parameters
        ----------
        x : torch.Tensor
            1D Tensor with the x component of the points in space. Expected units are in Kms. 
            The direction for positive x is East and negative x is West.
        y : torch.Tensor
            1D Tensor with the y component of the points in space. Expected units are in Kms. 
            The direction for positive y is North and negative y is South.
        z : torch.Tensor
            1D Tensor with the z component of the points in space. Expected units are in Kms. 
            The direction for positive z is up. 0 is ground level. 

        Returns
        -------
        torch.Tensor
            Returns the point intensity of a voxel. 
            The maximum output value at the center of the tube is 1 and falls off perpendicular to 
            the tube's direction with intensity decay.
            Intensity values are unitless.
        """        
        
        p = torch.cat((x[:,None], y[:,None], z[:,None]), dim=-1) 
        a = p - self.centerPoint[None,:]
        distance = a - (((p - self.centerPoint[None,:])*self.pointDirection[None,:]).sum(dim=1))[:,None]*self.pointDirection[None,:]
        distance = torch.norm(distance, dim=1)

        return 1/(distance*self.intensityDecay)
