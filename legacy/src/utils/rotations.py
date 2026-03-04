"""  File to handle everything related to rotations

Includes a class to represent a rotation

This file contains code to generate a random rotation and use
it in every representation that could be useful. The generation works in a way that
every possible rotation is eaually probable.

This file conatins the following classes and functions:
    * Rotation: Class to represent a rotation
    *

Notes:
    * At the moment omega is completely useless, because the speed of the rotation is calculated 
      using the fps and a rounds per second value.
"""

import numpy as np

class Rotation:
    """Class to represent a rotation
    
    This class is used to represent a rotation in 3D space. It can be used to
    generate a random rotation or convert different representations of a rotation to each other.
    
    Attributes:
        phi (float): Phi part of the exponential coordinates representation [0 -> 2π].
        theta (float): Theta part of the exponential coordinates representation cos(theta) [-1 -> 1].
        omega (float): angle of the rotation [0 -> 2π].

    Methods:

    """

    def __init__(self):
        """Constructor for the Rotation class
        
        This constructor initializes the rotation matrix to the identity matrix.
        """
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


    def get_axis(self) -> np.ndarray:
        """Get the axis of rotation
        
        This method returns the axis of rotation as a numpy array.
        
        Returns:
            np.ndarray: [x, y, z] Axis of rotation
        """
        return np.array([self.x,
                         self.y,
                         self.z])
    
    def set_axis(self, x:float, y:float, z:float) -> None:
        """ Sets roation axis and speed
        
            The internal values are set in a way, that the rotation represents a
            rotation around the given axis with a speed of the maginitude of the vector
        """
        self.x = x
        self.y = y
        self.z = z


    def set_axis_np(self, axis:np.ndarray) -> None:
        """ Set roation axis and speed

            Overload to include easy usage of np array
        """
        self.set_axis(axis[0], axis[1], axis[2])

    
    def get_angle(self) -> float:
        """Get the angle of rotation
        
        This method returns the angle of rotation in radians.
        
        Returns:
            float: Angle of rotation in radians
        """
        return np.linalg.norm([self.x, self.y, self.z])
    

    def set_spherical(self, phi: float, theta: float, omega: float) -> None:
        """Set the rotation using spherical coordinates
        
        This method sets the rotation using spherical coordinates.
        
        Args:
            phi (float): Phi part of the exponential coordinates representation [0 -> 2π].
            theta (float): Theta part of the exponential coordinates representation cos(theta) [-1 -> 1].
            omega (float): angle of the rotation [0 -> 2π].
        """
        x = np.cos(phi) * np.cos(theta)
        y = np.sin(phi) * np.cos(theta)
        z = np.sin(theta)
        vec = np.array([x, y, z])
        vec /= np.linalg.norm(vec)
        vec *= omega
        self.set_axis_np(vec)


# TODO: Add other functions as needed

def random_rotation() -> Rotation:
    """Generate a random rotation
    
    This function generates a random rotation in 3D space. It uses the exponential coordinates
    representation of a rotation to generate a random rotation.
    
    Returns:
        Rotation: A random rotation object
    """
    rot = Rotation()
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    theta = np.arccos(cos_theta)
    omega = np.random.uniform(0, 2 * np.pi)
    rot.set_spherical(phi, theta, omega)
    return rot
