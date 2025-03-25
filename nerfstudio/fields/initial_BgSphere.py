import numpy as np
import math
import open3d as o3d
# from plyfile import PlyData, PlyElement
X_MIN, X_MAX = -14, 14
Y_MIN, Y_MAX = -9, 3.8
Z_MIN, Z_MAX = 0, 50

def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = (i / float(samples - 1)) -1   # y goes from 0 to -1
        radius = math.sqrt(1 - y ** 2)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))

    return points

def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class GaussianBGInitializer:
    def __init__(self,resolution,radius,center=np.array([0,0,0])):
        super().__init__()
        self.resolution = resolution
        self.radius = radius
        self.center = center

    def build_model(self):
        num_background_points = self.resolution ** 2
        xyz = fibonacci_sphere(num_background_points)
        xyz = np.array(xyz) * self.radius
        sky_pnt = xyz.astype(np.float32)
        sky_pnt += self.center

        return sky_pnt

