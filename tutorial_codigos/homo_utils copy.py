import numpy as np
from collections import OrderedDict
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt


def plot_shape(shp, w=1.5, fill='b'):

  plt.scatter(shp[:, 0], shp[:, 1])

  colors = ['r', 'g', 'b', 'm']
  for p1, p2, c in zip(shp, shp[1:], colors):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=w, color=c, alpha=0.5)
  p1 = shp[-1]
  p2 = shp[0]
  plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=w, color=colors[-1], alpha=0.5)

  p = Polygon(shp, facecolor=fill, alpha=0.1)
  plt.gca().add_patch(p)


def homo(points_2d):
  n = points_2d.shape[0]
  z = np.ones((n, 1))
  return np.hstack((points_2d, z))

def cart(h_points):
  x = h_points[:, 0]
  y = h_points[:, 1]
  z = h_points[:, 2]
  return np.stack((x/z, y/z)).T


def apply_transform(shape, T):
  h_ori = homo(shape)
  h_dst = np.dot(T, h_ori.T)
  dst = cart(h_dst.T)
  return dst


def affine_inv(A):

  # extrae la parte lineal
  M = A[:2, :2]
  # y aparte la traslación
  t = A[:2, 2]

  # Computa la inversa de la parte lineal
  M_inv = np.linalg.inv(M)

  # Computa la traslación inversa
  t_inv = -np.dot(M_inv, t)

  # construye la matriz con la afinidad inversa
  A_inv = np.eye(3)
  A_inv[:2, :2] = M_inv
  A_inv[:2, 2] = t_inv
  return A_inv
  
