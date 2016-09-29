import numpy as np

"""
Transcribed from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
by hand for understanding.
"""
from sympy.simplify.radsimp import denom
from numpy.core.test_rational import denominator

def confusion_matrix(A, B, xmin=None, xmax=None):
  """
  Returns the confusion matrix for A and B
  """
  assert(len(A) == len(B))
  if xmin is None:
    xmin = min(A+B)
  if xmax is None:
    xmax = max(A+B)
  numc = int(xmax - xmin + 1)
  CM = [[0 for i in range(numc)]
        for j in range(numc)]
  for a,b in zip(A,B):
    CM[a-xmin][b-xmin] += 1
  return CM


def histogram(X, xmin=None, xmax=None):
  """
  Returns the histogram of x (should have reasonable number of unique values)
  """
  if xmin is None:
    xmin = min(X)
  if xmax is None:
    xmax = max(X)
  numc = int(xmax - xmin + 1)
  hist = [0 for x in range(numc)]
  for x in X:
    hist[x - xmin] += 1
  return hist

def quad_w_kappa(A, B, xmin=None, xmax=None):
  """
  Returns the Quadratic Weighted Kappa for two classifications
  """
  A = np.array(A)
  B = np.array(B)
  assert(len(A) == len(B))
  if xmin is None:
    xmin = min(min(A),min(B))
  if xmax is None:
    xmax = max(max(A),max(B))
  CM = confusion_matrix(A, B, xmin, xmax)
  numc = len(CM)
  numx = float(len(A))
  histA = histogram(A,xmin,xmax)
  histB = histogram(B,xmin,xmax)
  
  numerator   = 0.0
  demoninator = 0.0
  
  for i in range(numc):
    for j in range(numc):
      expected = (histA[i]*histB[j] / numx)
      d = pow(i-j, 2.0) / pow(numc - 1, 2.0)
      numerator   += CM[i][j] / numx
      denominator += expected / numx
  return 1.0 - numerator / denominator
     
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

    
