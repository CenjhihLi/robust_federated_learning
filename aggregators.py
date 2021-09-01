"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
functions: mean, coordinatewise, quantile, median, trimmed_mean_1d, trimmed_mean(line78)
we write our trimmed_mean since the original one is too slow

Author: Cen-Jhih Li
Belongs: Academia Sinica, Institute of Statistical Science, Robust federated learning project
"""

import numpy as np
import wquantiles as w

from functools import partial
from scipy.linalg import svd
#from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def mean(points, weights):
    return np.average(points, axis=0, weights=weights)#.astype(points.dtype)


def std(points, weights):
    mu = mean(points, weights)
    return np.sqrt(mean(np.subtract(points, mu)**2, weights))


def cov(points, weights):
    """
    points.shape = (m, p=[...])
    cov.shape should be (p, p)
    """
    if weights is None:
        return np.cov(np.transpose([p.reshape([-1]) for p in points]))
    return np.cov(np.transpose([p.reshape([-1]) for p in points]), aweights = weights)


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
#def coordinatewise(fn, points, weights= None):
#    points = np.asarray(points)
#    if points.ndim == 1:
#        return fn(points, weights)
#    shape = points.shape
#    res = np.empty_like(points, shape=shape[1:])
#    for index in np.ndindex(*shape[1:]):
#        coordinates = points[(..., *index)]
#        res[index] = fn(coordinates, weights)
#    return res
"""
Too slow in parameter for loop
"""


def coordinatewise(fn, points, weights):
    if len(points) == 1:
        return fn(points, weights)
    shape = np.shape(points[0])
    points = np.transpose([np.reshape(p, [-1]) for p in points])
    return np.transpose(list(map(lambda x: fn(x, weights), points))).reshape(shape)
    #return np.transpose([fn(v, weights) for v in points]).reshape(shape)


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def quantile(points, weights = None, quantile = 0.5):
    if weights is None:
        return np.quantile(points, quantile, axis=0).astype(np.float32)
    return coordinatewise(partial(w.quantile_1D, quantile=quantile), points, weights)


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def median(points, weights = None):
    return quantile(points, weights, 0.5)
    # return np.median(points, axis=0) if weights is None \
    #     else np.apply_along_axis(weightedstats.numpy_weighted_median, 0,
    #                              points,
    #                              weights)

"""
# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def trimmed_mean_1d(vector, weights, beta):
    if weights is None:
        lower_bound, upper_bound = np.quantile(vector, (beta, 1 - beta)).astype(np.float32)
        trimmed = [v for v in vector if lower_bound < v < upper_bound]
        if trimmed:
            return mean(trimmed, None)
        else:
            return (lower_bound + upper_bound) / 2
    else:
        lower_bound, upper_bound = w.quantile_1D(vector, weights, beta), w.quantile_1D(vector, weights, 1 - beta)

        trimmed = [(v, w) for v, w in zip(vector, weights) if lower_bound < v < upper_bound]
        if trimmed:
            trimmed_vector, trimmed_weights = zip(*trimmed)

            return mean(trimmed_vector, trimmed_weights)
        else:
            return (lower_bound + upper_bound) / 2


#reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def trimmed_mean(points, weights, beta):
    return coordinatewise(partial(trimmed_mean_1d, beta=beta), points, weights)
"""

"""
Too slow in parameter for-loop in coordinatewise
use below
"""

def trimmed_mean(points, weights=None, beta = 0.1):
  shape = np.shape(points[0])
  points = np.asarray([np.reshape(p, [-1]) for p in points])
  low = np.quantile(points, beta, axis=0) if weights is None else coordinatewise(partial(w.quantile_1D, quantile=beta), points, weights)
  high = np.quantile(points, 1-beta, axis=0)  if weights is None else coordinatewise(partial(w.quantile_1D, quantile=1-beta), points, weights)

  mask=np.multiply(points > low, points < high)  
  mu = (low+high)/2.0
  if mask.sum()==0:
      return mu.reshape(shape)
  index = mask.sum(axis = 0)>0  #shape (m, (p[index]))
  if weights is None:
    matrix_trim = np.multiply(points[:,index], mask[:,index])       
    #shape (m, (p[index]))
    mu[index] = matrix_trim.sum(axis=0) / mask[:,index].sum(axis=0) 
    #update shape (p[index])
  else:
    weights = mask[:,index] * (np.asarray(weights).reshape([-1,1])) 
    #shape (m, (p[index]))
    matrix_trim = np.multiply(points[:,index], mask[:,index])       
    #shape (m, (p[index]))
    mu[index] = np.multiply(matrix_trim, weights).sum(axis=0) / weights.sum(axis=0) 
    #shape(m, (p[index])) * (m, (p[index])).sum (p[index]) / (m , (p[index])) . sum (p[index])
  return mu.reshape(shape)



"""
winsorized mean: not truncate but give lower_bound, upper_bound
""" 


def ext_remove(points, weights=None, beta=0.1):
    """
    keep the data from quantile beta to quantile 1-beta
    compare np.linalg.norm(p) in points
    """
    if beta<=0:
        print("beta<=0 means to keep all points")
        return points, weights
    if beta>=0.5:
        raise ValueError("beta>=0.5 means to drop out all points")
    
    if weights is None:
        upper = quantile(points, None, 1 - beta)
        lower = quantile(points, None, beta)
    else:
        upper = quantile(points, weights, 1 - beta)
        lower = quantile(points, weights, beta)
        
    if weights is None:
        points = [p for p in points 
                  if np.linalg.norm(lower) < np.linalg.norm(p) < np.linalg.norm(upper)]
        return points, None
    new_points=[]
    new_weights=[]
    for p, ws in zip(points, weights):
        if (np.linalg.norm(lower) < np.linalg.norm(p) < np.linalg.norm(upper)):
            new_points.append(p)
            new_weights.append(ws)
    return new_points, new_weights


def gamma_mean_1D(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, remove=False, beta=0.1):
    """
    We use element-wise mu & sigma,
    gamma_mean
    """
    if history_points is None:
        mu_hat = mean(points, weights)
        sigma_hat = std(points, weights)
    else:
        mu_hat = mean(points, weights)
        sigma_hat = std(history_points, weights)
    #sigma_hat = np.diag(np.cov(np.transpose(points)))
    
    """
    tol should consider the scale of points
    
    @Todo: How to set a good tol?
    Even though we use  np.multiply(np.abs(mu_hat), tol) 
        and the computation do not consider the element 
        that sigma_hat <= tol, 
    ZeroDivisionError still happen rarely when updating mu_hat and sigma_hat
    However, setting a fix number is not a good idea
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol )

    if remove:
        """
        remove the extreme points
        """
        points, weights = ext_remove(points, weights, beta=beta)
    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat
    
    for _ in range(max_iter):
        if history_points is None:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.divide(np.subtract(d[index], mu_hat[index]), sigma_hat[index], [-1]))) ))
                for d in points]
        else:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.divide(np.subtract(d[index], mu_hat[index]), sigma_hat[index], [-1]))) ))
                for d in history_points]
        
        if np.all(np.array(d_gamma)==0):
            return mu_hat
        
        if weights is None:
            mu_hat[index] = mean(points, d_gamma )[index]
            sigma_hat[index] = np.sqrt((1+gamma)*mean(np.square(np.subtract(points,mu_hat)), 
                                               d_gamma ))[index]
        else:
            mu_hat[index] = mean(points, weights=np.multiply(d_gamma,weights) )[index]
            sigma_hat[index] = np.sqrt((1+gamma)*mean(np.square(np.subtract(points,mu_hat)), 
                                               np.multiply(d_gamma,weights) ))[index]
        index = (sigma_hat > tol)
        if np.sum(index,axis=None)==0:
            return mu_hat
        #if remove:
        #    #remove the extreme points
        #    points, weights = ext_remove(points, weights, beta=beta)
    return mu_hat#.astype(points.dtype)


def simple_gamma_mean(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, remove=False, beta=0.1):
    """
    Do not consider cov inverse
    """
    if history_points is None:
        mu_hat = mean(points, weights)
        sigma_hat = std(points, weights)
    else:
        mu_hat = mean(points, weights)
        sigma_hat = std(history_points, weights)
    #sigma_hat = np.diag(np.cov(np.transpose(points)))
    
    """
    tol should consider the scale of points
    
    @Todo: How to set a good tol?
    Even though we use  np.multiply(np.abs(mu_hat), tol) 
        and the computation do not consider the element 
        that sigma_hat <= tol, 
    ZeroDivisionError still happen rarely when updating mu_hat and sigma_hat
    However, setting a fix number is not a good idea
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol )

    if remove:
        """
        remove the extreme points
        """
        points, weights = ext_remove(points, weights, beta=beta)
    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat
    
    for _ in range(max_iter):
        if history_points is None:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.subtract(d[index], mu_hat[index]),[-1])) ))
                for d in points]
        else:
            d_gamma=[np.exp(-(gamma/2) * np.square(np.linalg.norm(
                np.reshape(np.subtract(d[index], mu_hat[index]),[-1])) ))
                for d in history_points]
        if np.all(np.array(d_gamma)==0):
            return mu_hat
        if weights is None:
            mu_hat[index] = mean(points, d_gamma )[index]
        else:
            mu_hat[index] = mean(points, weights=np.multiply(d_gamma,weights) )[index]
    return mu_hat#.astype(points.dtype)


def dim_reduce(points, weights, method, dim = None):
    """
    points: (m, p)
    cov--> (p, p)
    A v = lambda v => AV= Sigma V.transpose (V columns v1,v2,...)
    A = USV.transpose
    U: shape (p, p)
    S: eigenvalues 
    VT: shape (m, m)
    select max dim eigenvalues
    get transform_map by VT[:dim,:]    
    (p, dim)
    and inverse_transeform_map by transpose VT[:dim,:]
    (dim, p)
    
    return transform_map, inverse_transeform_map
    lowdim_data = np.dot(points, transform_map)
    approx_estimate = np.dot(lowdim_data, inverse_transeform_map)
    """
    points = [np.asarray(p).reshape([-1]) for p in points]
    if method=='pca':
        _, sigma, v = svd(cov(points, weights))
        expla_var = np.divide(np.cumsum(sigma),np.maximum(np.sum(sigma),1e-5))
        thred = 0.95
        if dim is None:
            dim = np.sum(expla_var<=thred)+1
        else:
            dim = np.minimum(np.sum(expla_var<=thred)+1,dim)
        return np.transpose(v[:dim,:]), np.array(v[:dim,:])
    elif method=='truncated_svd':
        return
    elif method=='kernal_pca':  
        return
    elif method=='sparse_pca':
        return
    elif method=='incremental_pca':
        return


def gamma_mean_2D(points, weights=None, history_points=None, gamma = 0.1, max_iter=10, tol = 1e-7, dim_red=False, red_method='pca'):
    """
    We use element-wise mu & sigma,
    gamma_mean
    """
    original_tol=tol
    shape = np.shape(points[0])
    points = [np.asarray(p).reshape([-1]) for p in points]
    history_points = [np.asarray(p).reshape([-1]) for p in history_points]
    if history_points is None:
        mu_hat = mean(points, weights)
        sigma_hat = cov(points, weights)
    else:
        mu_hat = mean(points, weights)
        sigma_hat = cov(history_points, weights)
    
    """
    cov allow weights have some 0s
    """
    tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), tol), tol )

    #the similar entries do not need to update
    index = (sigma_hat > tol)       
    if np.sum(index,axis=None)==0:
        return mu_hat.reshape(shape)

    if dim_red:
        """
        dimension reduction
        compute in low dimension and go back to original dim after computation
        """
        transform_map, inverse_transeform_map = dim_reduce(points, weights, red_method)
        points = np.dot(points, transform_map)
        if weights is None:
            mu_hat = mean(points, None)
            sigma_hat = cov(points, None)
        else:
            mu_hat = mean(points, weights)
            sigma_hat = cov(points, weights)
        tol = np.where(np.abs(mu_hat)>0, np.multiply(np.abs(mu_hat), original_tol), original_tol )
        #the similar entries do not need to update
        index = (sigma_hat > tol)  
    """
    @Todo, what if np.linalg.inv(sigma_hat) not exist?
    Face computation issue in inverse computing
    some columns=0, inverse does not exist 
    """
    for _ in range(max_iter):
        if history_points is None:
            d_gamma=[np.exp(-(gamma/2) * 
                np.dot(np.dot(np.subtract(d, mu_hat),
                    np.linalg.inv(sigma_hat)),
                    np.transpose(np.subtract(d, mu_hat))))
                for d in points]
        else:
            d_gamma=[np.exp(-(gamma/2) * 
                np.dot(np.dot(np.subtract(d, mu_hat),
                    np.linalg.inv(sigma_hat)),
                    np.transpose(np.subtract(d, mu_hat))))
                for d in history_points]
        if np.all(np.array(d_gamma)==0):
            if dim_red:
                """
                go back to original dim after computation
                """
                mu_hat = np.dot(mu_hat, inverse_transeform_map)
            return mu_hat.reshape(shape)
        
        if dim_red or weights is None:
            mu_hat = mean(points, d_gamma )
            sigma_hat = cov(points, d_gamma)
        else:
            mu_hat = mean(points, weights=np.multiply(d_gamma,weights) )
            sigma_hat = cov(points, np.multiply(d_gamma,weights) )
        index = (sigma_hat > tol)
        if np.sum(index,axis=None)==0:
            if dim_red:
                """
                go back to original dim after computation
                """
                mu_hat = np.dot(mu_hat, inverse_transeform_map)
            return mu_hat.reshape(shape)
    if dim_red:
        """
        go back to original dim after computation
        """
        mu_hat = np.dot(mu_hat, inverse_transeform_map)
    return mu_hat.reshape(shape)#.astype(points.dtype)

def gamma_mean(points, weights=None, history_points=None, compute = "1D", gamma = 0.1, max_iter=10, 
               tol = 1e-7, remove=False, beta=0.1, dim_red=False, red_method='pca'):
    if compute=="1D":
        return gamma_mean_1D(points=points, weights=weights, history_points=None, 
                             gamma = gamma, max_iter = max_iter, 
                             tol = tol, remove=remove, beta=beta)
    elif compute=="simple":
        return simple_gamma_mean(points=points, weights=weights, history_points=None, 
                                 gamma = gamma, max_iter = max_iter, 
                                 tol = tol, remove=remove, beta=beta)
    elif compute=="2D":
        return gamma_mean_2D(points=points, weights=weights, history_points=None, 
                             gamma = gamma, max_iter = max_iter, 
                             tol = tol, dim_red=dim_red, red_method=red_method)


def geometric_median(points, weights=None, max_iter = 1000, tol = 1e-7):
    """
    Use Weiszfeld's method
    """
    mu = mean(points, weights)
    def distance_func(x):
        #return np.linalg.norm(np.subtract(points,x), axis=1)
        return np.array([np.linalg.norm(np.subtract(w.reshape([-1]),x.reshape([-1])), axis=0) for w in points])   
    distances = distance_func(mu)
    for _ in range(max_iter):
        prev_mu = mu
        if weights is None:
          beta_weights = 1 / np.maximum(1e-5, distances)
        else:
          beta_weights = weights / np.maximum(1e-5, distances)
        mu = mean(points,beta_weights) #weight average
        
        distances = distance_func(mu)
        mu_movement = np.sqrt((np.subtract(prev_mu, mu)**2).sum())
        
        if mu_movement <= tol:
            break
    return mu