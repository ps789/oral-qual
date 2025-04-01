import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.widgets import Slider
from concurrent.futures import ProcessPoolExecutor

class KF:
    def __init__(self, dim_x:int, dim_y:int, R, H):
        """
        Kalman Filter
        Use forecast(), forward(), and analyze()
        
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        
        P (numpy.ndarray): P Covariance mxm
        
        M (numpy.ndarray): M Model Matrix mxm 
        
        Q (numpy.ndarray): Q Covariance of the model error mxm
        
        R (numpy.ndarray): R Covariance of the observation error jxj
        
        H (numpy.ndarray): H Observation Matrix jxm
        """
        self.filterType = "KF"
        
        self.dim_x = dim_x  # dimension of x (m)
        self.dim_y = dim_y  # dimension of Observation Y (j)
        self.R = R          # R (Covariance of the observation error) jxj
        self.H = H          # H (Observation Matrix) jxm
        

class EnKF(KF):
    """Introduction to the principles and methods of data assimilation 
       in the geosciences.pdf(Bocquet)"""
    def __init__(self, dim_x:int, dim_y:int, R, H, H_j, n=10):
        """
        Stochastic Ensemble Kalman Filter
        Use enForecast(), enForward(), and enAnalyze()
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        n (int): Number of ensemble members
        """
        KF.__init__(self, dim_x, dim_y, R, H)
        self.filterType = "StochasticEnKF"
        self.n = n
        self.H_j = H_j 
        
    def enForward(self, forecast):
        x = np.mean(forecast, axis=0) # analysis ensemble mean
        self.X_cStack = np.column_stack((self.X_cStack, x))

class ETKF(EnKF):
    """
    Local Ensemble Transform Kalman Filter 
    Local Ensemble Transform Kalman Filter: An Efficient Scheme for Assimilating Atmospheric Data
    (Harlim and Hunt, 2006)
    """
    def __init__(self, dim_x:int, dim_y:int, R, H, n=10, rho=1):
        """
        Ensemble Transform Kalman Filter
        Use etForecast(), etForward(), and etAnalyze()d
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        n (int): Number of ensemble members
        
        rho (float): multiplicative inflation factor. must be greater or equal than 1
        """
        EnKF.__init__(self, dim_x, dim_y, R, H, H, n=n)
        self.filterType = "ETKF"
        self.R_inv = np.linalg.inv(self.R)
        if (rho < 1):
            raise ValueError("inflation factor must be greater than one")
        self.rho = rho
        self.ones = np.ones((1,self.n)) #1xn

class LETKF(ETKF):
    """Efficient data assimilation for spatiotemporal chaos: A local ensemble
       transform Kalman filter (Hunt et al)"""
    def __init__(self, dim_x:int, dim_y:int, R, H, L, n=10, rho=1):
        """
        Local Ensemble Transform Kalman Filter
        Use leForecast(), leForward(), and leAnalyze()
        leAnalyze_Parallel() for parallelized local operations
        
        @param:
        
        dim_x (int): dimension of X (m)
        
        dim_y (int): dimension of Observation Y (j)
        
        X0 (numpy.ndarray): Initial X (mean) mx1 
        
        P (numpy.ndarray): P (Covariance) mxm
        
        M (function): M (Forecast model function. Must return (dim_x,1) array)
        
        Q (numpy.ndarray): Q (Covariance of the model error) mxm
        
        R (numpy.ndarray): R (Covariance of the observation error) jxj
        
        H (function): H (Observation Operator. must return (m,j)) 
        
        H_j (function): H_j (linearized H. must return (m,j))
        
        L (function): L(m) (Localization operator. m=index of the grid point.
            must return a list of N indices of the observations)
        
        n (int): Number of ensemble members
        
        rho (float): multiplicative inflation factor. must be greater or equal than 1
        """
        ETKF.__init__(self, dim_x, dim_y, R, H, n=n, rho=rho)
        self.filterType = "LETKF"
        self.L = L
        self.indices = [m for m in range(self.dim_x)]
        
    
    def leForward(self, forecast):
        self.enForward(forecast)
    
    def leForward(self, forecast):
        self.enForward(forecast)
    
    def leAnalyze(self, enX, y):
        """
        Analyze LETKF
        
        y (numpy.ndarray): Observation of the true state with observation error
        """
        y = y[:, np.newaxis]
        X = enX.T # stacked ensemble members
        x = np.mean(enX, axis=0)[:, np.newaxis] # analysis ensemble mean
        
        Y = self.H(enX)
        Y_mean = np.mean(Y, axis=0)[:, np.newaxis]
        Y = Y.T
        Y = Y - Y_mean @ self.ones # subtract the mean on each columns of Y
        
        X = X - x @ self.ones # subtract the mean on each columns of X
        
        Xa = [0 for i in range(self.dim_x)]
        for m in range(self.dim_x):
            b = self.L(m) # indices of localized lows of Y
            Y_local = Y[b,:] #Nxn
            
            X_local = X[m,:] #Nxn
            N = len(b)
            
            R_local = np.diag([self.R_inv[z,z] for z in b]) #NxN 
                # Assume R is diagonal (each observation is independent from others)
            C = Y_local.T @ R_local #nxN
            
            P_tilde = self.n * np.eye((self.n)) / self.rho + C @ Y_local #nxn
            P_tilde = np.linalg.inv(P_tilde) #nxn
            
            W = (self.n-1) * P_tilde 
            W = sp.linalg.fractional_matrix_power(W, 0.5) #nxn
            # sometimes converted to complex numbers.
            # it comes from computational rounding in python
            # imaginary parts are all 0, so neglectable
            W = W.real
            
            w = P_tilde @ C @ (y[b,:].reshape((N,1)) - Y_mean[b,:].reshape(N,1)) # weight vector nx1
            
            W = W + w @ self.ones #nxn
            
            X_local = X_local @ W #Nxn
            
            X_local = X_local + x[m,:].reshape((1, 1)) @ self.ones #Nxn
            
            Xa[m] = X_local # save analyzed local grid point
            
        Xa = np.row_stack(Xa) #mxn
        
        enX = Xa.reshape((self.dim_x, enX.shape[0])).T #mx1
        
        x = np.mean(enX, axis=0) # analysis ensemble mean
        return enX