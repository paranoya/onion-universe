
import numpy as np
#import scipy
from scipy.special import zeta
from time import time
from astropy import cosmology as cosmo
from astropy import units as u
from astropy import constants as c

# -----------------------------------------------------------------------------


class CosmologicalModelGeometrical(object):
    
    def __init__(self,
                 Omega_k:u.dimensionless_unscaled,                 
                 t_sorted:u.yr,
                 z_inverse_sorted:u.dimensionless_unscaled,
                 r_d:u.Mpc,
                ):
        
        # evolution. z(t) equivalent to a(t)
        self.t = t_sorted.decompose()
        self.z_t = z_inverse_sorted
        self.ln_a_t = -np.log(1+z_inverse_sorted)

        self.Omega_k = Omega_k
        
        self.init_H_and_D()

        # BAO scale
        self.r_d = r_d


    def init_H_and_D(self):
        
        d_ln_a = self.ln_a_t[1:] - self.ln_a_t[:-1]
        a_med = np.exp((self.ln_a_t[1:] + self.ln_a_t[:-1]) / 2)
        dt = self.t[1:] - self.t[:-1]
        t_med = (self.t[1:] + self.t[:-1]) / 2

        self.H_t = np.interp(self.t, t_med, d_ln_a/dt)
        self.rho_crit_t = 3/8/np.pi/c.G * self.H_t**2
        self.H0 = self.H_t[-1] # Do *not* extrapolate (spurious oscillations)

        Dc_t = np.cumsum(dt/a_med)
        D_nearest = self.z_t[-1]/self.H0
        Dc_t = Dc_t[-1]+D_nearest - Dc_t
        self.Dc_t = c.c * np.hstack((Dc_t[0], Dc_t))

        DH = c.c / self.H0
        if np.abs(self.Omega_k) < 1e-6:
            self.Dm_t = self.Dc_t
        elif self.Omega_k < 0:
            r = DH * np.abs(np.sin((self.Dc_t/DH).to_value(u.dimensionless_unscaled) * np.sqrt(-self.Omega_k)))
            self.Dm_t = r/np.sqrt(-self.Omega_k)
        else:
            r = DH * np.sinh((self.Dc_t/DH).to_value(u.dimensionless_unscaled) * np.sqrt(self.Omega_k))
            self.Dm_t = r/np.sqrt(self.Omega_k)

        self.DL_t = self.Dm_t * (1 + self.z_t)
        self.DA_t = self.Dm_t / (1 + self.z_t)

        
    def H(self, z):
        return np.interp(z, self.z_t[::-1], self.H_t[::-1])

    def comoving_distance(self, z):
        return np.interp(z, self.z_t[::-1], self.Dc_t[::-1])

    def angular_diameter_distance(self, z):
        return np.interp(z, self.z_t[::-1], self.Dm_t[::-1]) / (1+z)

    def luminosity_distance(self, z):
        return np.interp(z, self.z_t[::-1], self.Dm_t[::-1]) * (1+z)



    def compute_chi2(self, model, z, data, err_data):
        if err_data.ndim == 2:
            errors = np.nanmean(err_data, axis=0)
        else:
            errors = err_data
        chi2 = np.nanmean(((data - model)/errors)**2)
        return chi2
    
    def compute_complete_chi2(self, model, z, data, cov_matrix):
        N=len(data)
        chi2=0
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        for i in range(N):
            for j in range(N):
                chi2 += (data[i] - model[i]) * inv_cov_matrix[i,j] * (data[j] - model[j])
        return chi2/N
    
    def fit_mu(self, z, mu, err_mu):
        model = 5*np.log10(self.luminosity_distance(z)/10/u.pc).to_value(u.dimensionless_unscaled)        
        return self.compute_chi2(model, z, mu, err_mu)
    
    def fit_complete_mu(self, z, mu, covar_mu):
        model = 5*np.log10(self.luminosity_distance(z)/10/u.pc).to_value(u.dimensionless_unscaled)        
        return self.compute_complete_chi2(model, z, mu, covar_mu)

    def fit_DA(self, z, DA, err_DA):
        model = self.angular_diameter_distance(z).to_value(u.Mpc)
        return self.compute_chi2(model, z, DA, err_DA)

    def fit_DA_rd(self, z, DA_rd, err_DA_rd):
        model = self.angular_diameter_distance(z) / self.r_d
        return self.compute_chi2(model, z, DA_rd, err_DA_rd)

    def fit_H(self, z, H, err_H):
        model = self.H(z).to_value(u.km/u.s/u.Mpc)
        return self.compute_chi2(model, z, H, err_H)

    def fit_DH_rd(self, z, DH_rd, err_DH_rd):
        model = c.c/self.H(z) / self.r_d
        return self.compute_chi2(model, z, DH_rd, err_DH_rd)
    
    def DV(self, z):
        DA = self.angular_diameter_distance(z)
        DH = c.c * z / self.H(z)
        return np.power((1+z)**2 * DA**2 * DH, 1/3)

    def fit_DV_rd(self, z, DV_rd, err_DV_rd):
        model = self.DV(z) / self.r_d
        return self.compute_chi2(model, z, DV_rd, err_DV_rd)
    
    def fit_rd_DV(self, z, rd_DV, err_rd_DV):
        model = self.r_d / self.DV(z)
        return self.compute_chi2(model, z, rd_DV, err_rd_DV)
    
    def fit_theta_BAO(self, z, theta, err_theta):
        model = (self.r_d / self.angular_diameter_distance(z) * u.rad).to_value(u.deg)
        return self.compute_chi2(model, z, theta, err_theta)



# -----------------------------------------------------------------------------

class StandardModel(CosmologicalModelGeometrical):

    def __init__(self, model,
                 r_d:u.Mpc,
                 ):
        z = np.logspace(-5, 3.99, 1001)[::-1]
        super().__init__(model.Ode0, model.age(z), z, r_d) 


class FlatLCDM(StandardModel):

    def __init__(self,
                 Ol0:u.dimensionless_unscaled,
                 H0:1/u.s,
                 r_d:u.Mpc):
        model = cosmo.FlatLambdaCDM(H0, 1-Ol0, Tcmb0=cosmo.Planck18.Tcmb0, Ob0=cosmo.Planck18.Ob0)
        super().__init__(model, r_d)

# --------------------------------------------------
# ---------------------------
class OnionModel(CosmologicalModelGeometrical):
    def __init__(self,
                 Ok:u.dimensionless_unscaled,
                 H0:1/u.s,
                 r_d:u.Mpc,
                 k_st:u.dimensionless_unscaled,      #Generalized spacetime curvature to distinguish between onion models
                ):
        z_models = np.logspace(-5, 3.99, 1001)[::-1]
        if np.abs(k_st) < 1e-6:
            R0 = c.c / H0
            t = R0/c.c * (1+z_models)**-1
        elif k_st > 0:
            T = np.sqrt((1/H0**2)/(Ok - 1))
            t = T * np.asin(np.sqrt(1+1/Ok)/(1+z_models))
        else:
            T = np.sqrt((1/H0**2)/(Ok + 1))
            t = T * np.asinh(np.sqrt(-1-1/Ok)/(1+z_models))

        super().__init__(Ok, t, z_models, r_d)

# -----------------------------------------------------------------------------

