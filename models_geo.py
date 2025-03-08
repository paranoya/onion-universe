
import numpy as np
#import scipy
from scipy.special import zeta
from time import time
from astropy import cosmology as cosmo
from astropy import units as u
from astropy import constants as c

# -----------------------------------------------------------------------------


class CosmologicalModel(object):
    
    def __init__(self,
                 Omega_k:u.dimensionless_unscaled,                 
                 t_sorted:u.yr,
                 z_inverse_sorted:u.dimensionless_unscaled,
                 r_d:u.Mpc,
                ):
        
        # evolution
        
        self.t = t_sorted.decompose()
        self.z_t = z_inverse_sorted
        self.ln_a_t = -np.log(1+z_inverse_sorted)
        
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
        #print('aaa', Omega_k, self.Ode0, self.Om0, self.Ogamma0, (1+self.neutrinos))
        if np.abs(self.Omega_k) < 1e-6:
            #print(f'{Omega_k} is flat')
            self.Dm_t = self.Dc_t
        elif self.Omega_k < 0:
            #print(f'{Omega_k} is closed')
            r = DH * np.abs(np.sin((self.Dc_t/DH).to_value(u.dimensionless_unscaled) * np.sqrt(-self.Omega_k)))
            self.Dm_t = r/np.sqrt(-self.Omega_k)
        else:
            #print(f'{Omega_k} is open')
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


class StandardModel(CosmologicalModel):

    def __init__(self, model,
                 r_d:u.Mpc,
                 Y:u.dimensionless_unscaled=0.24):
        z = np.logspace(-5, 3.99, 1001)[::-1]
        super().__init__(model.Ode0, model.Om0, model.Ob0, Y, model.Ogamma0, model.Neff,
                         model.age(z), z, model.Ode(z), model.Om(z), model.Ogamma(z),
                         r_d)

class FlatLCDM(StandardModel):

    def __init__(self,
                 Ol0:u.dimensionless_unscaled,
                 H0:1/u.s,
                 r_d:u.Mpc):
        model = cosmo.FlatLambdaCDM(H0, 1-Ol0, Tcmb0=cosmo.Planck18.Tcmb0, Ob0=cosmo.Planck18.Ob0)
        super().__init__(model, r_d)

# -----------------------------------------------------------------------------


class Coasting(CosmologicalModel):

    def __init__(self,
                 Ok:u.dimensionless_unscaled,
                 H0:1/u.s,
                 r_d:u.Mpc,
                 Tcmb0:u.K=cosmo.Planck18.Tcmb0,
                 Neff:u.dimensionless_unscaled=cosmo.Planck18.Neff,
                 Ob0:u.dimensionless_unscaled=cosmo.Planck18.Ob0,
                 Y:u.dimensionless_unscaled=0.24,
                 eta:u.dimensionless_unscaled=6e-10,):

        # Present time
        #print('t0', (1/H0).to_value(u.Gyr), 'Gyr')

        ## photons
        rho_gamma0 = 4*c.sigma_sb * Tcmb0**4 / c.c**3
        Ogamma0 = (rho_gamma0 * 8*np.pi*c.G / 3/H0**2).to_value(u.dimensionless_unscaled)

        ## masless neutrinos
        neutrinos = (7/8)*Neff*(4/11)**(4/3)
        Or0 = Ogamma0 * (1 + neutrinos)

        ## (non-relativistic) matter
        mu0 = ((1 - Ok) / Or0 - 3) / 2
        t0_teq = np.sqrt(5*mu0**4 / (3+2*mu0))
        #print('Omega_r', Or0, Ogamma0*(1+neutrinos))

        #print('mu, zeq', mu0, t0_teq - 1)

        Om0 = mu0 * Or0
        rho_m0 = Om0 * 3*H0**2/(8*np.pi*c.G)
        #print('Omega_m', Om0, mu0 / (3 + 2 * mu0) * (1-Ok))

        n_gamma0 = 16*np.pi*zeta(3) * (c.k_B*Tcmb0 / c.h/c.c)**3
        rho_b0 = eta * c.m_p * n_gamma0
        Ob0 = (rho_b0 * 8*np.pi*c.G / 3/H0**2).to_value(u.dimensionless_unscaled)
        #print('Omega_b', Ob0)

        # Evolution

        mu = mu0 * np.hstack((.5*np.logspace(-4, -1e-4, 1001),
                              (1 - .5*np.logspace(-6, -1e-4, 1001))[::-1]))

        t_teq = np.sqrt(5*mu**4 / (3+2*mu))

        z = t0_teq / t_teq - 1
        t = 1/H0 / (1+z)

        Or = (1 - Ok) / (3 + 2 * mu)
        Om = (1 - Ok) * mu / (3 + 2 * mu)
        Ob = Ob0/Om0 * Om
        Ode = 2*Or + Om

        super().__init__(2*Or0+Om0, Om0, Ob0, Y, Ogamma0, Neff,
                         t, z, Ode, Om, Or/(1+neutrinos),
                         r_d)

# -----------------------------------------------------------------------------


flcdm = FlatLCDM(Ol0=cosmo.Planck18.Ode0, H0=cosmo.Planck18.H0, r_d=140*u.Mpc)
coasting = Coasting(Ok=-.05, H0=70*u.km/u.s/u.Mpc, r_d=140*u.Mpc, Tcmb0=cosmo.Planck18.Tcmb0, Neff=cosmo.Planck18.Neff, Ob0=cosmo.Planck18.Ob0)