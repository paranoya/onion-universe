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
                 Ode0, Om0, Ob0, Y, Ogamma0,
                 t_sorted, z_inverse_sorted, Ode_t, Om_t, Ogamma_t,
                 r_d,
                ):
        
        # present time
        
        self.Ode0 = Ode0
        self.Om0 = Om0
        self.Ob0 = Ob0
        self.Y = Y
        self.Ogamma0 = Ogamma0

        # TODO: implement neutrino masses
        #self.neutrinos = (7/8)*Neff*(4/11)**(4/3)
        #self.Onu0 = Ogamma0 * self.neutrinos
        #self.Or0 = Ogamma0 + self.Onu0

        # evolution
        
        self.t = t_sorted
        self.z_t = z_inverse_sorted
        self.ln_a_t = -np.log(1+z_inverse_sorted)
        self.ln_Ode_t = np.log(Ode_t)
        self.ln_Om_t = np.log(Om_t)
        self.ln_Ogamma_t = np.log(Ogamma_t)
        
        self.init_H()
        self.init_CMB(Ogamma_t * self.rho_crit_t,
                      Om_t * self.rho_crit_t * Ob0/Om0)
        
        # BAO scale
        
        self.r_d = r_d
        
        
    def Ode(self, z):
        ln_a = -np.log(1+z)
        return(np.exp(np.interp(ln_a, self.ln_a_t, self.ln_Ode_t)))
        
    def Om(self, z):
        ln_a = -np.log(1+z)
        return(np.exp(np.interp(ln_a, self.ln_a_t, self.ln_Om_t)))
        
    def Ob(self, z):
        return(self.Om(z) * self.Ob0/self.Om0)
        
    def Ogamma(self, z):
        ln_a = -np.log(1+z)
        return(np.exp(np.interp(ln_a, self.ln_a_t, self.ln_Ogamma_t)))
                
    def Onu(self, z):
        return(self.Ogamma(z) * self.Onu0/self.Ogamma0)
        
        
    def init_H(self):
        
        d_ln_a = self.ln_a_t[1:] - self.ln_a_t[:-1]
        dt = self.t[1:] - self.t[:-1]
        t_med = (self.t[1:] + self.t[:-1]) / 2
        self.H_t = np.interp(self.t, t_med, d_ln_a/dt)
        self.rho_crit_t = 3/8/np.pi/c.G * self.H_t**2
        self.H0 = self.H_t[-1] # Do *not* extrapolate (spurious oscillations)
    

    def init_CMB(self, rho_gamma_t, rho_b_t):
        
        Tcmb_t = np.power(c.c**3/(4*c.sigma_sb) * rho_gamma_t, 1/4)
        self.Tcmb_t = Tcmb_t

        n_H_t = (1 - self.Y) * rho_b_t / c.m_p
        y_t = (2/n_H_t) * np.power(2*np.pi*c.m_e*c.k_B* Tcmb_t /c.h**2, 1.5) * np.exp(-13.6*u.eV/c.k_B/ Tcmb_t)
        x_j = np.linspace(1e-12, 1-1e-12, 1001)
        y_j = x_j**2/(1-x_j)
        self.x_ion_t = np.interp(y_t, y_j, x_j)
        
        t_coll_t = 1/self.x_ion_t / n_H_t /c.sigma_T/c.c
        self.t_cmb = np.min(self.t[t_coll_t > self.t])

# -----------------------------------------------------------------------------


class StandardModel(CosmologicalModel):

    def __init__(self, model, r_d, Y=0.24):
        z = np.logspace(-5, 3.99, 1001)[::-1]
        super().__init__(model.Ode0, model.Om0, model.Ob0, Y, model.Ogamma0,
                         model.age(z), z, model.Ode(z), model.Om(z), model.Ogamma(z),
                         r_d)

class FlatLCDM(StandardModel):

    def __init__(self, Ol0, H0, r_d):
        model = cosmo.FlatLambdaCDM(H0, 1-Ol0, Tcmb0=cosmo.Planck18.Tcmb0, Ob0=cosmo.Planck18.Ob0)
        super().__init__(model, r_d)

# -----------------------------------------------------------------------------


class Coasting(CosmologicalModel):

    def __init__(self, Ok, H0, r_d, Tcmb0, Neff, Ob0, Y=0.24, eta=6e-10):

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

        mu = np.logspace(-4, 0, 501) * mu0

        t_teq = np.sqrt(5*mu**4 / (3+2*mu))

        z = t0_teq / t_teq - 1
        t = 1/H0 / (1+z)

        Or = (1 - Ok) / (3 + 2 * mu)
        Om = (1 - Ok) * mu / (3 + 2 * mu)
        Ob = Ob0/Om0 * Om
        Ode = 2*Or + Om

        super().__init__(0, Om0, Ob0, Y, Ogamma0,
                         t, z, Ode, Om, Or/(1+neutrinos),
                         r_d)

# -----------------------------------------------------------------------------


flcdm = FlatLCDM(Ol0=cosmo.Planck18.Ode0, H0=cosmo.Planck18.H0, r_d=140*u.Mpc)
coasting = Coasting(Ok=-.05, H0=70*u.km/u.s/u.Mpc, r_d=140*u.Mpc, Tcmb0=cosmo.Planck18.Tcmb0, Neff=cosmo.Planck18.Neff, Ob0=cosmo.Planck18.Ob0)

# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------
