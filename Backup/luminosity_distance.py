#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:43:43 2020

@author: yago
"""

# %% Init

from __future__ import division
from __future__ import print_function

import os
import numpy as np
from matplotlib import pyplot as plt

from astropy import cosmology as cosmo
# from astropy import units as u
from astropy.table import Table
# from astropy.visualization import quantity_support
# quantity_support()

# LCDM = cosmo.Planck15
# H0 = LCDM.H0

base_path = os.path.dirname(os.path.abspath(__file__))

SNe_dirty = Table.read(os.path.join(base_path, 'data', 'OpenSupernovaCatalog_Ia.csv'))
dirty = np.any([SNe_dirty['z'].mask,
                SNe_dirty['Mmax'].mask,
                SNe_dirty['mmax'].mask], axis=0)
SNe = SNe_dirty[~dirty]
z_SNe = np.array([float(x.split(',')[0]) for x in SNe['z'].data])
# dL_SNe = SNe['dL (Mpc)'].data
dL_SNe = 10**((SNe['mmax'].data-SNe['Mmax'].data)/5-5)  # Mpc

# %% Numerical integration

# Omega_r = LCDM.Onu0 + LCDM.Ogamma0
# Omega_nr = LCDM.Odm0 + LCDM.Ob0
# Omega_L = LCDM.Ode0
# initial_z = 1e6
# current_a = 1/(1+initial_z)
# current_t = LCDM.age(initial_z)
# a = []
# t = []
# step_factor = 1e-3
# while current_a < 1.01:
#     a.append(current_a)
#     t.append(current_t.to_value(u.Gyr))
#     # print(current_t, t[-1])
#     da_dt = H0*np.sqrt(Omega_r/current_a**2 + Omega_nr/current_a + Omega_L*current_a**2)
#     da = step_factor*current_a
#     current_a += da
#     current_t += da/da_dt
# t = np.array(t)


# %% Expansion history

# plt.plot(t, t/t[-1], 'k:')
# plt.plot(t, a, 'k-')
# plt.plot(LCDM.age(z).value, 1/(1+z), 'r:')
# plt.ylabel('a')
# plt.xlabel('t [Gyr]')
# plt.show()


# %% Luminosity distance

z = np.logspace(-3, 4, 200)

H0 = 73
H0_onion = 69
d_H = 3e5/H0  # c/H0, in Mpc
d_H_onion = 3e5/H0_onion  # c/H0, in Mpc

LCDM = cosmo.FlatLambdaCDM(H0=H0, Om0=.3)
LCDM_lo = cosmo.FlatLambdaCDM(H0=71.8, Om0=.3)
LCDM_hi = cosmo.FlatLambdaCDM(H0=74.2, Om0=.3)
EdS = cosmo.FlatLambdaCDM(H0=H0, Om0=1)

chi_onion = np.log(1+z)
r_onion = np.abs(np.sin(chi_onion))
dL_onion = d_H_onion * r_onion * (1+z)

dL_LCDM = LCDM.luminosity_distance(z)
dL_LCDM_lo = LCDM_lo.luminosity_distance(z)
dL_LCDM_hi = LCDM_hi.luminosity_distance(z)
dL_EdS = EdS.luminosity_distance(z)

plt.plot(z, 3e5*z/dL_LCDM, 'r-', label='LCDM')
plt.plot(z, 3e5*z/dL_onion, 'k-', label='onion')

plt.xlim(0, 10.0)
plt.ylim(20, 100)

plt.grid(axis='both', which='both', alpha=.1)
plt.xlabel(r'$z$')
plt.ylabel(r'$cz/d_L$ [km/s/Mpc]')
# plt.title(r'$H_0$ = {} km/s/Mpc'.format(H0))
plt.legend()
plt.show()

# %% Angular diameter distance

dA_LCDM = LCDM.angular_diameter_distance(z)
dA_LCDM_lo = LCDM_lo.angular_diameter_distance(z)
dA_LCDM_hi = LCDM_hi.angular_diameter_distance(z)
dA_EdS = EdS.angular_diameter_distance(z)

dA_onion = d_H_onion * r_onion / (1+z)

# plt.plot(1+z_SNe, dL_SNe/(1+z_SNe)**2, 'k.', alpha=.1)

plt.plot(z, dA_LCDM, 'r-', label='LCDM')
plt.plot(z, dA_LCDM_lo, 'r:', label='LCDM low H')
plt.plot(z, dA_LCDM_hi, 'r:', label='LCDM high H')
plt.plot(z, dA_EdS, 'b:', label='EdS')
plt.plot(z, dA_onion, 'k-', label='onion')

plt.xlim(0, 1.5)
plt.xlim(1e-3, 3e3)
plt.xscale('log')
# plt.ylim(0, 1800)
plt.yscale('log')

plt.grid(axis='both', which='both', alpha=.1)
# plt.xlabel(r'$1 + z$')
plt.xlabel(r'$z$')
plt.ylabel(r'$d_A$ [Mpc]')
plt.title(r'$H_0$ = {} km/s/Mpc'.format(H0))
plt.legend()
plt.show()


# %% Ratio

# plt.plot(z, dA_onion/dA_LCDM, 'k-')
# plt.xlabel(r'$z$')
# plt.xlim(0, 1.5)
# plt.show()

# %% Bye
print("... Paranoy@ Rulz!")

# %%
