# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 19:09:04 2025

@author: ychuang
"""

# -*-Python-*-
# Created by bgrierson at 14 Dec 2016  09:36

# Demonstration of the polyexp fitting in OMFITprofiles

from pylab import random
import numpy as np
import matplotlib.pyplot as plt
import lmfit



def mtanh(x, A, B, xsym, hwid, core, edge):
    """
    Groebner APS 2004
    y = A * MTANH(alpha,z) + B
    MTANH(alpha,z) = [(1+alpha*z)*exp(z) - exp(-z)] / [exp(z) + exp(-z)]
    generalized to have a arbitrary order poly on numerator exp(z) and exp(-z)
    to allow for more variation in the core and edge.  Re-formulated as
    MTANH(alpha,z) = [nc*exp(z) - ne*exp(-z)] / [exp(z) + exp(-z)]
    where nc and nc are the core and edge polynomials
    Pedestal height is A+B
    Pedestal offset is A-B
    Location of the knee is xsym-hwid
    Location of the foot is xsym+hwid
    :param x: x-axis values
    :param A: Amplitude
    :param B: Offset
    :param xsym: Symmetry location
    :param hwid: half-width
    :param core: core polynomial coefficients [1.0, ...]
    :param edge: edge polynomial coefficients [1.0, ...]
    :return: Function y(x)
    """
    # Z shifted function
    z = (xsym - x) / hwid
    # Core polynomial
    nc = core[0]
    for i in range(1, len(core)):
        nc += core[i] * z ** (i)
    # Edge polynomial
    ne = edge[0]
    for i in range(1, len(edge)):
        ne += edge[i] * z ** (i)
    # mtanh in all its glory
    mt = (nc * np.exp(z) - ne * np.exp(-z)) / (np.exp(z) + np.exp(-z))
    # Final function
    y = A * mt + B
    return y


# Turn parameters dictionary into inputs for mtanh
def params_to_fun(params):
    """
    Turn parameters dictionary into inputs for mtanh
    :param params: lmfit parameters object
    :return: parameters as a series of floats
    """
    xsym = params['xsym'].value
    hwid = params['hwid'].value
    pedestal = params['pedestal'].value
    offset = params['offset'].value
    B = (pedestal + offset) / 2.0
    A = B - offset
    nc = 0
    ne = 0
    for key in params:
        if 'core' in key:
            nc = nc + 1
        if 'edge' in key:
            ne = ne + 1
    core = np.zeros(nc)
    edge = np.zeros(ne)
    for key in params:
        if 'core' in key:
            core[int(key.split('core')[1])] = params[key].value
        if 'edge' in key:
            edge[int(key.split('edge')[1])] = params[key].value
    return A, B, xsym, hwid, core, edge


def funcm(params, xm, ym, em):
    """
    Cost function "residual" for lmfit leastsq solution
    :param params: lmfit parameters dictionary
    :param xm: x coordinate of measured data
    :param ym: measured data value
    :param em: measured data uncertainty
    :return: weighted residual
    """
    A, B, xsym, hwid, core, edge = params_to_fun(params)
    y = mtanh(xm, A, B, xsym, hwid, core, edge)
    r = (ym - y) / em
    return r


def funcm_b(params, xm, ym):
    """
    Cost function "residual" for lmfit leastsq solution
    :param params: lmfit parameters dictionary
    :param xm: x coordinate of measured data
    :param ym: measured data value
    :param em: measured data uncertainty
    :return: weighted residual
    """
    A, B, xsym, hwid, core, edge = params_to_fun(params)
    y = mtanh(xm, A, B, xsym, hwid, core, edge)




with open('49404_82.dat', mode='r') as dfile:
    lines = dfile.readlines()

profiles = {}
nlines_tot = len(lines)
x_n = np.zeros(nlines_tot)
te = np.zeros(nlines_tot)

i = 0

while i < nlines_tot:
    r_line = lines[i].split()
    x_n[i] = float(r_line[0])
    te[i] = float(r_line[1])
    i += 1




# ----
# Set default profile parameters
# ----
xsym = 1.0
hwid = 0.02
core = np.array([1.0, 1e-2])
edge = np.array([1.0, -1e-2, -3e-4])
pedestal = 5.0
offset = 0.1
B = (pedestal + offset) / 2.0
A = B - offset

# ----
#  Form "data" with noise and uncertainty
# ----


# x = np.linspace(0.0, 1.2, 101)
# ydata = mtanh(x, A, B, xsym, hwid, core, edge) + 0.5 * random(len(x))
# yerr = np.repeat(0.1, len(x))


x = x_n
ydata = te
yerr = np.repeat(0.001, len(x))




# ----
# Create the lmfit paramters with guesses near the true values
# ----
params = lmfit.Parameters()
params.add('xsym', value=1.38)
params.add('hwid', value=0.12)
params.add('core0', value=0.95)
params.add('core1', value=0.02)
params.add('edge0', value=0.95)
params.add('edge1', value=0.02)
params.add('pedestal', value=0.3)
params.add('offset', value=0.01)

# Do the fit
fitter = lmfit.Minimizer(funcm, params, fcn_args=(x, ydata, yerr))
out = fitter.minimize()
print(lmfit.fit_report(out))

# Get the fit result
A, B, xsym, hwid, core, edge = params_to_fun(out.params)
yfit = mtanh(x, A, B, xsym, hwid, core, edge)

# Compute the residual
# resid = funcm(out.params, x, ydata, yerr)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(x, ydata)
ax[0].plot(x, yfit, label='Fit')
ax[0].legend()
# ax[1].plot(x, (ydata - yfit) / yerr, marker='o', ls='None', label='Residual')
# ax[1].axhline(0.0, ls='dashed')
ax[1].legend()