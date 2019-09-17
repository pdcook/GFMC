# GFMC.py

#====FOREWORD====#
"""
This software is a minimum working example of the Gaussian Focused Monte Carlo
method as described in the accompanying paper by Clark and Cook.

This program was written for Python 3.6.3.

This software is being provided "as is", without any express or implied warranty. In
particular, the authors do not make any representation or warranty of any kind concerning the
merchantability of this software or its fitness for any particular purpose.
"""
#========#

from math import *
from random import uniform
from collections import defaultdict
import numpy as np
import scipy.io

N       = 1000000        # number of photons in simulation

#====Sample Definition====#
n       = 1.   # refractive index of sample
g       = 0.9  # anisotropy factor of sample
mu_a    = 10   # absorption coefficient of sample in 1/cm
mu_s_   = 0    # reduced scattering coefficient of sample in 1/cm

l0      = 1064 # incident beam freespace wavelength in nm

# specify wr and make z0 such that w0 is minimized
wr = 8e-4                   # incident beam radius in cm
z0 = n*pi*wr*wr/(2*l0*1e-7)
t = 2*z0                    # thickness of sample in cm

# bin configuration
rmax, dr, zmax, dz = 2*wr, wr/102, t, t/200

if abs(wr**4-4*((z0*l0*1e-7)/pi/n)**2) < 1e-15: w0 = wr/sqrt(2)
else: w0 = sqrt((wr**2+sqrt(wr**4-4*((z0*l0*1e-7)/pi/n)**2))/2) # beam waist radius in cm
print("w0: %f cm" %w0)
print("z0: %f cm" %z0)
#zr = pi*n*w0*w0/(l0*1e-7) # rayleigh range

if g == 1: mu_s = mu_s_    # scattering coefficient of sample in 1/cm
else: mu_s = mu_s_/(1-g)   #

mu_t    = mu_a + mu_s   # total interaction coefficient of sample in 1/cm
#========#

#====Physics Functions====#

# Snell's law
def multilayer_snell(ni, nt, mu_x, mu_y, mu_z):
    ti = acos(abs(mu_z))
    tt = asin((ni*sin(ti))/nt)
    return mu_x*ni/nt, mu_y*ni/nt, (mu_z/abs(mu_z))*cos(tt)

# function used for determining reflection/transmission
def fresnel_snell(ni,nt,mu_z):
    if abs(mu_z) > 0.99999: R = ((nt-ni)/(nt+ni))**2
    else:
        ti = acos(abs(mu_z))
        # if ni*sin(ti)/nt >=1 then total internal reflection occurs, and thus R = 1
        if (ni*sin(ti))/nt >=1.: R= 1.
        else:
            tt = asin((ni*sin(ti))/nt)
            R = 0.5 * ( (sin(ti-tt)**2)/(sin(ti+tt)**2) + (tan(ti-tt)**2)/(tan(ti+tt)**2) )
    return R

# determine if a photon incident on the sample begins to propagate or is reflected
def incident_reflection(mu_z, n):
    if uniform(0,1) > fresnel_snell(1., n, mu_z): return False
    else: return True

#========#

#====Single Photon Monte Carlo====#

def MC(Absorbed, n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z, m, threshold):

    """
        This function propagates a photon from position x, y, z with direction cosines mu_x,
        mu_y, mu_z, until it is either absorbed, reflected, or transmitted. The entire weight
        of the photon ends up in one of these three bins, which this function returns alongside
        the final direction of the photon.
    """

    Reflected, Transmitted = 0, 0

    while w > 0:

        s = -log(uniform(0,1))/mu_t     # stepsize

        x += s*mu_x     #
        y += s*mu_y     # Hop
        z += s*mu_z     #

        # boundary conditions
        while z > t or z < 0:           # photon is outside of sample
            if z > t:                   # photon attempts to transmit
                if uniform(0,1) < fresnel_snell(n, 1., mu_z):
                    # photon is internally reflected
                    z = 2*t - z
                    mu_z *= -1
                else:                   # photon is transmitted
                    Transmitted += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = multilayer_snell(n, 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

            elif z < 0:                 # photon attempts to reflect/backscatter
                if uniform(0,1) < fresnel_snell(n, 1., mu_z):
                    # photon is internally reflected
                    z *= -1
                    mu_z *= -1
                else:                   # photon backscatters
                    Reflected += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = multilayer_snell(n, 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

        if w > 0:
            # partial absorption event
            deltaW = w*mu_a/mu_t
            w -= deltaW
            Absorbed[(sqrt(x*x+y*y),z)] += deltaW

        # roullette
        if w <= threshold: w = m*w if uniform(0,1) <= 1/m else 0

        # scattering event: update the photon's direction cosines only if it's weight isn't 0
        ### Spin ###
        if w > 0:
            if g == 0.: cos_theta = 2*uniform(0,1) - 1
            else: cos_theta = (1/(2*g))*(1+g*g-((1-g*g)/(1-g+2*g*uniform(0,1)))**2)

            phi = 2 * pi * uniform(0,1)
            cos_phi, sin_phi = cos(phi), sin(phi)
            sin_theta = sqrt(1. - cos_theta**2)

            if abs(mu_z) > 0.99999:
                mu_x_ = sin_theta*cos_phi
                mu_y_ = sin_theta*sin_phi
                mu_z_ = (mu_z/abs(mu_z))*cos_theta
            else:
                z_sqrt = sqrt(1 - mu_z*mu_z)
                mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta
                mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta
                mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta

            mu_x, mu_y, mu_z = mu_x_, mu_y_, mu_z_

    return Absorbed, Reflected, Transmitted, x,y,z, mu_x, mu_y, mu_z

#========#

def GFMC(FOCUS_TYPE, N, n, g, t, mu_a, mu_s, mu_t):

    A           = defaultdict(float)
    R_diffuse   = 0 # total number of diffusely reflected/backscattered photons
    R_specular  = 0 # total number of specularly reflected photons
    T_diffuse   = 0 # total number of diffusely transmitted photons
    T_direct    = 0 # total number of directly transmitted photons

    m, threshold = 10, 0.0001

    for i in range(N):
        w                   = 1         # initial weight of photon

        if FOCUS_TYPE.lower() == "gaussian":
            # fancy sampling
            eta = uniform(0,1)
            z = -log(1-eta+eta*np.exp(-(mu_t)*t))/mu_t
            wz = w0*sqrt(1+((l0*1e-7)*(z-z0)/pi/n/(w0**2))**2) # Eq. 2
        elif FOCUS_TYPE.lower() == "traditional":
            z = 0
            wz = wr
        else:
            raise(RuntimeError("FOCUS_TYPE improperly set."))

        phi_i = 2*pi*uniform(0,1)           # Eq. 17
        r_i = (wz/sqrt(2))*sqrt(-log(uniform(0,1)))

        x, y             = r_i*cos(phi_i), r_i*sin(phi_i)   # initial position of photon

        if FOCUS_TYPE.lower() == "gaussian":
            if abs(z-z0) < 1e-30:
                R = 1e30
            else:
                R = -1*(z-z0)*(1+(pi*n*w0*w0/((z-z0)*l0*1e-7))**2) # Eq. 3
            # Eq. 20
            if (z-z0) < 0:
                mu_x, mu_y, mu_z = 0-x, 0-y, R
            else:
                mu_x, mu_y, mu_z = x-0, y-0, -1*R

        elif FOCUS_TYPE.lower() == "traditional":
            mu_x, mu_y, mu_z = 0-x, 0-y, z0-z

        # normalize
        d = sqrt(mu_x*mu_x+mu_y*mu_y+mu_z*mu_z)
        mu_x/=d
        mu_y/=d
        mu_z/=d

        # Gaussian Propagation must do one extra Drop and Spin before MC can start
        if FOCUS_TYPE.lower() == "gaussian":
            # add transmittance from non-interacting photons
            trans = w*np.exp(-mu_t*t)
            if abs(mu_z) == 1:
                T_direct += trans
                w = w-trans
            else:
                T_diffuse += trans
                w = w-trans

            # Drop
            if w > 0:
                # partial absorption event
                deltaW = w*mu_a/mu_t
                w -= deltaW
                A[(sqrt(x*x+y*y),z)] += deltaW

            # roullette
            if w <= threshold: w = m*w if uniform(0,1) <= 1/m else 0


            ### Spin ###
            if w > 0:
                if g == 0.: cos_theta = 2*uniform(0,1) - 1
                else: cos_theta = (1/(2*g))*(1+g*g-((1-g*g)/(1-g+2*g*uniform(0,1)))**2)

                phi = 2 * pi * uniform(0,1)
                cos_phi, sin_phi = cos(phi), sin(phi)
                sin_theta = sqrt(1. - cos_theta**2)

                if abs(mu_z) > 0.99999:
                    mu_x_ = sin_theta*cos_phi
                    mu_y_ = sin_theta*sin_phi
                    mu_z_ = (mu_z/abs(mu_z))*cos_theta
                else:
                    z_sqrt = sqrt(1 - mu_z*mu_z)
                    mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta
                    mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta
                    mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta

                mu_x, mu_y, mu_z = mu_x_, mu_y_, mu_z_
        while w > 0:

            # Monte Carlo Photon Transport
            A, Reflected, Transmitted, x, y, z, mu_x, mu_y, mu_z = \
                MC(A, n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z, m, threshold)

            if abs(mu_z) == 1:
                R_specular += Reflected
                T_direct += Transmitted
                w = 0
                break
            else:
                R_diffuse += Reflected
                T_diffuse += Transmitted
                w = 0
                break

    # convert values to percents and return them
    return A, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N

#========#

def integrate(defdict):

    return np.sum(np.array(list(defdict.values()))/N)

def Abin2D(A, N, rmax, dr, zmax, dz):
    rs = np.arange(0,rmax,dr)
    zs = np.arange(0,zmax,dz)

    V = 2*pi*(np.arange(0,int(rmax/dr))+0.5)*dr*dr*dz

    Abins = np.zeros((ceil(zmax/dz),int(rmax/dr)))
    extrabin = 0

    for coord, weight in A.items():
        r = coord[0]
        z = coord[1]
        zi = int(np.floor(z/dz))
        ri = int(np.floor(r/dr))

        if zi <= zmax/dz - 1 and ri <= rmax/dr - 1: Abins[zi,ri] += weight/N
        else: extrabin += weight/N

    Abins = Abins/V

    print("Extra: %f" %extrabin)
    return Abins

def savedata(A_G, A_T, filename, rmax, dr, zmax, dz):
    print("Saving Data to HDF5 file...")
    data = {"Abins_G":A_G, "Abins_T":A_T, "N":N, "n":n, "g":g, "t":t, "mu_a":mu_a, "mu_s":mu_s, "l0":l0, "z0":z0, "wr":wr, "w0":w0,"rmax":rmax,"dr":dr,"zmax":zmax,"dz":dz}
    scipy.io.savemat(filename, data)

if __name__ == '__main__':

    sigfigs = len(str(N))

    A_gaussian, R_diffuse, R_specular, T_diffuse, T_direct = GFMC("gaussian", N, n, g, t, mu_a, mu_s, mu_t)
    # after all of the photons have propagated, round values and report results
    Absorptance             = round(integrate(A_gaussian), sigfigs)
    diffuse_Reflectance     = round(R_diffuse, sigfigs)
    specular_Reflectance    = round(R_specular, sigfigs)
    diffuse_Transmittance   = round(T_diffuse, sigfigs)
    direct_Transmittance    = round(T_direct, sigfigs)
    total_Reflectance       = diffuse_Reflectance + specular_Reflectance
    total_Transmittance     = diffuse_Transmittance + direct_Transmittance

    print("""Absorptance: %f\nDiffuse Reflectance: %f\nSpecular Reflectance: %f\nDiffuse \
Transmittance: %f\nDirect Transmittance: %f\nTotal Reflectance: %f\nTotal Transmittance: \
%f""" %(Absorptance, diffuse_Reflectance, specular_Reflectance, diffuse_Transmittance, \
    direct_Transmittance, total_Reflectance, total_Transmittance))

    A_traditional, R_diffuse, R_specular, T_diffuse, T_direct = GFMC("traditional", N, n, g, t, mu_a, mu_s, mu_t)

    # after all of the photons have propagated, round values and report results
    Absorptance             = round(integrate(A_traditional), sigfigs)
    diffuse_Reflectance     = round(R_diffuse, sigfigs)
    specular_Reflectance    = round(R_specular, sigfigs)
    diffuse_Transmittance   = round(T_diffuse, sigfigs)
    direct_Transmittance    = round(T_direct, sigfigs)
    total_Reflectance       = diffuse_Reflectance + specular_Reflectance
    total_Transmittance     = diffuse_Transmittance + direct_Transmittance

    print("""Absorptance: %f\nDiffuse Reflectance: %f\nSpecular Reflectance: %f\nDiffuse \
Transmittance: %f\nDirect Transmittance: %f\nTotal Reflectance: %f\nTotal Transmittance: \
%f""" %(Absorptance, diffuse_Reflectance, specular_Reflectance, diffuse_Transmittance, \
    direct_Transmittance, total_Reflectance, total_Transmittance))

    # bin data
    Abins_G2D = Abin2D(A_gaussian, N, rmax, dr, zmax, dz)
    Abins_T2D = Abin2D(A_traditional, N, rmax, dr, zmax, dz)

    # save data to file
    filename = "../plots/z0_%f_wr_%f_w0_%f_mua_%f_mus_%f.mat" %(z0, wr, w0, mu_a, mu_s)
    savedata(Abins_G2D, Abins_T2D, filename, rmax, dr, zmax, dz)



