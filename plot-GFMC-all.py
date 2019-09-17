import sys
import matplotlib.pyplot as plt
import argparse
import scipy.io
from collections import defaultdict
import numpy as np
import h5py
from math import *
import os
import matplotlib

# number of ticks on x-axis for line plots
numxticks = 3
# number of ticks on y-axis for line plots
numyticks = 3

# number of ticks on z-axis for matrix plots
numzticks = 3
# number of ticks on r-axis for matrix plots
numrticks = 5

def factor(minimum,x):
    factors = []
    assert(type(x)==int)
    for i in range(minimum,x+1):
        if x%i == 0: factors.append(i)
    return np.array(factors)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def power_along_z(Abins1, Abins2, rmax, dr, zmax, dz):
    # graph power absorbed along z-axis:
    zs = np.arange(0,zmax,dz)
    fig, ax = plt.subplots()
    ax.semilogy(zs[:-1],Abins1[:-1,0], color = 'k', linestyle = ':', lw=2)
    ax.semilogy(zs[:-1],Abins2[:-1,0], color = 'k', linestyle = '-', lw=2)
    ax.legend(['Gaussian','Traditional'], prop={"size":14})
    ax.set_title("Absorbed Power Along Beam Axis", size = 18)
    ax.set_xticks(np.linspace(0,zmax,numxticks,endpoint=True))
    ax.set_xticklabels([("%.2f" %f) for f in np.linspace(0,zmax,numxticks,endpoint=True)*1e4],size=12)    # convert to um
    ax.set_xlabel("$z~(\mathrm{\mu m})$", size = 14)
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(10**np.floor(np.log10(current_ymin)),10**np.ceil(np.log10(current_ymax)))
    current_ymin, current_ymax = ax.get_ylim()
    current_yticks = 10**np.arange(np.floor(np.log10(current_ymin)),np.ceil(np.log10(current_ymax))+1)
    ax.set_yticks(current_yticks)
    current_yticks, _ = plt.yticks()
    # calculate numyticks so that all yticks are exactly a power of 10 (this prevents nasty-looking 3.16x10^7 ticks)
    candidates = factor(numyticks,int(np.log10(np.max(current_yticks))-np.log10(np.min(current_yticks))))
    new_numyticks = find_nearest(candidates, numyticks)+1
    if new_numyticks != numyticks: print("Number of y ticks changed from %d to %d" %(numyticks, new_numyticks))
    yticks = np.logspace(np.log10(np.min(current_yticks)),np.log10(np.max(current_yticks)),new_numyticks,endpoint=True)
    #ax.set_yticks(np.logspace(np.log10(np.min(current_yticks)),np.log10(np.max(current_yticks)),new_numyticks,endpoint=True))
    ax.set_yticklabels([("$10^{%.0f}$" %np.log10(f)) if (f in yticks) else "" for f in current_yticks],size=12)
    ax.set_ylabel("Absorbed Power (AU)", size = 14)
    ax.tick_params(axis='both', which='major', length = 7, width = 1)
    ax.tick_params(axis='both', which='minor', length = 4, width = 1)
    plt.tight_layout()
    plt.savefig("p_vs_z.ps",dpi=500)
    plt.savefig("p_vs_z.png",dpi=500)

def A_vs_r_vs_z(Abins1, Abins2, rmax, dr, zmax, dz):

    zs = np.arange(0,zmax,dz)
    rs = np.arange(0,rmax,dr)
    rs_mirrored = np.concatenate((-1*np.flip(rs)[:-1],rs[1:]))

    Abins1_mirrored = np.concatenate((Abins1[:,::-1],Abins1), axis=1)
    Abins2_mirrored = np.concatenate((Abins2[:,::-1],Abins2), axis=1)

    vmin = np.min([np.min(Abins1[Abins1>0]),np.min(Abins2[Abins2>0])])
    vmax = np.max([np.max(Abins1[Abins1>0]),np.max(Abins2[Abins2>0])])

    cmap = matplotlib.cm.gist_yarg
    cmap.set_bad('white',1.)

    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots()
    ax.imshow(Abins1_mirrored,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
    ax.set_title("Gaussian $A(r,z)$", size = 18)
    ax.set_yticks(np.linspace(0,zmax,numzticks,endpoint=True)/dz)
    ax.set_yticklabels([("%.2f" %f) for f in np.linspace(0,zmax,numzticks,endpoint=True)*1e4],size=12)    # convert to um
    ax.set_ylabel("$z~(\mathrm{\mu m})$", size = 14)
    ax.set_xticks((np.linspace(-rmax,rmax,numrticks,endpoint=True)+rmax-dr/2)/dr)
    ax.set_xticklabels([("%.2f" %f) for f in np.linspace(-rmax,rmax,numrticks,endpoint=True)*1e4],size=12) # convert to um
    ax.set_xlabel("$r~(\mathrm{\mu m})$", size = 14)
    ax.tick_params(axis='both', which='major', length = 7, width = 1)
    ax.tick_params(axis='both', which='minor', length = 4, width = 1)
    plt.tight_layout()
    plt.savefig("A_G.ps",dpi=500)
    plt.savefig("A_G.png",dpi=500)
    fig, ax = plt.subplots()
    ax.imshow(Abins2_mirrored,norm=LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
    ax.set_title("Traditional $A(r,z)$", size = 18)
    ax.set_yticks(np.linspace(0,zmax,numzticks,endpoint=True)/dz)
    ax.set_yticklabels([("%.2f" %f) for f in np.linspace(0,zmax,numzticks,endpoint=True)*1e4],size=12)    # convert to um
    ax.set_ylabel("$z~(\mathrm{\mu m})$", size = 14)
    ax.set_xticks((np.linspace(-rmax,rmax,numrticks,endpoint=True)+rmax-dr/2)/dr)
    ax.set_xticklabels([("%.2f" %f) for f in np.linspace(-rmax,rmax,numrticks,endpoint=True)*1e4],size=12) # convert to um
    ax.set_xlabel("$r~(\mathrm{\mu m})$", size = 14)
    ax.tick_params(axis='both', which='major', length = 7, width = 1)
    ax.tick_params(axis='both', which='minor', length = 4, width = 1)
    plt.tight_layout()
    plt.savefig("A_T.ps",dpi=500)
    plt.savefig("A_T.png",dpi=500)

def second_moment_width_vs_z(Abins1, Abins2, N, rmax, dr, zmax, dz):
    zs = np.arange(0,zmax,dz)
    rs = np.arange(0,rmax,dr)
    if abs(rs[-1]-rmax) < 1e-9:
        rs = rs[:-1]
    rs_mirrored = np.concatenate((-1*np.flip(rs),rs))

    d4s_1 = np.empty(np.size(zs))
    d4s_2 = np.empty(np.size(zs))


    Abins1_mirrored = np.concatenate((Abins1[:,::-1],Abins1), axis=1)
    Abins2_mirrored = np.concatenate((Abins2[:,::-1],Abins2), axis=1)

    for zi in range(zs.size):
        d4s_1[zi] = 4*sqrt(np.sum(((rs+dr*0.5)**3)*Abins1[zi,:])*pi/np.sum(2*pi*(rs+dr*0.5)*Abins1[zi,:]))
        d4s_2[zi] = 4*sqrt(np.sum(((rs+dr*0.5)**3)*Abins2[zi,:])*pi/np.sum(2*pi*(rs+dr*0.5)*Abins2[zi,:]))

    fig, ax = plt.subplots()
    ax.plot(zs[:-1],d4s_1[:-1], color = 'k', linestyle = ':', lw =2)
    ax.plot(zs[:-1],d4s_2[:-1], color = 'k', linestyle = '-', lw =2)
    ax.legend(["Gaussian","Traditional"], prop={"size":14})
    ax.set_title("Second Moment Width", size=18)
    ax.set_xticks(np.linspace(0,zmax,numxticks,endpoint=True))
    ax.set_xticklabels([("%.2f" %f) for f in np.linspace(0,zmax,numxticks,endpoint=True)*1e4],size=12)   # convert to um
    ax.set_xlabel("$z~(\mathrm{\mu m})$", size = 14)
    current_yticks,_ = plt.yticks()
    ax.set_yticks(np.linspace(0,np.max(current_yticks),numyticks,endpoint=True))
    ax.set_yticklabels([("%.2f" %f) for f in np.linspace(0,np.max(current_yticks),numyticks,endpoint=True)*1e4],size=12) # convert to um
    ax.set_ylabel("$D4\sigma~(\mathrm{\mu m})$", size = 14)
    ax.tick_params(axis='both', which='major', length = 7, width = 1)
    ax.tick_params(axis='both', which='minor', length = 4, width = 1)
    plt.tight_layout()
    plt.savefig("second_moment_width_vs_z.ps",dpi=500)
    plt.savefig("second_moment_width_vs_z.png",dpi=500)

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help="Full path to datafile created by GFMC.")
    args = parser.parse_args()
    try:
        data = h5py.File(args.datafile,'r')
        A_G = np.array(data["Abins_G"])
        A_T = np.array(data["Abins_T"])
        N = np.asscalar(np.array(data["N"]))
        n = np.asscalar(np.array(data["n"]))
        g = np.asscalar(np.array(data["g"]))
        t = np.asscalar(np.array(data["t"]))
        mu_a = np.asscalar(np.array(data["mu_a"]))
        mu_s = np.asscalar(np.array(data["mu_s"]))
        l0 = np.asscalar(np.array(data["l0"]))
        z0 = np.asscalar(np.array(data["z0"]))
        wr = np.asscalar(np.array(data["wr"]))
        w0 = np.asscalar(np.array(data["w0"]))
        rmax = np.asscalar(np.array(data["rmax"]))
        dr = np.asscalar(np.array(data["dr"]))
        zmax = np.asscalar(np.array(data["zmax"]))
        dz = np.asscalar(np.array(data["dz"]))
    except:
        data = scipy.io.loadmat(args.datafile)    # read data
        A_G = data["Abins_G"]
        A_T = data["Abins_T"]
        N = np.asscalar(np.array(data["N"]))
        n = np.asscalar(np.array(data["n"]))
        g = np.asscalar(np.array(data["g"]))
        t = np.asscalar(np.array(data["t"]))
        mu_a = np.asscalar(np.array(data["mu_a"]))
        mu_s = np.asscalar(np.array(data["mu_s"]))
        l0 = np.asscalar(np.array(data["l0"]))
        z0 = np.asscalar(np.array(data["z0"]))
        wr = np.asscalar(np.array(data["wr"]))
        w0 = np.asscalar(np.array(data["w0"]))
        rmax = np.asscalar(np.array(data["rmax"]))
        dr = np.asscalar(np.array(data["dr"]))
        zmax = np.asscalar(np.array(data["zmax"]))
        dz = np.asscalar(np.array(data["dz"]))
    print("""\nLoaded Data Parameters:
    N = %d photons
    n = %f
    g = %f
    t = %f cm
    mu_a = %f 1/cm
    mu_s = %f 1/cm
    l0 = %f cm
    z0 = %f cm
    wr = %f cm
    w0 = %f cm
    rmax = %f cm
    dr = %f cm
    zmax = %f cm
    dz = %f cm
    """ %(N, n, g, t, mu_a, mu_s, l0, z0, wr, w0, rmax, dr, zmax, dz))


    # graph data
    folder =  "z0_%f_wr_%f_w0_%f_mua_%f_mus_%f" %(z0,wr,w0,mu_a,mu_s)
    if os.path.isdir(folder):
        response = input("WARN: Plots have already been made for this configuration. Overwrite? [Y/N] >>> ")
        if response.lower() != 'y': sys.exit()
    else:
        os.system("mkdir %s" %folder)
    os.chdir(folder)
    second_moment_width_vs_z(A_G, A_T, N, rmax, dr, zmax, dz)
    power_along_z(A_G, A_T, rmax, dr, zmax, dz)
    A_vs_r_vs_z(A_G, A_T, rmax, dr, zmax, dz)

