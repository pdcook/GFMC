// nvcc -x cu -arch=sm_60 -std=c++14 GFMLMC.cu -o GFMLMC.o -ccbin /usr/bin/g++-6 -I /usr/include/hdf5/serial/ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/ -lhdf5 -lstdc++fs

#include "cmdlineparse.h"
#include "CUDAGFMLMC.h"
#include <iostream>
#include <fstream>
#include <thrust/tuple.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <future>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#include "hdf5.h"
#include <stdlib.h>
#include <experimental/filesystem>
#define _USE_MATH_DEFINES

float aggregate(float* X, int N_r, int N_z)
{
    float tot = 0;
    for (int r_i = 0; r_i < N_r; r_i++)
    {
        for (int z_i = 0; z_i < N_z; z_i++)
        {
            tot += X[z_i*N_r + r_i];
        }
    }
    return tot;
}

float* array_uncertainty(float* X, int N_r, int N_z, int N)
{
    float* X_unc = new float[N_r*N_z];
    for (int r_i = 0; r_i < N_r; r_i++)
    {
        for (int z_i = 0; z_i < N_z; z_i++)
        {
            X_unc[z_i*N_r + r_i] = sqrt(X[z_i*N_r + r_i]/N);
        }
    }
    return X_unc;
}

float* normalize(float* X, int N_r, int N_z, float dr, float dz)
{
    for (int r_i = 0; r_i < N_r; r_i++)
    {
        for (int z_i = 0; z_i < N_z; z_i++)
        {
            X[z_i*N_r + r_i] = X[z_i*N_r + r_i] / (2*M_PI*(r_i+0.5)*dr*dr*dz);
        }
    }

    return X;

}

void write_HDF5(string s_filename, float* G_data1D, float* T_data1D, int N_r, int N_z, float N, float n, float g, float t, float mu_a, float mu_s, float l0, float z0, float wr, float rmax, float dr, float zmax, float dz)
{
    float w0;
    // calculate w0 in order to save it
    if (abs(wr*wr*wr*wr-4*pow(((z0*l0*1e-7)/M_PI/n),2)) < 1e-15)
    {
        w0 = wr / sqrtf(2);
    }
    else
    {
        w0 = sqrt((wr*wr+sqrt(wr*wr*wr*wr-4*pow(((z0*l0*1e-7)/M_PI/n),2)))/2);
    }

    int rank = 2;

    char* filename = new char[s_filename.length()+1];
    strcpy(filename, s_filename.c_str());

    hid_t file, G_dataset, T_dataset, N_dataset, n_dataset, g_dataset, t_dataset, mu_a_dataset, mu_s_dataset, l0_dataset, z0_dataset, wr_dataset, w0_dataset, rmax_dataset, dr_dataset, zmax_dataset, dz_dataset;
    hid_t datatype, array_dataspace, scalar_dataspace;
    hsize_t dimsf[2];
    hsize_t scalar_dimsf[1];
    herr_t status;

    // convert 1D data into 2D array, each row has a static z value, each column has a static r value
    float G_data2D[N_z][N_r];

    for (int r_i = 0; r_i < N_r; r_i++)
    {
        for (int z_i = 0; z_i < N_z; z_i++)
        {
            G_data2D[z_i][r_i] = G_data1D[z_i*N_r + r_i];
        }
    }

    float T_data2D[N_z][N_r];

    for (int r_i = 0; r_i < N_r; r_i++)
    {
        for (int z_i = 0; z_i < N_z; z_i++)
        {
            T_data2D[z_i][r_i] = T_data1D[z_i*N_r + r_i];
        }
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dimsf[0] = N_z;
    dimsf[1] = N_r;
    scalar_dimsf[0] = 1;
    array_dataspace = H5Screate_simple(rank, dimsf, NULL);
    scalar_dataspace = H5Screate_simple(1, scalar_dimsf, NULL);

    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    status = H5Tset_order(datatype, H5T_ORDER_LE);

    G_dataset = H5Dcreate(file, "Abins_G", H5T_NATIVE_FLOAT, array_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    T_dataset = H5Dcreate(file, "Abins_T", H5T_NATIVE_FLOAT, array_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    float N_[1];
    N_[0] = N;
    N_dataset = H5Dcreate(file, "N", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float n_[1];
    n_[0] = n;
    n_dataset = H5Dcreate(file, "n", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float g_[1];
    g_[0] = g;
    g_dataset = H5Dcreate(file, "g", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float t_[1];
    t_[0] = t;
    t_dataset = H5Dcreate(file, "t", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float mu_a_[1];
    mu_a_[0] = mu_a;
    mu_a_dataset = H5Dcreate(file, "mu_a", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float mu_s_[1];
    mu_s_[0] = mu_s;
    mu_s_dataset = H5Dcreate(file, "mu_s", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float l0_[1];
    l0_[0] = l0;
    l0_dataset = H5Dcreate(file, "l0", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float z0_[1];
    z0_[0] = z0;
    z0_dataset = H5Dcreate(file, "z0", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float wr_[1];
    wr_[0] = wr;
    wr_dataset = H5Dcreate(file, "wr", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float w0_[1];
    w0_[0] = w0;
    w0_dataset = H5Dcreate(file, "w0", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float rmax_[1];
    rmax_[0] = rmax;
    rmax_dataset = H5Dcreate(file, "rmax", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float dr_[1];
    dr_[0] = dr;
    dr_dataset = H5Dcreate(file, "dr", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float zmax_[1];
    zmax_[0] = zmax;
    zmax_dataset = H5Dcreate(file, "zmax", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    float dz_[1];
    dz_[0] = dz;
    dz_dataset = H5Dcreate(file, "dz", H5T_NATIVE_FLOAT, scalar_dataspace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(G_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, G_data2D);
    status = H5Dwrite(T_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, T_data2D);
    status = H5Dwrite(N_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, N_);
    status = H5Dwrite(n_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, n_);
    status = H5Dwrite(g_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, g_);
    status = H5Dwrite(t_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, t_);
    status = H5Dwrite(mu_a_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, mu_a_);
    status = H5Dwrite(mu_s_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, mu_s_);
    status = H5Dwrite(l0_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, l0_);
    status = H5Dwrite(z0_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, z0_);
    status = H5Dwrite(wr_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, wr_);
    status = H5Dwrite(w0_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, w0_);
    status = H5Dwrite(rmax_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, rmax_);
    status = H5Dwrite(dr_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, dr_);
    status = H5Dwrite(zmax_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, zmax_);
    status = H5Dwrite(dz_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
		      H5P_DEFAULT, dz_);

    H5Sclose(array_dataspace);
    H5Sclose(scalar_dataspace);
    H5Tclose(datatype);
    H5Dclose(T_dataset);
    H5Dclose(G_dataset);
    H5Dclose(N_dataset);
    H5Dclose(n_dataset);
    H5Dclose(g_dataset);
    H5Dclose(t_dataset);
    H5Dclose(mu_a_dataset);
    H5Dclose(mu_s_dataset);
    H5Dclose(l0_dataset);
    H5Dclose(z0_dataset);
    H5Dclose(wr_dataset);
    H5Dclose(w0_dataset);
    H5Dclose(rmax_dataset);
    H5Dclose(dr_dataset);
    H5Dclose(zmax_dataset);
    H5Dclose(dz_dataset);
    H5Fclose(file);

}

int main(int argc, char* argv[])
{

    if (cmdOptionExists(argv, argv+argc, "--help") || cmdOptionExists(argv, argv+argc, "-h"))
    {
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nGaussian Focused Monte Carlo\n";
        cout << "\nWritten by Patrick Cook | Fort Hays State University | 24 August 2019\n";
        cout << "pdcook@mail.fhsu.edu or qphysicscook@gmail.com\n";
        cout << "\nBased on the method in Clark and Cook (2019)\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nUseful Flags:\n";
        cout << "\n--help -    Shows this help text.\n";
        cout << "\n-h     -    Same as --help.\n";
        cout << "\n--aggregate\n       -    Sum over each bin and return aggregate values for\n            each measurement instead of returning the bins.\n";
        cout << "\n--example\n       -    Show an example configuration and run it.\n            Makes all required parameters except --GPU optional.\n            Useful for ensuring your installation works.\n";
        cout << "\n--go   -    Just run the simulation. Don't ask to confirm.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nRequired Parameters:\n";
        cout << "\n--GPU  -    Device ID for GPU to run on.\n            Use the 'nvidia-smi' command to list available GPUs.\n            Mutliple GPUs are NOT supported.\n            Example: --GPU 0\n";
        cout << "\n--prefix\n       -    Prefix for output filenames.\n";
        cout << "\n-N     -    Number of photons to use per simulation.\n";
        cout << "\n--layers\n       -    Number of layers in the sample.\n            >>> GFMLMC CURRENTLY ONLY SUPPORTS SINGLE LAYER SAMPLES <<<\n";
        cout << "\n--focustype\n       -    Focus type, 0 for traditional, 1 for gaussian, or 2 for both.\n            If only one focustype is done, and aggregate is false, the data from the\n             specified focus type will be copied to all fields.\n";
        cout << "\n--wr   -    Incident beam radius in centimeters.\n";
        cout << "\n--z0   -    Beam waist position in centimeters.\n";
        cout << "\n--l0   -    Beam wavelength in nanometers.\n";
        cout << "\n-n     -    Relative refractive index (relative to the surrounding medium) for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -n \"1.33 1.4 1.33\"\n";
        cout << "\n-g     -    Anisotropy for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -g \"0.0 -0.2 1\"\n";
        cout << "\n-t     -    Thickness for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -t \"0.1 0.1 0.2\"\n";
        cout << "\n--mua  -    Absorption coefficient for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -mua \"1.1 0.5 1\"\n";
        cout << "\n--mus  -    REDUCED scattering coefficient for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -mus \"30 10 1\"\n";
        cout << "            To convert from the reduced scattering coefficient, mus', to the scattering coefficient:\n";
        cout << "            mus = mus'/(1-g) for g!=1 and mus = mus' if g = 1.\n";
        cout << "\n--rmax -    Maximum radial coordinate in centimeters to be binned.\n";
        cout << "\n--dr   -    Resolution of bins in the r direction in centimeters\n";
        cout << "\n--zmax -    Maximum axial coordinate in centimeters to be binned.\n";
        cout << "\n--dz   -    Resolution of bins in the z direction in centimeters\n";
        cout << "\n\nBins:\n";
        cout << "\n    Bins    |       Size       | Minimum | Maximum | Resolution | Units";
        cout << "\n-----------------------------------------------------------------------";
        cout << "\n   rbins    |  ceil(rmax/dr)   |    0    |  rmax   |     dr     |  cm  ";
        cout << "\n   zbins    |  ceil(zmax/dz)   |    0    |  zmax   |     dz     |  cm  ";
        cout << "\n\nNotes:\nBinning is done left-hand-inclusive, so a bin that\nis from 0cm to 1cm will include photons at 0cm but exclude those at 1cm.";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nNotes:\n";
        cout << "If you are getting 'out of memory' errors, reduce N or change the RNGS variable in the source\ncode to something smaller.\n\n";

        return 1;
    }

    if (!cmdOptionExists(argv, argv+argc, "--GPU"))
    {
        cout << "Must specify device ID with the --GPU flag.\nUse --help to see available options.\n";
        return 2;
    }

    srand(time(NULL));

    int GPU = readIntOption(argv, argv+argc, "--GPU");
    int nGPU = 1;   // only single GPU supported at this time

    bool go = cmdOptionExists(argv, argv+argc, "--go");

    if (!go)
    {
        cout << "\n";
        CUDABasicProperties(GPU);
    }

    //// Declare global variables for CURAND
    rand_set = new bool[nGPU]();
    globalDeviceStates = new curandState*[nGPU];
    ////

    // initial parameters that will persist if the user doesn't specify them

    int N = -1;
    int layers = -1;
    int focustype = -1;

    string prefix = "";

    float rmax = nanf("0");
    float zmax = nanf("0");
    float dr = nanf("0");
    float dz = nanf("0");

    float wr = nanf("0");
    float z0 = nanf("0");
    float l0 = nanf("0");

    float* n;
    float* g;
    float* t;

    float* mu_a;
    float* mu_s_;

    bool aggregate_bool = false;

    // use example parameters
    if (cmdOptionExists(argv, argv+argc, "--example"))
    {

        N = 1000000;
        layers = 1;

        prefix = "EXAMPLE";
        focustype = 2;

        aggregate_bool = true;

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        rmax = 16e-4;
        zmax = 18.8e-3;
        dr = rmax/204;
        dz = zmax/202;

        wr = 8e-4;
        z0 = 9.4e-3;
        l0 = 1064;

        n[0] = 1.;
        g[0] = 0.9;
        t[0] = 18.8e-3;

        mu_a = new float[layers];
        mu_s_ = new float[layers];

        mu_a[0] = 10;
        mu_s_[0] = 0;

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            cout << "\nEXAMPLE GFMLMC SIMULATION\n";
            printf("\nParameters (As found in Clark and Cook 2019):\n\n--prefix %s\n-N %d\n--layers %d\n--focustype %d\n-n", prefix.c_str(), N, layers, focustype);
            for (int i = 0; i < layers; i++)
            {
                cout << " " << n[i];
            }
            cout << "\n-g";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << g[i];
            }
            cout <<"\n-t";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << t[i];
            }
            cout << "\n--mua";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_a[i];
            }
            cout << "\n--mus";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_s_[i];
            }
            printf("\n--wr %f\n--z0 %f\n--l0 %f\n--rmax %f\n--dr %f\n--zmax %f\n--dz %f", wr, z0, l0, rmax, dr, zmax, dz);
            if (aggregate_bool){ cout << "\n--aggregate\n"; }
            cout << "\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // user specified parameters parsed from command line
    else
    {

        N = readIntOption(argv, argv+argc, "-N");

        layers = readIntOption(argv, argv+argc, "--layers");
        focustype = readIntOption(argv, argv+argc, "--focustype");

        prefix = readStrOption(argv, argv+argc, "--prefix");

        aggregate_bool = cmdOptionExists(argv, argc+argv, "--aggregate");

        rmax = readFloatOption(argv, argc+argv, "--rmax");
        zmax = readFloatOption(argv, argc+argv, "--zmax");
        dr = readFloatOption(argv, argc+argv, "--dr");
        dz = readFloatOption(argv, argc+argv, "--dz");

        wr = readFloatOption(argv, argc+argv, "--wr");
        z0 = readFloatOption(argv, argc+argv, "--z0");
        l0 = readFloatOption(argv, argc+argv, "--l0");

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        n = readArrOption(argv, argv+argc, "-n", layers);
        g = readArrOption(argv, argv+argc, "-g", layers);
        t = readArrOption(argv, argv+argc, "-t", layers);

        mu_a = new float[layers];
        mu_s_ = new float[layers];

        mu_a = readArrOption(argv, argv+argc, "--mua", layers);
        mu_s_ = readArrOption(argv, argv+argc, "--mus", layers);

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            printf("\nParameters:\n");
            printf("\n--prefix %s", prefix.c_str());
            printf("\n-N %d", N);
            printf("\n--layers %d\n--focustype %d\n-n", layers, focustype);
            for (int i = 0; i < layers; i++)
            {
                cout << " " << n[i];
            }
            cout << "\n-g";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << g[i];
            }
            cout <<"\n-t";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << t[i];
            }
            cout << "\n--mua";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_a[i];
            }
            cout << "\n--mus";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_s_[i];
            }
            printf("\n--wr %f\n--z0 %f\n--l0 %f\n--rmax %f\n--dr %f\n--zmax %f\n--dz %f", wr, z0, l0, rmax, dr, zmax, dz);
            if (aggregate_bool)
            {
                cout << "\n--aggregate";
            }
            cout <<"\n\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // calculate mus and mut from mua and mus'
    float* mu_s = new float[layers];
    float* mu_t = new float[layers];
    for (int i = 0; i < layers; i++)
    {
        if (g[i] == 1) { mu_s[i] = mu_s_[i]; }
        else { mu_s[i] = mu_s_[i]/(1-g[i]); }
        mu_t[i] = mu_a[i] + mu_s[i];

    }

    // define layer boundaries
    float* bounds = new float[layers+1];
    bounds[0] = 0;

    for( int i = 1; i < layers+1; i++ )
    {
        bounds[i] = bounds[i-1] + t[i-1];
    }

    // calc bin array sizes

    int N_r = ceil(rmax/dr);
    int N_z = ceil(zmax/dz);

    // declare result variables
    // arrays are <r,z>
    float* A_G = new float[N_r*N_z];
    float* A_T = new float[N_r*N_z];
    float E_G;
    float R_d_G;
    float R_s_G;
    float T_d_G;
    float T_u_G;
    float E_T;
    float R_d_T;
    float R_s_T;
    float T_d_T;
    float T_u_T;
    float* A_T_unc = new float[N_r*N_z];
    float* A_G_unc = new float[N_r*N_z];
    float E_G_unc;
    float R_d_G_unc;
    float R_s_G_unc;
    float T_d_G_unc;
    float T_u_G_unc;
    float E_T_unc;
    float R_d_T_unc;
    float R_s_T_unc;
    float T_d_T_unc;
    float T_u_T_unc;

    // run the simulation once for each focus type
    if ( focustype == 0 )
    {
        thrust::tie(A_T,E_T,R_d_T,R_s_T,T_d_T,T_u_T) = GFMLMC(GPU, 0, wr, z0, l0, rmax, dr, zmax, dz, N, layers, n, g, t, bounds, mu_a, mu_s, mu_t);
        thrust::tie(A_G,E_G,R_d_G,R_s_G,T_d_G,T_u_G) = thrust::make_tuple(A_T,E_T,R_d_T,R_s_T,T_d_T,T_u_T);
    }
    else if (focustype == 1)
    {
        thrust::tie(A_G,E_G,R_d_G,R_s_G,T_d_G,T_u_G) = GFMLMC(GPU, 1, wr, z0, l0, rmax, dr, zmax, dz, N, layers, n, g, t, bounds, mu_a, mu_s, mu_t);
        thrust::tie(A_T,E_T,R_d_T,R_s_T,T_d_T,T_u_T) = thrust::make_tuple(A_G,E_G,R_d_G,R_s_G,T_d_G,T_u_G);
    }
    else if (focustype == 2)
    {
        thrust::tie(A_T,E_T,R_d_T,R_s_T,T_d_T,T_u_T) = GFMLMC(GPU, 0, wr, z0, l0, rmax, dr, zmax, dz, N, layers, n, g, t, bounds, mu_a, mu_s, mu_t);
        thrust::tie(A_G,E_G,R_d_G,R_s_G,T_d_G,T_u_G) = GFMLMC(GPU, 1, wr, z0, l0, rmax, dr, zmax, dz, N, layers, n, g, t, bounds, mu_a, mu_s, mu_t);
    }


    // find uncertainties given by the poisson counting distribution
    E_G_unc = sqrt(E_G/N);
    R_d_G_unc = sqrt(R_d_G/N);
    R_s_G_unc = sqrt(R_s_G/N);
    T_d_G_unc = sqrt(T_d_G/N);
    T_u_G_unc = sqrt(T_u_G/N);
    E_T_unc = sqrt(E_T/N);
    R_d_T_unc = sqrt(R_d_T/N);
    R_s_T_unc = sqrt(R_s_T/N);
    T_d_T_unc = sqrt(T_d_T/N);
    T_u_T_unc = sqrt(T_u_T/N);

    A_G_unc = array_uncertainty(A_G, N_r, N_z, N);
    A_T_unc = array_uncertainty(A_T, N_r, N_z, N);

    if (!aggregate_bool)
    {
        A_G = normalize(A_G, N_r, N_z, dr, dz);
        A_T = normalize(A_T, N_r, N_z, dr, dz);
        A_G_unc = normalize(A_G_unc, N_r, N_z, dr, dz);
        A_T_unc = normalize(A_T_unc, N_r, N_z, dr, dz);
        if (std::experimental::filesystem::exists(prefix+"_GFMLMC.h5"))
        {

            cout <<"\n\nWARN: HDF5 output file already exists. \n Press [enter] to overwrite or Ctrl+C to cancel.";
            getchar();
            string s_filename = prefix+"_GFMLMC.h5";
            char* filename = new char[s_filename.length()+1];
            strcpy(filename, s_filename.c_str());
            remove(filename);
        }

        write_HDF5(prefix+"_GFMLMC.h5", A_G, A_T, N_r, N_z, N, n[0], g[0], t[0], mu_a[0], mu_s[0], l0, z0, wr, rmax, dr, zmax, dz);
    }

    if (aggregate_bool)
    {
        float A_G_tot;
        float A_T_tot;
        float A_G_tot_unc;
        float A_T_tot_unc;

        A_G_tot = aggregate(A_G, N_r, N_z);
        A_T_tot = aggregate(A_T, N_r, N_z);

        A_G_tot_unc = sqrt(A_G_tot/N);
        A_T_tot_unc = sqrt(A_T_tot/N);

        if (focustype == 0 || focustype == 2)
        {
            printf("\nTRADITIONAL:\nA: %f +/- %f\nR_d: %f +/- %f\nR_s: %f +/- %f\nT_d: %f +/- %f\nT_u: %f +/- %f\nE: %f +/- %f\n", A_T_tot, A_T_tot_unc, R_d_T, R_d_T_unc, R_s_T, R_s_T_unc, T_d_T, T_d_T_unc, T_u_T, T_u_T_unc, E_T, E_T_unc);
        }
        if (focustype == 1 || focustype == 2)
        {
            printf("\nGAUSSIAN:\nA: %f +/- %f\nR_d: %f +/- %f\nR_s: %f +/- %f\nT_d: %f +/- %f\nT_u: %f +/- %f\nE: %f +/- %f\n", A_G_tot, A_G_tot_unc, R_d_G, R_d_G_unc, R_s_G, R_s_G_unc, T_d_G, T_d_G_unc, T_u_G, T_u_G_unc, E_G, E_G_unc);
        }

    }

    return 0;


}
