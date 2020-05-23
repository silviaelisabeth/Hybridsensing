__author__ = 'szieger'
__project__ = 'dualsensor T/O2 sensing'

import matplotlib
import matplotlib.pyplot as plt
import additional_functions as af
import numpy as np
import matplotlib.dates as mdates
import datetime
from lmfit import Model
from scipy.signal import savgol_filter
from scipy import interpolate
import pandas as pd
from time import gmtime, strftime
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, LinearLocator, FormatStrFormatter, MaxNLocator
from matplotlib import cm
from matplotlib import interactive
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

sns.set_context('paper', 1.5)
sns.set_style('ticks')
sns.set_palette('Set1')

# gobal variables
switcher = {'s': 0, 'ms': -3, 'µs': -6, 'ns': -9}
colors_freq = ['slategrey', 'forestgreen', 'darkorange', 'navy', 'crimson']

# --------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------
# fitting functions
def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def func_linear(x, m, t):
    return m*x + t


def function_TSM(x, m, k, f):
    """
    fitting function according to the common two site model. In general, x represents the pO2 or pCO2 content, whereas
    m, k and f are the common fitting parameters
    :param x:   list
    :param m:   np.float
    :param k:   np.float
    :param f:   np.float
    :return:
    """
    return f / (1. + k*x) + (1.-f)/(1. + k*m*x)


def twoSiteModel_calib_ksv(tau0, tau1, m, f, pO2_calib1):
    # preparation
    if isinstance(tau0, np.float):
        if isinstance(tau1, np.float):
            tau_quot = tau1 / tau0
        else:
            tau_quot = [x1/tau0 for x1 in tau1]
    else:
        if isinstance(tau1, np.float):
            tau_quot = [tau1 / x2 for x2 in tau0]
        else:
            tau_quot = [x1/x2 for (x1, x2) in zip(tau1, tau0)]

    if isinstance(tau_quot, np.float):
        # parts of pq equation
        a = (pO2_calib1 ** 2) * (m * tau_quot)
        b = pO2_calib1 * ((m + 1) * tau_quot - (m * f - f + 1))
        c = tau_quot - 1

        sqrt1 = b ** 2
        sqrt2 = 4 * a * c

        z_ = np.sqrt(sqrt1 - sqrt2)
        z1 = -1 * b + z_
        z2 = -1 * b - z_
        n = 2 * a

        ksv_1 = z1 / n
        ksv_2 = z2 / n

        if ksv_1 < 0 and ksv_2 >=0:
            Ksv_fit1 = ksv_2
        elif ksv_1 >= 0 and ksv_2 < 0:
            Ksv_fit1 = ksv_1
        else:
            print(ksv_1, ksv_2)
            raise ValueError('decide about Ksv')

        # combining all (fit) parameter for two-site-model
        para_TSM = pd.Series({'tauP0': tau0, 'tauP1': tau1, 'prop Ksv': m, 'slope': f, 'Ksv_fit1': Ksv_fit1,
                              'Ksv_fit2': Ksv_fit1 * m})

    else:
        # parts of pq equation
        a = [(pO2_calib1**2)*(m*t) for t in tau_quot]
        b = [pO2_calib1*((m + 1)*t - (m*f - f + 1)) for t in tau_quot]
        c = [t - 1 for t in tau_quot]

        sqrt1 = [b_**2 for b_ in b]
        sqrt2 = [4*a_*c_ for (a_, c_) in zip(a, c)]

        z_ = [np.sqrt(s1 - s2) for (s1, s2) in zip(sqrt1, sqrt2)]
        z1 = [-1*b_ + z1_ for (b_, z1_) in zip(b, z_)]
        z2 = [-1*b_ - z1_ for (b_, z1_) in zip(b, z_)]
        n = [2*a_ for a_ in a]

        ksv_1 = [x/y for (x, y) in zip(z1, n)]
        ksv_2 = [x/y for (x, y) in zip(z2, n)]

        if ksv_1[1] < 0 and ksv_2[1] >=0:
            Ksv_fit1 = ksv_2
        elif ksv_1[1] >= 0 and ksv_2[1] < 0:
            Ksv_fit1 = ksv_1
        else:
            print(ksv_1, ksv_2)
            raise ValueError('decide about Ksv')
            # Ksv_fit1 = ksv_1.append(ksv_2)

        # combining all (fit) parameter for two-site-model
        para_TSM = pd.Series({'tauP0': tau0, 'tauP1': tau1, 'prop Ksv': m, 'slope': f, 'Ksv_fit1': Ksv_fit1,
                              'Ksv_fit2': [k*m for k in Ksv_fit1]})

    return para_TSM


def daterange(start_date, end_date, delta_seconds):
    delta = datetime.timedelta(seconds=delta_seconds)
    while start_date < end_date:
        yield start_date
        start_date += delta


def twoSiteModel_evaluation_(tau0, tau, m, f, ksv, pO2_range):
    # preparation pq equation
    if isinstance(tau0, np.float):
        if isinstance(tau, np.float):
            quot = tau / tau0
        else:
            quot = [t1 / tau0 for t1 in tau]
    else:
        if isinstance(tau, np.float):
            quot = [tau / t0 for t0 in tau0]
        else:
            quot = [t1 / t0 for (t1, t0) in zip(tau, tau0)]

    c = [1 - 1 / q for q in quot]
    if isinstance(ksv, np.float):
        b = [ksv * (m * (1 - f / t) + 1 / t * (f - 1) + 1) for t in quot]
        a = m * (ksv ** 2)
    else:
        b = [k * (m * (1 - f / t) + 1 / t * (f - 1) + 1) for (k, t) in zip(ksv, quot)]
        a = [m * (p ** 2) for p in ksv]

    if isinstance(a, np.float):
        sqrt = [np.sqrt(b_ ** 2 - 4 * a * c_) for (b_, c_) in zip(b, c)]
        n = 2 * a
    else:
        sqrt = [np.sqrt(b_ ** 2 - 4 * a_ * c_) for (b_, a_, c_) in zip(b, a, c)]
        n = [2 * a_ for a_ in a]
    z1 = [-1 * b_ + s for (b_, s) in zip(b, sqrt)]
    z2 = [-1 * b_ - s for (b_, s) in zip(b, sqrt)]

    if isinstance(n, np.float):
        pO2_1 = sorted([z / n for z in z1])
        pO2_2 = sorted([z / n for z in z2])
    else:
        pO2_1 = sorted([z / n for (z, n) in zip(z1, n)])
        pO2_2 = sorted([z / n for (z, n) in zip(z2, n)])

    return pO2_1, pO2_2


def plotting_calibration_input(df, usedata=None, plotting=True):
    if usedata is None:
        usedata = ['tauP', 'tau0 / tauP', 'I-ratio', 'I0 / I-ratio', 'I-ratio / I0']
    calib_data_series = pd.Series([None, None, None, None, None],
                                  index=['tauP', 'tau0 / tauP', 'tauP / tau0', 'I-ratio', 'I0 / I-ratio'])

    if 'tauP' in usedata or 'tauP [ms]' in usedata or 'tau' in usedata:
        tau_calib = df['tauP']
        # pd.concat([calib_data['tauP'][:1], calib_data['tauP'][:20]])
        # pd.concat([tau_calib.T[:10], tau_calib.T[18:]]).T
        calib_data_series['tauP'] = tau_calib
        tau_quot_invers = None
    if 'tau0 / tauP' in usedata or 'tau0/tauP' in usedata or 'tau quot' in usedata or 'tau_quot' in usedata:
        tau_quot = df['tau0 / tauP']
        # pd.concat([calib_data['tau0 / tauP'][:1], calib_data['tau0 / tauP'][:20]])
        # pd.concat([tau_quot.T[:10], tau_quot.T[18:]]).T
        tau_quot_invers = 1 / tau_quot
        calib_data_series['tau0 / tauP'] = tau_quot
        calib_data_series['tauP / tau0'] = tau_quot_invers

    if 'I-ratio' in usedata or 'i-ratio' in usedata or 'Iratio' in usedata or 'I_ratio' in usedata or 'iratio' in usedata:
        i_ratio = df['I-ratio']
        # pd.concat([calib_data['I-ratio'][:1], calib_data['I-ratio'][:20]])
        i_ratio_invers = 1 / i_ratio
        calib_data_series['I-ratio'] = i_ratio
        #calib_data_series['1/ I-ratio'] = i_ratio_invers
        i_i0_iratio = None

    if 'I0 / I-ratio' in usedata or 'i0/i-ratio' in usedata or 'I0/Iratio' in usedata or 'I0 / I-ratio' in usedata or 'i0 /iratio' in usedata:
        i_i0_iratio = df['I0 / I-ratio']
        # pd.concat([calib_data['I-ratio'][:1], calib_data['I-ratio'][:20]])
        # pd.concat([i_ratio.T[:10], i_ratio.T[18:]]).T
        i_iratio_i0 = 1 / i_i0_iratio
        calib_data_series['I0 / I-ratio'] = i_i0_iratio
        calib_data_series['I-ratio / I0'] = i_iratio_i0

    # ==============================================================================================
    if plotting is True:
        fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(5, 3))
        # mgr = plt.get_current_fig_manager()
        # mgr.window.move(1300, 100)

        ax[0][0].plot(tau_quot, lw=0., marker='s')
        ax[0][1].plot(tau_calib, lw=0., marker='s')
        ax[1][0].plot(i_iratio_i0, lw=0., marker='s')
        ax[1][1].plot(i_ratio, lw=0., marker='s')

        # legend
        legend_str = ['T = {:.2f}°C'.format(i) for i in tau_quot.columns]
        ax[0][0].legend(legend_str, loc='upper center', bbox_to_anchor=(1., 1.3), ncol=3, fontsize=10, frameon=False,
                        fancybox=True, borderaxespad=.5, labelspacing=5)
        ax[0][0].tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
        ax[0][1].tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
        ax[1][0].tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
        ax[1][1].tick_params(which='both', direction='in', top=True, right=True, labelsize=10)

        # layout
        ax[0][0].set_ylabel('tau0 / tauP', fontsize=11)
        ax[0][1].set_ylabel('tauP [ms]', fontsize=11)
        ax[1][0].set_xlabel('pO2 [hPa]', fontsize=11)
        ax[1][1].set_xlabel('pO2 [hPa]', fontsize=11)
        ax[1][0].set_ylabel('I-ratio / I0', fontsize=11)
        ax[1][1].set_ylabel('I-ratio', fontsize=11)

        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.9, wspace=0.4, hspace=0.1)

    return calib_data_series


def plotting_fit_regression(res_fit, regression, type_='tauP', figposition=None):
    f, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(5, 3))
    # mgr = plt.get_current_fig_manager()
    if figposition is None:
        figposition = [1300, 100]
    # mgr.window.move(figposition[0], figposition[1])

    ax[0][0].plot(res_fit.columns, res_fit.T['m'], marker='s', lw=0., label='m', color='navy')
    ax[1][0].plot(res_fit.columns, res_fit.T['k'], marker='s', lw=0., label='k', color='forestgreen')
    ax[0][1].plot(res_fit.columns, res_fit.T['f'], marker='s', lw=0., label='f', color='dodgerblue')
    ax[1][1].plot(res_fit.columns, res_fit.T['chi square'], marker='s', lw=0., label='chi square', color='darkorange')

    ax[0][0].plot(regression.columns, regression.T['m'], lw=0.75, ls='--', label='fit', color='k')
    ax[1][0].plot(regression.columns, regression.T['k'], lw=0.75, ls='--', label='fit', color='k')
    ax[0][1].plot(regression.columns, regression.T['f'], lw=0.75, ls='--', label='fit', color='k')
    ax[1][1].plot(regression.columns, regression.T['chi square'], lw=0.75, ls='--', label='fit', color='k')

    min_m, max_m = limit_range_plotting(m_min=res_fit.T['m'].min(), m_max=res_fit.T['m'].max(), percent=10.)
    min_k, max_k = limit_range_plotting(m_min=res_fit.T['k'].min(), m_max=res_fit.T['k'].max(), percent=10.)
    min_f, max_f = limit_range_plotting(m_min=res_fit.T['f'].min(), m_max=res_fit.T['f'].max(), percent=10.)
    min_chi, max_chi = limit_range_plotting(m_min=res_fit.T['chi square'].min(), m_max=res_fit.T['chi square'].max(),
                                            percent=10.)

    ax[0][0].tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    ax[1][0].tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    ax[0][1].tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    ax[1][1].tick_params(which='both', direction='in', top=True, right=True, labelsize=9)

    ax[0][0].set_ylim(min_m, max_m)
    ax[0][1].set_ylim(min_f, max_f)
    ax[1][0].set_ylim(min_k, max_k)
    ax[1][1].set_ylim(min_chi, max_chi)

    ax[0][0].set_ylabel('m', fontsize=11)
    ax[1][0].set_ylabel('k', fontsize=11)
    ax[0][1].set_ylabel('f', fontsize=11)
    ax[1][1].set_ylabel('χ$^2$', fontsize=11)
    ax[1][0].set_xlabel('Temperature [°C]', fontsize=11)
    ax[1][1].set_xlabel('Temperature [°C]', fontsize=11)
    # f.delaxes(ax[1][1])

    time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    f.suptitle(time_now.split()[0] + ' Parameterfit of {} for two-site-model'.format(type_), fontsize=10, y=.98)
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=.5, hspace=.25)

    return f


def plotting_intersection_planes(tauP_meas, tauP_calib, distance_tau, iratio_calib, i_ratio_meas, distance_int, run,
                                 zlim_tau=None, zlim_int=None):

    temp_X, pO2_Y = np.meshgrid(tauP_calib.columns, tauP_calib.index)

    fig = plt.figure(figsize=(5, 4.75))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax1 = fig.add_subplot(2, 1, 2, projection='3d')
    # mgr = plt.get_current_fig_manager()
    # mgr.window.move(970, 10)

    # lifetime tauP [ms]
    ax.plot_surface(X=temp_X, Y=pO2_Y, Z=tauP_meas*1e3, cstride=2, rstride=1, color='k', linewidth=0, antialiased=False,
                    alpha=0.25)
    if run == 0.:
        ax.plot_surface(X=temp_X, Y=pO2_Y, Z=tauP_calib*1e3, cstride=2, rstride=1, color='#38678f', linewidth=0,
                        antialiased=True, alpha=0.8)
        if distance_tau.empty:
            print('')
            print('[Error] Cannot determine plane intersection according to chosen error. Error for tauP')
        else:
            ax.plot(xs=distance_tau.index, ys=distance_tau['pO2'].values, zs=distance_tau['zvalue'].values*1e3,
                    color='crimson', lw=1.)
    # intensity ratio
    ax1.plot_surface(X=temp_X, Y=pO2_Y, Z=iratio_calib, cstride=2, rstride=1, color='forestgreen', linewidth=0,
                     antialiased=False, alpha=0.5)
    ax1.plot_surface(X=temp_X, Y=pO2_Y, Z=i_ratio_meas, color='k', linewidth=0, antialiased=True, alpha=0.8)
    if distance_int.empty:
        print('')
        print('[Error] Cannot determine plane intersection according to chosen error. Error for I-ratio')
    else:
        ax1.plot(xs=distance_int.index, ys=distance_int['pO2'].values, zs=distance_int['zvalue'].values, color='orange',
                 lw=1.)

    if zlim_tau is not None:
        ax.set_zlim(zlim_tau[0], zlim_tau[1])
    if zlim_int is not None:
        ax1.set_zlim(zlim_int[0], zlim_int[1])

    ax.tick_params(which='both', labelsize=10, pad=2)
    ax.set_xlabel('Temperature [°C]', fontsize=11, labelpad=8)
    ax.set_ylabel('pO2 [hPa]', fontsize=11, labelpad=8)
    ax.set_zlabel('tauP [ms]', fontsize=11, labelpad=2)

    ax1.tick_params(which='both', labelsize=10, pad=2)
    ax1.set_xlabel('Temperature [°C]', fontsize=11, labelpad=8)
    ax1.set_ylabel('pO2 [hPa]', fontsize=11, labelpad=8)
    ax1.set_zlabel('I-ratio', fontsize=11, labelpad=2)

    plt.subplots_adjust(top=0.98, bottom=0.1, left=0.02, right=0.9, hspace=0.2)  # tight_layout()
    plt.show()

    return fig


def plotting_intersection_lines(results_calc, distance_int, distance_tau, x_int_temp, x_tau_temp, d_int, d_tau, fig_):
    # Plotting intersection of lines
    if fig_ is None:
        fig1, ax3 = plt.subplots(nrows=2, sharex=True, figsize=(5, 3))
    else:
        fig1 = fig_
    # mgr = plt.get_current_fig_manager()
    # mgr.window.move(970, 600)

    ax3[0].plot(distance_int['pO2'], color='forestgreen', lw=0.75, label='I-ratio')
    ax3[0].plot(distance_tau['pO2'], color='#38678f', lw=0.75, label='tau')
    ax3[0].legend(fontsize=7)
    ax3[0].axhline(y=results_calc['intensity domain']['pO2 [hPa]'], color='crimson', lw=1., ls='--')
    ax3[0].axvline(x=results_calc['intensity domain']['T [°C]'], color='crimson', lw=1., ls='--')

    ax3[1].plot(x_int_temp, d_int, color='forestgreen', lw=0, marker='o')
    ax3[1].plot(x_tau_temp, d_tau, color='#38678f', lw=0, marker='o')

    ax3[1].set_xlabel('Temperature [°C]', fontsize=11, labelpad=2)
    ax3[1].set_ylabel('variance', fontsize=11, labelpad=2)
    ax3[0].set_ylabel('pO2 [hPa]', fontsize=11, labelpad=2)
    ax3[0].set_title('intersection of lines', loc='left', fontsize=11)
    ax3[1].tick_params(which='both', direction='in', labelsize=8)
    ax3[0].tick_params(which='both', direction='in', labelsize=8)

    plt.tight_layout(h_pad=0)
    plt.show()

    return fig1


def plotting_multisensing(parameter, xs_all, xs_temp, ys_all, ys, df_dphi_all, zs_temp, start_val, set_T_constant,
                          ylabel, freq_pair, freq_Hz):
    # Lifetime dual sensor
    if 'tauP' in parameter:
        fig1 = plt.figure(figsize=(8, 5))

        ax = fig1.add_subplot(121, projection='3d')
        ax1 = fig1.add_subplot(322)
        ax1_ref = fig1.add_subplot(324)
        ax2 = fig1.add_subplot(326)

        # -----------------------------------------------------------------------------------
        # 3D - tauP vs pO2 (or Time) and temperature
        ax.plot(xs_all, ys, df_dphi_all['tau.dual [s]'] * 1e3, marker='o', lw=0., c='#71ae9d', markersize=2)
        ax.view_init(elev=15., azim=19)
        ax.invert_xaxis()

        # 2D - tauP vs pO2 / Time or temperature common with reference
        temp = df_dphi_all.loc[df_dphi_all.index[start_val]:, 'Temp. Probe']
        ax1.scatter(ys[start_val:], df_dphi_all.loc[df_dphi_all.index[start_val]:, 'tau.dual [s]'] * 1e3, marker='.',
                    s=5, color='navy')
        ax1_ref.scatter(ys[start_val:], df_dphi_all.loc[df_dphi_all.index[start_val]:, 'tau.ref [s]'] * 1e6, marker='.',
                        s=0.5, color='k')
        ax1.legend(['T {:.2f} +/- {:.2f}°C'.format(temp.mean(), temp.std())], fontsize=9)

        if set_T_constant is True:
            po2 = df_dphi_all.loc[xs_temp.index, 'pO2.mean']
            y = xs_temp
            z = zs_temp['tau.dual [s]'] * 1e3
        else:
            po2 = df_dphi_all['pO2.mean']
            y = df_dphi_all['Temp. Probe']
            z = df_dphi_all['tau.dual [s]'] * 1e3
        ax2.scatter(y, z, marker='.', s=5, color='navy')
        ax2.legend(['pO2 {:.2f} +/- {:.2f}hPa'.format(po2.mean(), po2.std())], fontsize=9)
        # -----------------------------------------------------------------------------------

        ax.tick_params(labelsize=10., direction='in', right=True, top=True)
        ax1.tick_params(labelsize=10., direction='in', right=True, top=True)
        ax1_ref.tick_params(labelsize=10., direction='in', right=True, top=True)
        ax2.tick_params(labelsize=10., direction='in', right=True, top=True)

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1_ref.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax1_ref.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))

        # -----------------------------------------------------------------------------------
        ax.set_xlabel('Temperature [°C]', fontsize=9.)
        ax.set_ylabel(ylabel, fontsize=9.)
        ax.set_zlabel('tau [ms]', fontsize=9.)

        ax1.set_ylabel('tauP [ms]', fontsize=9.)
        ax1.set_xlabel(ylabel, fontsize=9.)
        ax1_ref.set_xlabel(ylabel, fontsize=9.)
        ax1_ref.set_ylabel('tauP ref [µs]', fontsize=9.)

        ax2.set_xlabel('Temperature [°C]', fontsize=9.)
        ax2.set_ylabel('tauP [ms]', fontsize=9.)

        # -----------------------------------------------------------------------------------
        plt.subplots_adjust(wspace=5, hspace=0.1, left=.1, right=.99)
        plt.tight_layout()
        plt.show()

    # ================================================================================================================
    # tau quotient
    if 'tau0/tau' in parameter:
        fig2 = plt.figure(figsize=(7, 4))
        ax3 = fig2.add_subplot(121, projection='3d')
        ax4 = fig2.add_subplot(222)
        ax5 = fig2.add_subplot(224)

        # -----------------------------------------------------------------------------------
        # 3D - tauP vs pO2 (or Time) and temperature
        ax3.plot(xs_all, ys, df_dphi_all['tau0/tau.dual.mean'], marker='o', lw=0., c='forestgreen', markersize=2)
        ax3.view_init(elev=15., azim=19)
        ax3.invert_xaxis()
        ax3.set_zlim(0, 15)

        # 2D - tauP vs pO2 / Time or temperature common with reference
        temp = df_dphi_all.loc[df_dphi_all.index[start_val]:, 'Temp. Probe']
        ax4.scatter(ys[start_val:], df_dphi_all.loc[df_dphi_all.index[start_val]:, 'tau0/tau.dual.mean'], marker='.',
                    s=5, color='navy')
        ax4.legend(['T {:.2f} +/- {:.2f}°C'.format(temp.mean(), temp.std())], fontsize=9)

        if set_T_constant is True:
            po2 = df_dphi_all.loc[xs_temp.index, 'pO2.mean']
            y = xs_temp
            z = zs_temp['tau0/tau.dual.mean']
        else:
            po2 = df_dphi_all['pO2.mean']
            y = df_dphi_all['Temp. Probe']
            z = df_dphi_all['tau0/tau.dual.mean']
        ax5.scatter(y, z, marker='.', s=5, color='navy')
        ax5.legend(['pO2 {:.2f} +/- {:.2f}hPa'.format(po2.mean(), po2.std())], fontsize=9)
        ax5.set_ylim(0, 15)

        # -----------------------------------------------------------------------------------
        ax3.tick_params(labelsize=10., direction='in')
        ax4.tick_params(labelsize=10., direction='in')
        ax5.tick_params(labelsize=10., direction='in')

        ax3.set_xlabel('Temperature [°C]', fontsize=9.)
        ax3.set_ylabel(ylabel, fontsize=9.)
        ax3.set_zlabel('tau0/tauP ', fontsize=9.)

        ax5.set_xlabel('Temperature [°C]', fontsize=9.)
        ax4.set_ylabel('tau0/tauP', fontsize=9.)
        ax4.set_xlabel(ylabel, fontsize=9.)
        ax5.set_ylabel('tau0/tauP', fontsize=9.)

        plt.tight_layout(pad=1.25)
        plt.show()

    # ================================================================================================================
    # Iratio
    if 'Iratio' in parameter:
        fig3 = plt.figure(figsize=(7, 4))
        ax6 = fig3.add_subplot(121, projection='3d')
        ax7 = fig3.add_subplot(222)
        ax8 = fig3.add_subplot(224)

        # -----------------------------------------------------------------------------------
        # 3D - tauP vs pO2 (or Time) and temperature
        ax6.plot(xs_all, ys, df_dphi_all['I pF/dF'], marker='o', lw=0., c='crimson', markersize=2)
        ax6.view_init(elev=15., azim=19)
        ax6.invert_xaxis()

        # 2D - tauP vs pO2 / Time or temperature common with reference
        temp = df_dphi_all.loc[df_dphi_all.index[start_val]:, 'Temp. Probe']
        ax7.scatter(ys[start_val:], df_dphi_all.loc[df_dphi_all.index[start_val]:, 'I pF/dF'], marker='.', s=5,
                    color='navy')
        ax7.legend(['T {:.2f} +/- {:.2f}°C'.format(temp.mean(), temp.std())], fontsize=9)

        if set_T_constant is True:
            po2 = df_dphi_all.loc[xs_temp.index, 'pO2.mean']
            y = xs_temp
            z = zs_temp['I pF/dF']
        else:
            po2 = df_dphi_all['pO2.mean']
            y = df_dphi_all['Temp. Probe']
            z = df_dphi_all['I pF/dF']
        ax8.scatter(y, z, marker='.', s=5, color='navy')
        ax8.legend(['pO2 {:.2f} +/- {:.2f}hPa'.format(po2.mean(), po2.std())], fontsize=9)

        # -----------------------------------------------------------------------------------
        ax6.tick_params(labelsize=10., direction='in')
        ax7.tick_params(labelsize=10., direction='in')
        ax8.tick_params(labelsize=10., direction='in')

        ax6.set_xlabel('Temperature [°C]', fontsize=9.)
        ax6.set_ylabel(ylabel, fontsize=9.)
        ax6.set_zlabel('I pF/dF', fontsize=9.)

        ax8.set_xlabel('Temperature [°C]', fontsize=9.)
        ax7.set_ylabel('I pF/dF', fontsize=9.)
        ax7.set_xlabel(ylabel, fontsize=9.)
        ax8.set_ylabel('I pF/dF', fontsize=9.)

        plt.tight_layout(pad=1.25)
        plt.show()

    #  ================================================================================================================
    #  1/Iratio
    if '1/Iratio' in parameter:
        fig4 = plt.figure(figsize=(7, 4))
        ax9 = fig4.add_subplot(121, projection='3d')
        ax10 = fig4.add_subplot(222)
        ax11 = fig4.add_subplot(224)

        # -----------------------------------------------------------------------------------
        # 3D - tauP vs pO2 (or Time) and temperature
        ax9.plot(xs_all, ys, 1 / df_dphi_all['I pF/dF'], marker='o', lw=0., c='darkorange', markersize=2)
        ax9.view_init(elev=15., azim=19)
        ax9.invert_xaxis()
        ax9.set_zlim(-0.05, 1.)

        # 2D - tauP vs pO2 / Time or temperature common with reference
        temp = df_dphi_all.loc[df_dphi_all.index[start_val]:, 'Temp. Probe']
        ax10.scatter(ys[start_val:], 1 / df_dphi_all.loc[df_dphi_all.index[start_val]:, 'I pF/dF'], marker='.', s=5,
                     color='navy')
        ax10.legend(['T {:.2f} +/- {:.2f}°C'.format(temp.mean(), temp.std())], fontsize=9)

        if set_T_constant is True:
            po2 = df_dphi_all.loc[xs_temp.index, 'pO2.mean']
            y = xs_temp
            z = 1 / zs_temp['I pF/dF']
        else:
            po2 = df_dphi_all['pO2.mean']
            y = df_dphi_all['Temp. Probe']
            z = 1 / df_dphi_all['I pF/dF']
        ax11.scatter(y, z, marker='.', s=5, color='navy')
        ax11.legend(['pO2 {:.2f} +/- {:.2f}hPa'.format(po2.mean(), po2.std())], fontsize=9)

        # -----------------------------------------------------------------------------------
        ax9.tick_params(labelsize=10., direction='in')
        ax10.tick_params(labelsize=10., direction='in')
        ax11.tick_params(labelsize=10., direction='in')

        ax9.set_xlabel('Temperature [°C]', fontsize=9.)
        ax9.set_ylabel(ylabel, fontsize=9.)
        ax9.set_zlabel('I dF/pF', fontsize=9.)

        ax11.set_xlabel('Temperature [°C]', fontsize=9.)
        ax10.set_ylabel('I dF/pF', fontsize=9.)
        ax10.set_xlabel(ylabel, fontsize=9.)
        ax11.set_ylabel('I dF/pF', fontsize=9.)

        plt.tight_layout(pad=1.25)
        plt.show()

    #  ================================================================================================================
    #  dPhi
    if 'dPhi' in parameter:
        fig5 = plt.figure(figsize=(7, 4))
        ax12 = fig5.add_subplot(121, projection='3d')
        ax13 = fig5.add_subplot(222)
        ax14 = fig5.add_subplot(224)

        # -----------------------------------------------------------------------------------
        # 3D - tauP vs pO2 (or Time) and temperature
        ax12.plot(xs_all, ys, df_dphi_all['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[0]-1])], marker='.', lw=0.,
                  c='slategrey', markersize=2)
        ax12.plot(xs_all, ys_all, df_dphi_all['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[1]-1])], marker='.',
                  lw=0., c='navy', markersize=2)
        ax12.view_init(elev=15., azim=19)
        ax12.invert_xaxis()

        # 2D - tauP vs pO2 / Time or temperature common with reference
        temp = df_dphi_all.loc[df_dphi_all.index[start_val]:, 'Temp. Probe']
        ax13.scatter(ys[start_val:],
                     df_dphi_all.loc[df_dphi_all.index[start_val]:, 'dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[0]-1])], marker='.',
                     s=5, color='slategrey')
        ax13.scatter(ys[start_val:],
                     df_dphi_all.loc[df_dphi_all.index[start_val]:, 'dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[1]-1])], marker='.',
                     s=5, color='navy')
        ax13.legend(['T {:.2f} +/- {:.2f}°C'.format(temp.mean(), temp.std())], fontsize=9)

        if set_T_constant is True:
            po2 = df_dphi_all.loc[xs_temp.index, 'pO2.mean']
            y = xs_temp
            z1 = zs_temp['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[0]-1])]
            z2 = zs_temp['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[1]-1])]
        else:
            po2 = df_dphi_all['pO2.mean']
            y = df_dphi_all['Temp. Probe']
            z1 = df_dphi_all['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[0]-1])]
            z2 = df_dphi_all['dPhi({:.0f}Hz) [deg]'.format(freq_Hz[freq_pair[1]-1])]
        ax14.scatter(y, z1, marker='.', s=5, color='slategrey')
        ax14.scatter(y, z2, marker='.', s=5, color='navy')
        ax14.legend(['pO2 {:.2f} +/- {:.2f}hPa'.format(po2.mean(), po2.std())], fontsize=9)

        # -----------------------------------------------------------------------------------
        ax12.tick_params(labelsize=10., direction='in')
        ax13.tick_params(labelsize=10., direction='in')
        ax14.tick_params(labelsize=10., direction='in')

        ax12.set_xlabel('Temperature [°C]', fontsize=9.)
        ax12.set_ylabel(ylabel, fontsize=9.)
        ax12.set_zlabel('dPhi [deg]', fontsize=9.)

        ax14.set_xlabel('Temperature [°C]', fontsize=9.)
        ax13.set_ylabel('dPhi [deg]', fontsize=9.)
        ax13.set_xlabel(ylabel, fontsize=9.)
        ax14.set_ylabel('dPhi [deg]', fontsize=9.)

        plt.tight_layout(pad=1.25)
        plt.show()


# test data loading
def preparation_input(f_, ddphi_f, time_d):
    # prepare phase shift and frequencies
    f = np.float64(f_.replace(',', '.'))
    if ddphi_f is None:
        ddphi_f = 0.
    else:
        ddphi_f = np.float(ddphi_f.replace(',', '.'))

    time_step = np.float(time_d.replace(',', '.'))

    return f, ddphi_f, time_step


def split_lockin_data_into_dataframes(num_freq, df_lockIn, run):
    df = pd.DataFrame(np.zeros(shape=(round(len(df_lockIn)/num_freq), len(df_lockIn.columns))))
    n = 0
    for i in df_lockIn.index:
        if i%num_freq == run:
            df.loc[n, :] = df_lockIn.loc[i, :].values
            n += 1
    return df


def load_calibration_data(p):
    calib_data = pd.read_csv(p, header=None, index_col=0, encoding='latin-1').T
    
    # temperature points
    for t in calib_data.columns:
        if 'Temperature' in t:
            temp_name = t
    temp_deg_ = calib_data[temp_name].values[0]

    temp = []
    for i, k in enumerate(temp_deg_.split(',')):
        if i == 0:
            temp.append(np.float64(k.split('[')[1]))
        elif i == len(temp_deg_.split(',')) - 1:
            temp.append(np.float(k.split(']')[0]))
        else:
            temp.append(np.float(k))

    # pO2 calibration points
    pO2 = []
    for o in calib_data.columns:
        if 'pO2' in o:
            ox_name = o
    ox_ = calib_data[ox_name].values[0]
    if (',' in ox_) is False:
        for i, k in enumerate(ox_.split(' ')):
            if len(k) > 1:
                pO2.append(np.float(k))
    else:
        for i, k in enumerate(ox_.split(',')):
            if i == 0:
                pO2.append(np.float64(k.split('[')[1]))
            elif i == len(ox_.split(',')) - 1:
                pO2.append(np.float(k.split(']')[0]))
            else:
                pO2.append(np.float(k))

    # I-ratio
    list_Iratio = []
    iratio_ = calib_data['I-ratio'].values[0]
    for i, k in enumerate(iratio_.split(' ')):
        if len(k) > 1:
            if '\r' in k or '\n' in k:
                if '.' in k:
                    if '\r\n' in k:
                        list_Iratio.append(np.float(k.split('\r\n')[0]))
                    elif '\n' in k:
                        list_Iratio.append(np.float(k[:-2]))
                    else:
                        list_Iratio.append(np.float(k))
            else:
                list_Iratio.append(np.float(k))

    if len(list_Iratio) > len(pO2):
        list_Iratio = list_Iratio[len(temp):]

    df_Iratio = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
    for i, k in enumerate(list_Iratio):
        df_Iratio.iloc[i % len(temp), int(i / len(temp))] = k
    df_Iratio.index = temp
    df_Iratio.columns = pO2
    df_Iratio = df_Iratio.T

    # I0 / I-ratio
    list_int0_iratio = []
    iratio_invers_ = calib_data['I0 / I-ratio'].values[0]
    for i, k in enumerate(iratio_invers_.split(' ')):
        if len(k) > 1:
            if '\r' in k or '\n' in k:
                if '.' in k:
                    if '\r' in k:
                        list_int0_iratio.append(np.float(k.split('\r\n')[0]))
                    else:
                        list_int0_iratio.append(np.float(k.split('\n')[0]))
            else:
                list_int0_iratio.append(np.float(k))
    if len(list_int0_iratio) > len(pO2):
        list_int0_iratio = list_int0_iratio[len(temp):]

    df_int0_Iratio = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
    for i, k in enumerate(list_int0_iratio):
        df_int0_Iratio.iloc[i % len(temp), int(i / len(temp))] = k
    df_int0_Iratio.index = temp
    df_int0_Iratio.columns = pO2
    df_int0_Iratio = df_int0_Iratio.T

    # tauP
    list_tauP = []
    for i, k in enumerate(calib_data['tauP [ms]'].values[0].split(' ')):
        if len(k) > 1:
            if '\r' in k or '\n' in k:
                if '.' in k:
                    if '\r' in k:
                        list_tauP.append(np.float(k.split('\r\n')[0]))
                    else:
                        list_tauP.append(np.float(k.split('\n')[0]))
            else:
                list_tauP.append(np.float(k))
    if len(list_tauP) > len(pO2):
        list_tauP = list_tauP[len(temp):]
    df_tauP = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
    for i, k in enumerate(list_tauP):
        df_tauP.iloc[i % len(temp), int(i / len(temp))] = k

    df_tauP.index = temp
    df_tauP.columns = pO2
    df_tauP = df_tauP.T

    # tau0 / tauP
    list_tau_quot = []
    for i, k in enumerate(calib_data['tau0 / tauP'].values[0].split(' ')):
        if len(k) > 1:
            if '\r' in k or '\n' in k:
                if '.' in k:
                    if '\r' in k:
                        list_tau_quot.append(np.float(k.split('\r\n')[0]))
                    else:
                        list_tau_quot.append(np.float(k.split('\n')[0]))
            else:
                list_tau_quot.append(np.float(k))
    if len(list_tau_quot) > len(pO2):
        list_tau_quot = list_tau_quot[len(temp):]

    df_tau_quot = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
    for i, k in enumerate(list_tau_quot):
        df_tau_quot.iloc[i % len(temp), int(i / len(temp))] = k

    df_tau_quot.index = temp
    df_tau_quot.columns = pO2
    df_tau_quot = df_tau_quot.T

    dic_calib_data = pd.Series({'Temperature': temp, 'pO2': pO2, 'I-ratio': df_Iratio, 'I0 / I-ratio': df_int0_Iratio,
                                'tauP': df_tauP, 'tau0 / tauP': df_tau_quot})

    if 'error_tauP [ms]' in calib_data.columns:
        list_tau_er = []
        tau_er_ = calib_data['error_tauP [ms]'].values[0]
        for i, k in enumerate(tau_er_.split(' ')):
            if len(k) > 1:
                if '\r' in k or '\n' in k:
                    if '.' in k:
                        if '\r\n' in k:
                            list_tau_er.append(np.float(k.split('\r\n')[0]))
                        elif '\n' in k:
                            list_tau_er.append(np.float(k[:-2]))
                        else:
                            list_tau_er.append(np.float(k))
                else:
                    list_tau_er.append(np.float(k))
        if len(list_tau_er) > len(pO2):
            list_tau_er = list_tau_er[len(temp):]

        df_tau_er = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
        for i, k in enumerate(list_tau_er):
            df_tau_er.iloc[i % len(temp), int(i/ len(temp))] = k
        df_tau_er.index = temp
        df_tau_er.columns = pO2
        df_tau_er = df_tau_er.T
        dic_calib_data['error tau'] = df_tau_er

    if 'error_I-ratio' in calib_data.columns.tolist():
        list_Iratio_er = []
        iratio_er_ = calib_data['error_I-ratio'].values[0]
        for i, k in enumerate(iratio_er_.split(' ')):
            if len(k) > 1:
                if '\r' in k or '\n' in k:
                    if '.' in k:
                        if '\r\n' in k:
                            list_Iratio_er.append(np.float(k.split('\r\n')[0]))
                        elif '\n' in k:
                            list_Iratio_er.append(np.float(k[:-2]))
                        else:
                            list_Iratio_er.append(np.float(k))
                else:
                    list_Iratio_er.append(np.float(k))

        if len(list_Iratio_er) > len(pO2):
            list_Iratio_er = list_Iratio_er[len(temp):]

        df_Iratio_er = pd.DataFrame(np.zeros(shape=(len(temp), len(pO2))))
        for i, k in enumerate(list_Iratio_er):
            df_Iratio_er.iloc[i % len(temp), int(i / len(temp))] = k
        df_Iratio_er.index = temp
        df_Iratio_er.columns = pO2
        df_Iratio_er = df_Iratio_er.T
        dic_calib_data['error I-ratio'] = df_Iratio_er

    return dic_calib_data


def convert_lifetime_unit_into_seconds(tau):
    if isinstance(tau, np.float):
        test_tau = tau
    elif isinstance(tau, pd.DataFrame):
        test_tau = tau.loc[tau.index[0], tau.columns[0]]
    else:
        test_tau = tau[0]

    if (np.abs(test_tau) > 1000000.) == False:
        if (np.abs(test_tau) > 1000.) == False:
            if (np.abs(test_tau) > 1.) == False:
                if (np.abs(test_tau) > 1e-3) == False:
                    unit = None
                else:
                    unit = 's'
            else:
                unit = 'ms'
        else:
            unit = 'µs'
    else:
        unit = 'ns'

    return unit


def linear_regression(res_fit, temp_fit, param_list):
    regression = pd.DataFrame(np.zeros(shape=(4, len(temp_fit))), index=param_list, columns=temp_fit)
    reg_param = pd.DataFrame(np.zeros(shape=(4, 2)), index=param_list, columns=['slope', 'abscissa'])

    for p in res_fit.index:
        arg = stats.linregress(x=res_fit.columns, y=res_fit.T[p])
        regression.loc[p] = arg[0] * temp_fit + arg[1]
        reg_param.loc[p, 'slope'] = arg[0]
        reg_param.loc[p, 'abscissa'] = arg[1]

    return reg_param, regression


def twoSiteModel_calculation(tau0, tau, ksv, m, f):
    # preparation pq equation
    if isinstance(tau0, np.float):
        if isinstance(tau, np.float):
            quot = tau / tau0
        else:
            quot = [t1 / tau0 for t1 in tau]
    else:
        if isinstance(tau, np.float):
            quot = [tau / t0 for t0 in tau0]
        else:
            quot = [t1 / t0 for (t1, t0) in zip(tau, tau0)]

    c = [1 - 1 / q for q in quot]
    if isinstance(ksv, np.float):
        b = [ksv * (m * (1 - f / t) + 1 / t * (f - 1) + 1) for t in quot]
        a = m * (ksv ** 2)
    else:
        b = [k * (m * (1 - f / t) + 1 / t * (f - 1) + 1) for (k, t) in zip(ksv, quot)]
        a = [m * (p ** 2) for p in ksv]

    if isinstance(a, np.float):
        sqrt = [np.sqrt(b_ ** 2 - 4 * a * c_) for (b_, c_) in zip(b, c)]
        n = 2 * a
    else:
        sqrt = [np.sqrt(b_ ** 2 - 4 * a_ * c_) for (b_, a_, c_) in zip(b, a, c)]
        n = [2 * a_ for a_ in a]
    z1 = [-1 * b_ + s for (b_, s) in zip(b, sqrt)]
    z2 = [-1 * b_ - s for (b_, s) in zip(b, sqrt)]

    if isinstance(n, np.float):
        pO2_1 = sorted([z / n for z in z1])
        pO2_2 = sorted([z / n for z in z2])
    else:
        pO2_1 = sorted([z / n for (z, n) in zip(z1, n)])
        pO2_2 = sorted([z / n for (z, n) in zip(z2, n)])

    return pO2_1, pO2_2


def temperature_oxygen_individual_fit(file_path, to_fit, to_fit1, temp_range, params_tsm, res_fit_tau=None,
                                      res_fit_int=None, which='tauP', savefig=False, plotting=False, info=True):
    if info is True:
        print('\n')
        print('#2 linear regression along the temperature range')
    temp_fit = np.linspace(start=temp_range[0], stop=temp_range[1], num=10)
    param_list = list(params_tsm.keys())
    param_list.append('chi square')

    if which == 'tauP': # meaning tau / tau0
        if info is True:
            print('regression for ', to_fit)
        if res_fit_tau is None:
            raise ValueError('res_fit_tau_tau0 is required')
        reg_param_tau, regression_tau = linear_regression(res_fit=res_fit_tau, temp_fit=temp_fit,
                                                          param_list=param_list)
        if plotting is True:
            f_tau = plotting_fit_regression(res_fit=res_fit_tau, regression=regression_tau, type_='tauP')
        else:
            f_tau = None
        f_iratio = None
        reg_param_iratio = None
        regression_iratio = None
    elif which == 'I': # meaning to_fit1
        if info is True:
            print('regression for ', to_fit1)
        if res_fit_int is None:
            raise ValueError('res_fit_int is required')
        regression_tau = None
        reg_param_tau = None
        reg_param_iratio, regression_iratio = linear_regression(res_fit=res_fit_int, temp_fit=temp_fit,
                                                                param_list=param_list)
        if plotting is True:
            f_iratio = plotting_fit_regression(res_fit=res_fit_int, regression=regression_iratio, type_='I-ratio')
        else:
            f_iratio = None
        f_tau = None
    else: # which == 'both'
        if info is True:
            print('regression for ', to_fit)
            print('regression for ', to_fit1)
        if res_fit_int is None:
            raise ValueError('res_fit_int is required')
        if res_fit_tau is None:
            raise ValueError('res_fit_tau_quot is required')

        reg_param_tau, regression_tau = linear_regression(res_fit=res_fit_tau, temp_fit=temp_fit,
                                                          param_list=param_list)
        reg_param_iratio, regression_iratio = linear_regression(res_fit=res_fit_int, temp_fit=temp_fit,
                                                                param_list=param_list)
        if plotting is True:
            f_tau = plotting_fit_regression(res_fit=res_fit_tau, regression=regression_tau, type_=to_fit,
                                            figposition=[700, 100])
            f_iratio = plotting_fit_regression(res_fit=res_fit_int, regression=regression_iratio, type_=to_fit1,
                                               figposition=[700, 600])
        else:
            f_tau = None
            f_iratio = None

    if savefig is True:
        path_folder = '/'.join(file_path.split('/')[:-1]) + '/FitReport/'
        time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        if which == 'tauP':
            sav_name_tau = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                       '_parameterfit_tsm_{}.png'.format(to_fit.replace('/', '-'))
            f_tau.savefig(sav_name_tau)
        elif which == 'I-ratio':
            sav_name_i = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                         '_parameterfit_tsm_{}.png'.format(to_fit1.replace('/', '-'))
            f_iratio.savefig(sav_name_i)
        else:
            sav_name_i = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                         '_parameterfit_tsm_{}.png'.format(to_fit1.replace('/', '-'))
            sav_name_tau = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                           '_parameterfit_tsm_{}.png'.format(to_fit.replace('/', '-'))
            f_tau.savefig(sav_name_tau)
            f_iratio.savefig(sav_name_i)
    return f_tau, f_iratio, reg_param_tau, regression_tau, reg_param_iratio, regression_iratio


def limit_range_plotting(m_min, m_max, percent=5.):
    if m_min < 0:
        min_m = m_min * (1. + percent / 100)
    else:
        min_m = m_min * (1. - percent / 100)
    if m_max < 0:
        max_m = m_max * (1. - percent / 100)
    else:
        max_m = m_max * (1. + percent / 100)

    return min_m, max_m


def fitting_parameters_tsm(params_tsm, df, tsm_fit, fitpara='tauP', saving=False, p=None, info=True):
    """

    :param p:
    :param params_tsm:      parameters of the two-site model: m, k and f
    :param df:              measured data that should be fitted (y-values)
    :param tsm_fit:         model defined by lmfit
    :param pO2:             oxygen range where the sensor should be fitted
    :param saving:
    :return:
    """
    if info is True:
        print('fitting ', fitpara, '...')
    param_list = list(params_tsm.keys())
    param_list.append('chi square')

    if saving is True:
        if p is None:
            raise ValueError('Path to store the results is required!')

    res_fitparam = pd.DataFrame(np.zeros(shape=(len(params_tsm) + 1, len(df.columns))), index=param_list,
                                columns=df.columns)

    for temp_used in df.columns:
        data_ = df[temp_used].replace([np.inf, -np.inf], np.nan)
        data = data_.dropna()
        result_tsm_fit = tsm_fit.fit(data=data, x=data.index, params=params_tsm)
        report_ = result_tsm_fit.fit_report().split('\n')

        m_param = []
        k_param = []
        f_param = []
        chi_sq = []
        for k, line in enumerate(report_):
            if 'chi-square' in line:
                if 'reduced ' in line:
                    pass
                else:
                    chi_sq.append(np.float(line.split('=')[1]))
            if 'm:' in line:
                for s in line.split(' '):
                    if len(s) > 0 and '.' in s:
                        m_param.append(s)
            if 'k:' in line:
                for s in line.split(' '):
                    if len(s) > 0 and '.' in s:
                        k_param.append(s)
            if 'f:' in line:
                for s in line.split(' '):
                    if len(s) > 0 and '.' in s:
                        f_param.append(s)

        res_fitparam.loc['m', temp_used] = np.float(m_param[0])
        res_fitparam.loc['k', temp_used] = np.float(k_param[0])
        res_fitparam.loc['f', temp_used] = np.float(f_param[0])
        res_fitparam.loc['chi square', temp_used] = chi_sq[0]

        # ----------------------------------------------------------------
        # saving reports
        if saving is True:
            time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            path = '/'.join(p.split('/')[:-1]) + '/'
            folder_fitreport = path + 'FitReport/'
            if not os.path.exists(folder_fitreport):
                os.makedirs(folder_fitreport)

            sav_name = folder_fitreport + time_now.split(' ')[0] + '_' + \
                       time_now.split(' ')[1][:-3].replace(':', '') +\
                       '_fitreport_pO2_tsm-{}_{:.2f}deg.txt'.format(fitpara.replace('/', '-'), temp_used)
            with open(sav_name, 'w') as f:
                for item in report_:
                    f.write("%s\n" % item)

    if saving is True:
        # res_fitparam
        sav_param = folder_fitreport + time_now.split(' ')[0] + '_' + \
                    time_now.split(' ')[1][:-3].replace(':', '') + '_fitparameters_{}_common.txt'.format(fitpara.replace('/', '-'))
        res_fitparam.to_csv(sav_param, sep='\t', decimal='.')

    return res_fitparam


def multi_regression_temp_oxygen(param_temp, temp_range, pO2_range):
    # !!!TODO: pre-check if param_temp contains slope and abszissa for m, f and Ksv
    y_y0 = pd.DataFrame(np.zeros(shape=(len(pO2_range), len(temp_range))), index=pO2_range, columns=temp_range)
    for i in pO2_range:
        for c in temp_range:
            # linear regression
            m_reg = param_temp.loc['m', 'slope'] * c + param_temp.loc['m', 'abscissa']
            f_reg = param_temp.loc['f', 'slope'] * c + param_temp.loc['f', 'abscissa']
            k_reg = param_temp.loc['k', 'slope'] * c + param_temp.loc['k', 'abscissa']
            y_y0.loc[i, c] = f_reg / (1 + k_reg * i) + (1 - f_reg) / (1 + k_reg * i * m_reg)
    return y_y0


def multi_regression_temp_oxygen_common_plot(to_fit, to_fit1, file_path, pO2_range, temp_range, plotting=True,
                                             which='both', slit=1., reg_param_tau=None, df_tau0_fit=None, zlim_tau=None,
                                             df_i0_fit=None, temp_lim=None, reg_param_int=None, pO2_lim=None,
                                             zlim_int=None, labelpad_=1., saving=False, info=True):
    if len(pO2_range) <= 3:
        pO2_range = np.linspace(start=pO2_range[0], stop=pO2_range[1],
                                num=int((pO2_range[1] - pO2_range[0]) / slit + 1))
    else:
        pO2_range = pO2_range
    if len(temp_range) <= 3:
        temp_range = np.linspace(start=temp_range[0], stop=temp_range[1],
                                 num=int((temp_range[1] - temp_range[0]) / slit + 1))
    else:
        temp_range = temp_range

    # multi regression in two dimensions either for tau / tau0 (according to original two-site-model),
    # I-ratio or for both
    if which == 'tauP': # lifetime fitting
        if reg_param_tau is None:
            raise ValueError('reg_param_tau is required')
        if df_tau0_fit is None:
            raise ValueError('df_tau0_fit is required')

        # calculate tau / tau0 across the temperature and oxygen range
        if to_fit == 'tauP / tau0':
            if info is True:
                print('calculate tau0 / tau after fitting of tauP / tau0')
            tau_tau0 = multi_regression_temp_oxygen(param_temp=reg_param_tau, temp_range=temp_range, pO2_range=pO2_range)
            tau0_tau = 1 / tau_tau0
        elif to_fit == 'tau0 / tauP':
            if info is True:
                print('calculate tauP / tau0 after fitting of tau0 / tauP')
            tau0_tau = multi_regression_temp_oxygen(param_temp=reg_param_tau, temp_range=temp_range, pO2_range=pO2_range)
            tau_tau0 = 1 / tau0_tau
        else: # to_fit might be tauP
            if info is True:
                print('calculate tau / tau0 after fitting of tau...')
                print('... what to do with tauP?')
            tau_tau0 = None

        # re-calculate tauP across the temperature range
        tau0_fit_3d = tau_tau0.copy()
        for i in temp_range:
            tau0_fit_3d[i] = tau_tau0[i] * df_tau0_fit[i].values  # tauP in ms

        Iratio_i0 = None
    elif which == 'I': # I-ratio fitting
        if reg_param_int is None:
            raise ValueError('reg_param_iratio is required')
        if df_i0_fit is None:
            raise ValueError('df_Iratio0_fit is required')

        # calculate I-ratio / I0 across the temperature and oxygen range
        if to_fit1 == 'I0 / I':
            if info is True:
                print('calculate I-ratio/I0 after fitting of I0 / I-ratio')
            i0_Iratio = multi_regression_temp_oxygen(param_temp=reg_param_int, temp_range=temp_range, pO2_range=pO2_range)
            Iratio_i0 = 1 / i0_Iratio
        elif to_fit1 == 'I / I0':
            if info is True:
                print('calculate I0 / I-ratio after fitting of I-ratio / I0')
            Iratio_i0 = multi_regression_temp_oxygen(param_temp=reg_param_int, temp_range=temp_range, pO2_range=pO2_range)
            i0_Iratio = 1 / Iratio_i0
        else:
            if info is True:
                print('calculate I-ratio / I0 after fitting of I-ratio...')
                print('... what to do with I-ratio?')
            Iratio_i0 = None

        # re-calculate Iratio across the temperature range
        Iratio_fit_3d = i0_Iratio.copy()
        for i in temp_range:
            Iratio_fit_3d[i] = df_i0_fit[i].values / i0_Iratio[i] # I-ratio @ pO2 = 0hPa

        tau_tau0 = None
    else:  # which == 'both'
        if reg_param_tau is None:
            raise ValueError('reg_param_tau is required')
        if reg_param_int is None:
            raise ValueError('reg_param_int is required')
        if df_tau0_fit is None:
            raise ValueError('df_tau0_fit is required')
        if df_i0_fit is None:
            raise ValueError('df_Iratio0_fit is required')

        # -------------------------------------------------------------------------------
        # calculate tau / tau0 across the temperature and oxygen range
        if to_fit == 'tauP / tau0':
            if info is True:
                print('calculate tau0 / tauP after fitting of tauP / tau0')
            tau_tau0 = multi_regression_temp_oxygen(param_temp=reg_param_tau, temp_range=temp_range,
                                                    pO2_range=pO2_range)
            tau0_tau = 1 / tau_tau0
        elif to_fit == 'tau0 / tauP':
            if info is True:
                print('calculate tauP / tau0 after fitting of tau0 / tauP')
            tau0_tau = multi_regression_temp_oxygen(param_temp=reg_param_tau, temp_range=temp_range,
                                                    pO2_range=pO2_range)
            tau_tau0 = 1 / tau0_tau
        else:  # to_fit might be tauP
            if info is True:
                print('calculate tau / tau0 after fitting of tau...')
                print('... what to do with tauP?')
            tau_tau0 = None

        # re-calculate tauP across the temperature range
        tau0_fit_3d = tau_tau0.copy()
        for i in temp_range:
            tau0_fit_3d[i] = tau_tau0[i] * df_tau0_fit[i].values  # tauP in ms

        # -------------------------------------------------------------------------------
        # calculate I-ratio / I0 across the temperature and oxygen range
        if to_fit1 == 'I0 / I-ratio':
            if info is True:
                print('calculate I-ratio/I0 after fitting of I0 / I-ratio')
            i0_Iratio = multi_regression_temp_oxygen(param_temp=reg_param_int, temp_range=temp_range,
                                                     pO2_range=pO2_range)
            Iratio_i0 = 1 / i0_Iratio
        elif to_fit1 == 'I-ratio / I0':
            if info is True:
                print('calculate I0 / I-ratio after fitting of I-ratio / I0')
            Iratio_i0 = multi_regression_temp_oxygen(param_temp=reg_param_int, temp_range=temp_range,
                                                     pO2_range=pO2_range)
            i0_Iratio = 1 / Iratio_i0
        else:
            if info is True:
                print('calculate I-ratio / I0 after fitting of I-ratio...')
                print('... what to do with I-ratio?')
            Iratio_i0 = None

        # re-calculate I-ratio across the temperature range
        Iratio_fit_3d = i0_Iratio.copy()
        for i in temp_range:
            Iratio_fit_3d[i] = df_i0_fit[i].values * Iratio_i0[i]  # I-ratio @ pO2 = 0hPa

    # ------------------------------------------------------------------------------------------------------
    # generate plots - depending whether plotting is True display figures (True) or not (False)
    temp_X, pO2_Y = np.meshgrid(temp_range, pO2_range)
    plt.ioff()
    fig = plt.figure(figsize=(6, 3))

    if which == 'tauP':
        ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    elif which == 'Iratio':
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    if which == 'tauP':
        ax0.plot_wireframe(X=temp_X, Y=pO2_Y, Z=tau0_fit_3d, color='navy')
        ax0.tick_params(which='both', labelsize=8, labelpad=labelpad_)
        ax0.set_xlabel('Temperature [°C]', fontsize=8)
        ax0.set_ylabel('pO2 [hPa]', fontsize=8, labelpad=labelpad_)
        ax0.set_zlabel('tauP [ms]', fontsize=8, labelpad=labelpad_)
        if temp_lim is not None:
            ax0.set_xlim(temp_lim[0], temp_lim[1])
        if pO2_lim is not None:
            ax0.set_ylim(pO2_lim[0], pO2_lim[1])
        if zlim_tau is not None:
            ax0.set_zlim(zlim_tau[0], zlim_tau[1])
    elif which == 'Iratio':
        ax1.plot_wireframe(X=temp_X, Y=pO2_Y, Z=Iratio_fit_3d, color='forestgreen')
        ax1.tick_params(which='both', labelsize=8)
        ax1.set_xlabel('Temperature [°C]', fontsize=8, labelpad=labelpad_)
        ax1.set_ylabel('pO2 [hPa]', fontsize=8, labelpad=labelpad_)
        ax1.set_zlabel('I-ratio', fontsize=8, labelpad=labelpad_)
        if temp_lim is not None:
            ax1.set_xlim(temp_lim[0], temp_lim[1])
        if pO2_lim is not None:
            ax1.set_ylim(pO2_lim[0], pO2_lim[1])
        if zlim_int is not None:
            ax1.set_zlim(zlim_int[0], zlim_int[1])
    else:
        # lifetime tau0 / tauP
        ax0.plot_wireframe(X=temp_X, Y=pO2_Y, Z=tau0_fit_3d, color='navy')
        ax0.tick_params(which='both', labelsize=8)
        ax0.set_xlabel('Temperature [°C]', fontsize=8, labelpad=labelpad_)
        ax0.set_ylabel('pO2 [hPa]', fontsize=8, labelpad=labelpad_)
        ax0.set_zlabel('tauP [ms]', fontsize=8, labelpad=labelpad_)
        if temp_lim is not None:
            ax0.set_xlim(temp_lim[0], temp_lim[1])
        if pO2_lim is not None:
            ax0.set_ylim(pO2_lim[0], pO2_lim[1])
        if zlim_tau is not None:
            ax0.set_zlim(zlim_tau[0], zlim_tau[1])

        # intensity ratio
        ax1.plot_wireframe(X=temp_X, Y=pO2_Y, Z=Iratio_fit_3d, color='forestgreen')
        ax1.tick_params(which='both', labelsize=8, pad=1)
        ax1.set_xlabel('Temperature [°C]', fontsize=8, labelpad=labelpad_)
        ax1.set_ylabel('pO2 [hPa]', fontsize=8, labelpad=labelpad_)
        ax1.set_zlabel('I-ratio', fontsize=8, labelpad=labelpad_)
        if temp_lim is not None:
            ax1.set_xlim(temp_lim[0], temp_lim[1])
        if pO2_lim is not None:
            ax1.set_ylim(pO2_lim[0], pO2_lim[1])
        if zlim_int is not None:
            ax1.set_zlim(zlim_int[0], zlim_int[1])

    fig.subplots_adjust(left=0.05, right=0.94, wspace=0.12)

    if plotting is True:
        plt.show()
    else:
        plt.close(fig)

    # ---------------------------------------------------------------------------------------------------------------
    if saving is True:
        path_folder = '/'.join(file_path.split('/')[:-1]) + '/FitReport/'
        time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        if which == 'tauP':
            sav_name = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                       '_3DFit_tsm_tauP.png'
        elif which == 'I-ratio':
            sav_name = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                       '_3DFit_tsm_Iratio.png'
        else:
            sav_name = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
                       '_3DFit_tsm_tauP_Iratio.png'
        fig.savefig(sav_name)

    return tau0_fit_3d, tau_tau0, Iratio_fit_3d, Iratio_i0, fig


# def fitting_tsm_for_T_O2(file, calib_data, temp_range, pO2_range, params_tsm_tau, params_tsm_int, tsm_fit, usedata=None,
#                          to_fit='tauP / tau0', to_fit1='1/I', slit=1., zlim_tau=None, zlim_int=None, savefig=False,
#                          saving_report=False, plotting=False, info=True):
#     # Load calibration input data
#     calib_data_df = plotting_calibration_input(df=calib_data, usedata=usedata, plotting=plotting)
#
#     check_tau = 'tauP / tau0' in calib_data.index
#     check_tau1 = 'tau0 / tauP' in calib_data.index
#     if check_tau == False and check_tau1 == True:
#         calib_data_df['tauP / tau0'] = 1 / calib_data['tau0 / tauP']
#     else:
#         raise ValueError('tauP / tau0 or tau0 / tauP is required')
#
#     check_int = 'I-ratio' in calib_data.index
#     check_int1 = 'I' in calib_data.index
#     if check_int == False and check_int1 == False:
#         raise ValueError('I-ratio is required')
#
#     # ===================================================================================
#     # Fitting of calibration data
#     # ===================================================================================
#     # linear regression along temperature
#     # --------------------------------------
#     if len(temp_range) <= 3:
#         temp_fit = np.linspace(start=temp_range[0], stop=temp_range[1], num=int((temp_range[1] - temp_range[0])/slit + 1.))
#     else:
#         temp_fit = temp_range
#
#     tau_calib_temp = pd.DataFrame(np.nan, index=calib_data['tauP'].index, columns=temp_fit)
#     iratio_calib_temp = pd.DataFrame(np.nan, index=calib_data['I-ratio'].index, columns=temp_fit) # i-ratio := promt/delayed
#
#     for i in tau_calib_temp.index:
#         arg = stats.linregress(x=calib_data['tauP'].columns, y=calib_data['tauP'].loc[i].values)
#         yreg = arg[0] * temp_fit + arg[1]
#         tau_calib_temp.loc[i, :] = yreg
#     tau0 = tau_calib_temp.loc[0, :]
#     tau_tau0 = tau_calib_temp / tau_calib_temp.loc[0, :]
#
#     # tau0 = calib_data['tauP'].loc[0.]
#     # arg = stats.linregress(x=tau0.index, y=tau0)
#     # tau0_fit = arg[0] * temp_fit + arg[1]
#     # df_tau0_fit = pd.DataFrame(tau0_fit, index=temp_fit).T
#
#     # --------------------------------------
#     # linear regression for I-ratio @ pO2 = 0hPa --> I-ratio(0)
#     for i in iratio_calib_temp.index:
#         arg = stats.linregress(x=calib_data['I-ratio'].columns, y=calib_data['I-ratio'].loc[i].values)
#         yreg = arg[0] * temp_fit + arg[1]
#         iratio_calib_temp.loc[i, :] = yreg
#     int0 = iratio_calib_temp.loc[0, :]
#
#     # i_ratio0 = calib_data['I-ratio'].loc[0.]
#     # arg = stats.linregress(x=i_ratio0.index, y=i_ratio0)
#     # i_ratio0_fit = arg[0] * temp_fit + arg[1]
#     # df_i0_fit = pd.DataFrame(i_ratio0_fit, index=temp_fit).T
#
#     # ===================================================================================
#     # tsm fit along pO2 --> fitting as tau/tau0 or i-ratio for each temperature
#     # ===================================================================================
#     if len(pO2_range) <= 3:
#         pO2_fit = np.linspace(start=pO2_range[0], stop=pO2_range[1], num=int((pO2_range[1] - pO2_range[0])/slit + 1.))
#     else:
#         pO2_fit = pO2_range
#
#     # lifetime
#     if info is True:
#         print('#1 parameter fitting for two site model')
#         print('lifetime: ', to_fit)
#
#     if to_fit == 'tauP / tau0':
#         ydata = tau_tau0
#     else:
#         ydata = 1 / tau_tau0
#
#     res_fitparam_tau = fitting_parameters_tsm(params_tsm=params_tsm_tau, df=ydata, tsm_fit=tsm_fit,
#                                               fitpara=to_fit, saving=False, info=True)
#     y_reg = pd.DataFrame(np.nan, index=pO2_fit, columns=temp_fit)
#     y_reg2 = y_reg.copy()
#     for t in y_reg2.columns:
#         y_reg.loc[:, t] = function_TSM(x=pO2_fit, f=res_fitparam_tau[t]['f'], m=res_fitparam_tau[t]['m'],
#                                        k=res_fitparam_tau[t]['k'])
#         y_reg2.loc[:, t] = y_reg[t] * tau0.loc[t]
#
#     if to_fit == 'tau/tau0':
#         tau_tau0_contour = y_reg
#         tau_contour = y_reg2
#     else:
#         tau_tau0_contour = y_reg2
#         tau_contour = y_reg
#     # res_fit_lifetime = fitting_parameters_tsm(p=file, params_tsm=params_tsm_tau, df=calib_data_df[to_fit],
#     #                                           tsm_fit=tsm_fit, fitpara='tau', saving=saving_report, info=info)
#     # ---------------------------------------------------------------------------
#     # intensity ratio
#     if info is True:
#         print('intensity: ', to_fit1)
#     if to_fit1 in calib_data_df.index.tolist():
#         ydata_int = iratio_calib_temp
#     else:
#         if to_fit1 == '1/I' or to_fit1 == '1 / I':
#             ydata_int = 1/iratio_calib_temp
#         else:
#             if to_fit1.split('/')[0].rstrip() == 'I0':
#                 to_fit_find = to_fit1.split('/')[0].rstrip() + ' / '
#             else:
#                 to_fit_find = to_fit1.split('/')[0].rstrip() + '-ratio /'
#             for c in calib_data_df.index:
#                 if to_fit_find in c:
#                     to_fit1 = c
#             ydata_int = calib_data_df[to_fit1]
#
#     # fitting tauP / tau0 according to two-site-model
#     res_fitparam_int = fitting_parameters_tsm(params_tsm=params_tsm_int, df=ydata_int, tsm_fit=tsm_fit,
#                                               fitpara=to_fit1, saving=saving_report, info=info)
#
#     df_reg = pd.DataFrame(np.nan, index=pO2_fit, columns=temp_fit)
#     df_reg2 = df_reg.copy()
#     for t in df_reg2.columns:
#         df_reg.loc[:, t] = function_TSM(x=pO2_fit, f=res_fitparam_int[t]['f'], m=res_fitparam_int[t]['m'],
#                                         ksv=res_fitparam_int[t]['k']) / int0.loc[t]
#         df_reg2.loc[:, t] = 1 / df_reg[t]
#
#     if to_fit1 == '1/I' or to_fit1 == '1 / I':
#         i_d_p_contour = df_reg
#         i_p_d_contour = df_reg2
#     else:
#         i_d_p_contour = df_reg2
#         i_p_d_contour = df_reg
#
#     #res_fit_int = fitting_parameters_tsm(p=file, params_tsm=params_tsm_int, df=calib_data_df[to_fit1],  tsm_fit=tsm_fit,
#     #                                     fitpara=to_fit1, saving=saving_report, info=info)
#
#     # individual parameters (tau / tau0 and Iratio) are linear fitted across the temperature range
#     #[fig_tau, fig_iratio, reg_param_lifetime, reg_lifetime, reg_param_int,
#     # reg_int] = temperature_oxygen_individual_fit(to_fit=to_fit, to_fit1=to_fit1, file_path=file, which='both',
#     #                                             temp_range=temp_range, params_tsm=params_tsm_tau, savefig=savefig,
#     #                                              res_fit_tau=res_fit_lifetime, res_fit_int=res_fit_int,
#     #                                              plotting=plotting, info=info)
#
#     # ---------------------------------------------------------------------------------------------------------------
#     if info is True:
#         print('\n')
#         print('#3 multiparametric regression')
#     [tau0_3d, tau_tau0, iratio_3d, Iratio_i0,
#      fig_3D] = multi_regression_temp_oxygen_common_plot(to_fit=to_fit, to_fit1=to_fit1, file_path=file, which='both',
#                                                         pO2_range=pO2_range, temp_range=temp_range, plotting=plotting,
#                                                         slit=slit, temp_lim=None, reg_param_tau=reg_param_lifetime,
#                                                         reg_param_int=reg_param_int, df_tau0_fit=df_tau0_fit,
#                                                         df_i0_fit=df_i0_fit, pO2_lim=None, zlim_tau=zlim_tau,
#                                                         zlim_int=zlim_int, labelpad_=-1., saving=savefig, info=info)
#
#     # check if tau0_3d is given in ms or seconds and, in case, convert to seconds
#     unit = convert_lifetime_unit_into_seconds(tau=tau0_3d)
#     tau0_3d = tau0_3d * 10 ** (switcher[unit])
#
#     return tau0_3d, tau_tau0, df_tau0_fit, iratio_3d, Iratio_i0, df_i0_fit, reg_param_lifetime, reg_lifetime, \
#            reg_param_int, reg_int


def fitting_tsm_for_T_O2_(file, temp_reg, pO2_reg, calib_range_temp=None, calib_range_O2=None, fitpara1='tau/tau0',
                          fitpara2='I/I0', plot_calib_data=True):
    # pre-defined TSM fit
    tsm_fit = Model(function_TSM)

    # load data
    calib_data = load_calibration_data(p=file)

    if calib_range_temp is None:
        calib_range_temp = [calib_data['tauP'].columns[0], calib_data['tauP'].columns[-1]]
    if calib_range_O2 is None:
        calib_range_O2 = [calib_data['tauP'].index[0], calib_data['tauP'].index[-1]]

    tau_meas = calib_data['tauP']*1e-3
    tau_calib_s = calib_data['tauP'].loc[calib_range_O2[0]:calib_range_O2[1],
                  calib_range_temp[0]:calib_range_temp[1]] * 1e-3
    tau0_tau_calib = calib_data['tau0 / tauP'].loc[calib_range_O2[0]:calib_range_O2[1],
                     calib_range_temp[0]:calib_range_temp[1]]
    tau0 = tau_calib_s.loc[0]
    tau0_meas = tau_meas.loc[0]
    iratio_meas = calib_data['I-ratio']
    i_i0_meas = iratio_meas / iratio_meas.loc[0]
    int0_meas = iratio_meas.loc[0]
    iratio_calib = calib_data['I-ratio'].loc[calib_range_O2[0]:calib_range_O2[1],
                   calib_range_temp[0]:calib_range_temp[1]]
    i_i0 = iratio_calib / iratio_calib.loc[0]
    int0 = iratio_calib.loc[0]

    # ==============================================================================================
    # initial parameters
    # for lifetime
    m0_tau = .95
    k0_tau = 0.25
    f0_tau = 2.

    # for i-ratio
    m0_iratio = 1.
    k0_iratio = 0.2
    f0_iratio = 0.4

    # make them to a parameter
    params_tsm_tau = tsm_fit.make_params(m=m0_tau, f=f0_tau, k=k0_tau)
    params_tsm_int = tsm_fit.make_params(m=m0_iratio, f=f0_iratio, k=k0_iratio)

    if plot_calib_data is True:
        col = sns.hls_palette(4, l=.3, s=.8)

        # plotting original data
        fig1 = plt.figure(figsize=(6, 3))
        ax0 = fig1.add_subplot(2, 2, 1, projection='3d')
        ax1 = fig1.add_subplot(2, 2, 2, projection='3d')
        ax2 = fig1.add_subplot(2, 2, 3, projection='3d')
        ax3 = fig1.add_subplot(2, 2, 4, projection='3d')

        for n, i in enumerate(i_i0.columns):
            ax0.plot(tau_calib_s.index, [i] * len(tau_calib_s.index), tau_calib_s[i] * 1e3, marker='o', markersize=3,
                     color=col[0], alpha=0.8, lw=0.5)
            ax1.plot(iratio_calib.index, [i] * len(iratio_calib.index), iratio_calib[i], marker='o', markersize=3,
                     color=col[1], alpha=0.8, lw=0.5)
            ax2.plot(tau0_tau_calib.index, [i] * len(tau0_tau_calib.index), tau0_tau_calib[i], marker='o', markersize=3,
                     color=col[2], alpha=0.8, lw=0.5)
            ax3.plot(i_i0.index, [i] * len(i_i0.index), i_i0[i], marker='o', markersize=3, color=col[3], alpha=0.8,
                     lw=0.5)

        ax0.tick_params(which='both', labelsize=8)
        ax1.tick_params(which='both', labelsize=8)
        ax2.tick_params(which='both', labelsize=8)
        ax3.tick_params(which='both', labelsize=8)

        ax0.set_ylabel('Temperature [°C]', fontsize=8)
        ax0.set_xlabel('pO2 [hPa]', fontsize=8)
        ax1.set_ylabel('Temperature [°C]', fontsize=8)
        ax1.set_xlabel('pO2 [hPa]', fontsize=8)
        ax2.set_ylabel('Temperature [°C]', fontsize=8)
        ax2.set_xlabel('pO2 [hPa]', fontsize=8)
        ax3.set_ylabel('Temperature [°C]', fontsize=8)
        ax3.set_xlabel('pO2 [hPa]', fontsize=8)

        ax0.set_zlabel('tauP [ms]', fontsize=8)
        ax1.set_zlabel('I-ratio', fontsize=8)
        ax2.set_zlabel('tau0 / tauP', fontsize=8)
        ax3.set_zlabel('I/I0', fontsize=8)

        plt.subplots_adjust(left=0.1, right=0.9, wspace=.5, hspace=0.5)
    else:
        fig1 = None

    # ===================================================================================
    # 1) tsm fit along pO2 --> fitting as tau/tau0 or i-ratio for each temperature
    # ===================================================================================
    tau_calib_tsm = pd.DataFrame(np.nan, index=pO2_reg, columns=tau_calib_s.columns)
    iratio_calib_tsm = pd.DataFrame(np.nan, index=pO2_reg, columns=tau_calib_s.columns)

    # lifetime tau
    if fitpara1 == 'tau/tau0':
        print('plotting tau/tau0 but fitting according to tau0/tau')
        ydata = tau0_tau_calib
    else:
        ydata = 1 / tau0_tau_calib

    res_fitparam_tau = fitting_parameters_tsm(params_tsm=params_tsm_tau, df=ydata, tsm_fit=tsm_fit, fitpara=fitpara1,
                                              saving=False, info=True)
    print('...fit done... ')

    y_reg = tau_calib_tsm
    y_reg2 = tau_calib_tsm.copy()
    for t in y_reg.columns:
        y_reg.loc[:, t] = function_TSM(x=pO2_reg, f=res_fitparam_tau[t]['f'], m=res_fitparam_tau[t]['m'],
                                       k=res_fitparam_tau[t]['k'])
        y_reg2.loc[:, t] = tau0.loc[t] / y_reg[t]

    if fitpara1 == 'tau/tau0':
        tau_tau0_ox = 1 / y_reg.sort_index(axis=1)
        tau_ox = y_reg2.sort_index(axis=1)
    else:
        tau_tau0_ox = y_reg2.sort_index(axis=1)
        tau_ox = y_reg.sort_index(axis=1)

    # ---------------------------------------------------------------------------
    # intensity ratio
    if fitpara2 == '1/I':
        ydata_int = 1 / iratio_calib
    elif fitpara2 == 'I/I0':
        print('plotting I/I0 but fitting according to I0/I')
        ydata_int = 1 / i_i0
    else:
        ydata_int = i_i0
    res_fitparam_int = fitting_parameters_tsm(params_tsm=params_tsm_int, df=ydata_int, tsm_fit=tsm_fit,
                                              fitpara=fitpara2, saving=False, info=True)
    print('...fit done... ')

    df_reg = iratio_calib_tsm
    for t in res_fitparam_int.columns:
        if fitpara2 == 'I/I0':
            df_reg.loc[:, t] = function_TSM(x=pO2_reg, f=res_fitparam_int[t]['f'], m=res_fitparam_int[t]['m'],
                                            k=res_fitparam_int[t]['k']) / int0.loc[t]
        else:
            df_reg.loc[:, t] = function_TSM(x=pO2_reg, f=res_fitparam_int[t]['f'], m=res_fitparam_int[t]['m'],
                                            k=res_fitparam_int[t]['k'])
    df_reg2 = 1 / df_reg

    if fitpara2 == '1/I':
        i_d_p_ox = df_reg.sort_index(axis=1)
        i_p_d_ox = df_reg2.sort_index(axis=1)
    elif fitpara2 == 'I/I0':
        i_d_p_ox = df_reg.sort_index(axis=1)
        i_p_d_ox = df_reg2.sort_index(axis=1)
    else:
        i_d_p_ox = df_reg2.sort_index(axis=1)
        i_p_d_ox = df_reg.sort_index(axis=1)
    print('done')

    # ===================================================================================
    # 2) linear regression along temperature
    # ===================================================================================
    # lifetime
    tau_contour = pd.DataFrame(np.nan, index=pO2_reg, columns=temp_reg)
    tau_tau0_contour = pd.DataFrame(np.nan, index=pO2_reg, columns=temp_reg)

    for i in tau_ox.index:
        arg_tau = stats.linregress(x=tau_ox.columns, y=tau_ox.loc[i].values)
        yreg = arg_tau[0] * temp_reg + arg_tau[1]
        tau_contour.loc[i, :] = yreg

        arg_tau_tau0 = stats.linregress(x=tau_tau0_ox.columns, y=tau_tau0_ox.loc[i].values)
        yreg = arg_tau_tau0[0] * temp_reg + arg_tau_tau0[1]
        tau_tau0_contour.loc[i, :] = yreg

    # intensity ratio
    int_contour = pd.DataFrame(np.nan, index=pO2_reg, columns=temp_reg)
    int_invert_contour = pd.DataFrame(np.nan, index=pO2_reg, columns=temp_reg)

    for i in i_p_d_ox.index:
        arg_i = stats.linregress(x=i_p_d_ox.columns, y=i_p_d_ox.loc[i].values)
        yreg = arg_i[0] * temp_reg + arg_i[1]
        int_contour.loc[i, :] = yreg

        arg_i_inv = stats.linregress(x=i_d_p_ox.columns, y=i_d_p_ox.loc[i].values)
        yreg = arg_i_inv[0] * temp_reg + arg_i_inv[1]
        int_invert_contour.loc[i, :] = yreg

    # combining parameters
    para_ox = dict({'tau0 / tau': res_fitparam_tau, 'iratio': res_fitparam_int})
    para_temp = dict({'tau': arg_tau, 'tau_tau0': arg_tau_tau0, 'iratio': arg_i, '1/i-ratio': arg_i_inv})
    para_fit = dict({'SV oxygen fit': para_ox, 'linear temperature fit': para_temp})

    # ===================================================================================
    # plot resulting calibration surface tau and I-ratio
    # ===================================================================================
    temp_X, pO2_Y = np.meshgrid(tau_contour.columns, tau_contour.index)

    fig = plt.figure(figsize=(6, 3))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    # lifetime tau0 / tauP
    ax0.plot_surface(X=temp_X, Y=pO2_Y, Z=tau_contour * 1e3, rstride=1, cstride=1, color='navy', alpha=.9, linewidth=0.,
                     antialiased=True)

    ax0.tick_params(which='both', labelsize=8)
    ax0.set_xlabel('Temperature [°C]', fontsize=8)
    ax0.set_ylabel('pO2 [hPa]', fontsize=8)
    ax0.set_zlabel('tauP [ms]', fontsize=8)
    ax0.set_ylim(0, pO2_Y.max())
    ax0.set_xlim(temp_X.max(), 0)

    # intensity ratio
    ax1.plot_surface(X=temp_X, Y=pO2_Y, Z=int_contour, rstride=1, cstride=1, color='forestgreen', linewidth=0.,
                     antialiased=True)
    ax1.tick_params(which='both', labelsize=8, pad=1)
    ax1.set_xlabel('Temperature [°C]', fontsize=8)
    ax1.set_ylabel('pO2 [hPa]', fontsize=8)
    ax1.set_zlabel('I-ratio', fontsize=8)
    ax1.set_ylim(0, pO2_Y.max())
    ax1.set_xlim(temp_X.max(), 0)

    fig.subplots_adjust(left=0.05, right=0.94, wspace=0.25)

    return tau_contour, tau_tau0_contour, int_contour, int_invert_contour, fig1, fig, ax0, ax1, temp_reg, pO2_reg, para_fit


def intersection_surface_plane(tau_contour, tauP_meas, int_contour, iratio_meas):
    df_tauP_meas = tau_contour.copy()
    for ind in df_tauP_meas.index:
        for c in df_tauP_meas.columns:
            df_tauP_meas.loc[ind, c] = tauP_meas

    df_iratio_meas = int_contour.copy()
    for ind in df_iratio_meas.index:
        for c in df_iratio_meas.columns:
            df_iratio_meas.loc[ind, c] = iratio_meas

    # ===================================================================================
    # plane intersection
    # ===================================================================================
    # lifetime
    diff_tau = tau_contour - df_tauP_meas
    para_tau = pd.DataFrame(np.nan, index=['pO2'], columns=tau_contour.columns)

    for t in tau_contour.columns:
        para_ = find_limits_plane_intersection(T=t, diff=diff_tau, min_direction1=np.abs(diff_tau).min())
        arg = stats.linregress(y=diff_tau.loc[sorted(para_.index), t].index,
                               x=diff_tau.loc[sorted(para_.index), t].values)
        para_tau.loc['pO2', t] = arg[1]

    # ---------------------------------
    # intensity ratio
    diff_int = int_contour - df_iratio_meas
    para_int = pd.DataFrame(np.nan, index=['pO2'], columns=int_contour.columns)

    for t in int_contour.columns:
        para_ = find_limits_plane_intersection(T=t, diff=diff_int, min_direction1=np.abs(diff_int).min())
        arg = stats.linregress(y=para_.index, x=para_[para_.columns[0]].values)

        para_int.loc['pO2', t] = arg[1]

    return df_tauP_meas, df_iratio_meas, para_tau, para_int


def intersection_lines(temp_reg, para_tau, para_int):
    # ===================================================================================
    # line intersection
    # ===================================================================================
    temP_new = np.linspace(start=temp_reg[0], stop=temp_reg[-1], num=int((temp_reg[-1] - temp_reg[0]) / .1 + 1))

    # lifetime
    x = para_tau.columns
    y = para_tau.loc['pO2'].values
    tck = interpolate.splrep(x, y, s=10)
    y_tau = pd.DataFrame(interpolate.splev(temP_new, tck, der=0), columns=['pO2'], index=temP_new).T

    # intensity
    x = para_int.columns
    y = para_int.loc['pO2'].values
    tck = interpolate.splrep(x, y, s=10)
    y_int = pd.DataFrame(interpolate.splev(temP_new, tck, der=0), columns=['pO2'], index=temP_new).T

    diff_line = y_tau - y_int
    T_calc = np.abs(diff_line).idxmin(axis=1).values[0]
    pO2_calc = y_tau[T_calc].values[0]

    return y_tau, y_int, T_calc, pO2_calc


def measurement_evaluation_(temp_reg, tau_contour, int_contour, dphi_f1=None, dphi_f2=None, error_meas=None, f1=None,
                            f2=None, tauP_meas=None, iratio_meas=None, plotting=True, fig=None, ax_tau=None,
                            ax_int=None):
    if tauP_meas is None:
        if dphi_f1 is None or dphi_f2 is None or error_meas is None:
            raise ValueError('dPhi and assumed error are required')
        if f1 is None or f2 is None:
            raise ValueError('modulation frequencies are required')

        if isinstance(dphi_f1, list):
            er_ = False
        else:
            er_ = True

        [tauP_meas, Phi_f1_rad_er,
         Phi_f2_rad_er] = af.phi_to_lifetime_including_error(phi_f1=dphi_f1, phi_f2=dphi_f2, err_phaseangle=error_meas,
                                                             f1=f1, f2=f2, er=er_)
    if iratio_meas is None:
        dPhi_f1_deg = np.rad2deg(Phi_f1_rad_er)
        dPhi_f2_deg = np.rad2deg(Phi_f2_rad_er)
        iratio_meas = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tauP_meas, dphi_f1=dPhi_f1_deg, dphi_f2=dPhi_f2_deg)
    print('tauP ~ {:.2f}ms'.format(tauP_meas[1] * 1e3))
    print('Iratio calculated ~ {:.2f}'.format(iratio_meas[1]))

    # pre-check if tau  is given in seconds
    unit = convert_lifetime_unit_into_seconds(tau=tau_contour)
    tau_contour = tau_contour * 10 ** (switcher[unit])
    unit_meas = convert_lifetime_unit_into_seconds(tau=tauP_meas)
    tauP_meas = tauP_meas * 10 ** (switcher[unit_meas])

    ls_temp_calc = []
    ls_pO2_calc = []
    dic_tau_line = {}
    dic_int_line = {}
    dic_tau_3d = {}
    dic_int_3d = {}
    dic_tau_intersec = {}
    dic_int_intersec = {}
    for i, meas in enumerate(tauP_meas):
        [df_tauP_meas, df_iratio_meas, para_tau,
         para_int] = intersection_surface_plane(tau_contour=tau_contour, tauP_meas=meas, int_contour=int_contour,
                                                iratio_meas=iratio_meas[i])
        y_tau, y_int, T_calc, pO2_calc = intersection_lines(temp_reg=temp_reg, para_tau=para_tau, para_int=para_int)
        ls_temp_calc.append(T_calc)
        ls_pO2_calc.append(pO2_calc)
        dic_tau_line[i] = y_tau
        dic_int_line[i] = y_int
        dic_tau_3d[i] = df_tauP_meas
        dic_int_3d[i] = df_iratio_meas
        dic_tau_intersec[i] = para_tau
        dic_int_intersec[i] = para_int

    intersec_ = pd.Series({'df_tau': dic_tau_3d, 'df_int': dic_int_3d, 'line_tau': dic_tau_intersec,
                               'line_int': dic_int_intersec, 'point_tau': dic_tau_line, 'point_int': dic_int_line})
    # ------------------------------------------------------------------
    if plotting is True:
        if ax_tau is None or ax_int is None:
            # raise ValueError('axes of calibration surfaces are required!')
            fig = plt.figure(figsize=(6, 3))
            ax_tau = fig.add_subplot(1, 2, 1, projection='3d')
            ax_int = fig.add_subplot(1, 2, 2, projection='3d')

        # plotting intersecting surfaces
        temp_X, pO2_Y = np.meshgrid(tau_contour.columns, tau_contour.index)

        # lifetime tau0 / tauP
        ax_tau.plot_surface(X=temp_X, Y=pO2_Y, Z=tau_contour * 1e3, rstride=1, cstride=1, color='navy', alpha=.9,
                            linewidth=0., antialiased=True)
        ax_tau.plot_surface(X=temp_X, Y=pO2_Y, Z=dic_tau_3d[1] * 1e3, rstride=1, cstride=1, color='lightgrey',
                            linewidth=0, antialiased=True)
        ax_tau.plot(dic_tau_intersec[1].T.index, dic_tau_intersec[1].T.values,
                    [tauP_meas[1] * 1e3] * len(dic_tau_intersec[1].T.index), lw=2., color='darkorange')

        ax_tau.tick_params(which='both', labelsize=8)
        ax_tau.set_xlabel('Temperature [°C]', fontsize=8)
        ax_tau.set_ylabel('pO2 [hPa]', fontsize=8)
        ax_tau.set_zlabel('tauP [ms]', fontsize=8)

        if pO2_Y.max() > 100:
            pO2_max = 100.
        else:
            pO2_max = pO2_Y.max()*1.05
        if temp_X.max() > 50:
            temp_max = 50.
        else:
            temp_max = temp_X.max()*1.05
        ax_tau.set_ylim(0, pO2_max)
        ax_tau.set_xlim(temp_max, 0)

        # intensity ratio
        ax_int.plot_surface(X=temp_X, Y=pO2_Y, Z=int_contour, rstride=1, cstride=1, color='forestgreen', linewidth=0.,
                            antialiased=True)
        ax_int.plot_surface(X=temp_X, Y=pO2_Y, Z=dic_int_3d[1], rstride=1, cstride=1, color='grey', alpha=0.8,
                            linewidth=0, antialiased=True)
        ax_int.plot(dic_int_intersec[1].T.index, dic_int_intersec[1].T.values,
                    [iratio_meas[1]] * len(dic_int_intersec[1].T.index), lw=2., color='crimson')
        ax_int.tick_params(which='both', labelsize=8, pad=1)
        ax_int.set_xlabel('Temperature [°C]', fontsize=8)
        ax_int.set_ylabel('pO2 [hPa]', fontsize=8)
        ax_int.set_zlabel('I-ratio', fontsize=8)
        ax_int.set_ylim(0, pO2_max)
        ax_int.set_xlim(temp_max, 0)

        fig.subplots_adjust(left=0.05, right=0.94, wspace=0.25)

        # ------------------------------------------------------------------
        # intersection of lines
        fig1, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(dic_tau_line[1].columns, dic_tau_line[1].loc['pO2'].values, color='navy', label='tau intersec')
        ax2.plot(dic_int_line[1].columns, dic_int_line[1].loc['pO2'].values, color='forestgreen', lw=1.,
                 label='Iratio intersec')
        ax2.axvline(x=ls_temp_calc[1], ymin=0, ymax=1, color='crimson', ls='--')
        ax2.axhline(y=ls_pO2_calc[1], xmin=0, xmax=1, color='crimson', ls='--')

        ax2.tick_params(direction='in', right=True, top=True, labelsize=10)
        ax2.set_xlabel('Temperature [°C]', fontsize=8)
        ax2.set_ylabel('$pO_2$ [hPa]', fontsize=8)
        ax2.legend(fontsize=9, frameon=True, fancybox=True, loc=0)
        ax2.set_title('Temp: {:.2f} ± {:.2f} °C - '
                      'pO2: {:.2f} ± {:.2f} hPa'.format(ls_temp_calc[1], np.array(ls_temp_calc).std(), ls_pO2_calc[1],
                      np.array(ls_pO2_calc).std()), fontsize=9)
        plt.tight_layout()

    return ls_temp_calc, ls_pO2_calc, intersec_


def plane_intersection(iratio_calib, iratio_meas, tauP_calib, tauP_meas, deviation):
    # calculated tauP and I-ratio from measurement data
    i_ratio_meas_3d = iratio_calib.copy()
    i_ratio_meas_3d.loc[:, :] = iratio_meas

    tauP_meas_3d = tauP_calib.copy()
    tauP_meas_3d.loc[:, :] = tauP_meas

    # ----------------------------------------------------------------------------
    pO2_inter = np.abs(tauP_calib - tauP_meas).idxmin().values
    d_tau = pd.DataFrame(np.abs(tauP_calib - tauP_meas).min(), columns=['distance'])
    d_tau['pO2'] = pO2_inter

    drop_index_tau = []
    for i, j in zip(d_tau.index, d_tau['pO2']):
        if d_tau.loc[i]['distance'] > deviation / 100.:
            drop_index_tau.append(i)
    d_tau = d_tau.drop(drop_index_tau)

    for t in d_tau.index:
        d_tau['zvalue'] = tauP_calib[t][d_tau.T[t].values[1]]
    # print(d_tau)
    # ----------------------------------------------------------
    pO2_inter_int = np.abs(iratio_calib - iratio_meas).idxmin().values
    d_int = pd.DataFrame(np.abs(iratio_calib - iratio_meas).min(), columns=['distance'])
    d_int['pO2'] = pO2_inter_int

    drop_index = []
    for i, j in zip(d_int.index, d_int['pO2']):
        if d_int.loc[i]['distance'] > deviation / 100.:
            drop_index.append(i)
    d_int = d_int.drop(drop_index)

    for t in d_int.index:
        d_int['zvalue'] = iratio_calib[t][d_int.T[t].values[1]]

    return d_tau, d_int, i_ratio_meas_3d, tauP_meas_3d


def line_intersection(d_int, d_tau, tauP_meas, tauP_calib, i_ratio_meas, iratio_calib):
    x_int_temp = d_int.index
    x_tau_temp = d_tau.index
    #distance_int = d_int['distance'].values
    #distance_tau = d_tau['distance'].values

    y_distance = d_int - d_tau
    y_distance = y_distance['pO2']
    diff_tau = np.abs(tauP_meas - tauP_calib)
    leasq_tau = diff_tau.min().min()

    diff_iratio = np.abs(i_ratio_meas - iratio_calib)
    leasq_iratio = diff_iratio.min().min()

    return leasq_tau, leasq_iratio, diff_tau, diff_iratio, y_distance, d_int, d_tau, x_int_temp, x_tau_temp


def calculating_results(difference_tau, difference_iratio, leasq_tau, leasq_iratio):
    results_calc = pd.DataFrame(np.zeros(shape=(2, 2)), index=['T [°C]', 'pO2 [hPa]'], columns=['tau domain',
                                                                                                'intensity domain'])
    temp_calc = []
    pO2_calc = []
    for temp in difference_tau.columns:
        for pO2 in difference_tau.index:
            if difference_tau[temp][pO2] == leasq_tau:
                temp_calc.append(temp)
                pO2_calc.append(pO2)
                results_calc['tau domain']['T [°C]'] = temp
                results_calc['tau domain']['pO2 [hPa]'] = pO2

    for temp in difference_iratio.columns:
        for pO2 in difference_iratio.index:
            if difference_iratio[temp][pO2] == leasq_iratio:
                temp_calc.append(temp)
                pO2_calc.append(pO2)
                results_calc['intensity domain']['T [°C]'] = temp
                results_calc['intensity domain']['pO2 [hPa]'] = pO2

    return results_calc


# def measurement_evaluation(dphi_f1, dphi_f2, error_meas, ampl_f1, ampl_f2, f1, f2, tauP_3d, iratio_3d, deviation=5.,
#                            zlim_tau=None, zlim_int=None, show_er=False, saving=False, saving_path=None):
#     """
#
#     :param tauP_meas:       np.array including measurement uncertainty in seconds
#     :param iratio_meas:     np.array including measurement uncertainty
#     :param tauP_3d:         calibration plane in seconds
#     :param iratio_3d:       calibration plane as intensity ratio (prompt fluorescence / delayed fluorescence)
#     :param deviation:       relative distance of calculated value (tau or Iratio) and calibration plane
#     :param saving:
#     :param saving_path:
#     :return:
#     """
#     if isinstance(dphi_f1, list):
#         er_ = False
#     else:
#         er_ = True
#     [tauP_meas, Phi_f1_rad_er,
#      Phi_f2_rad_er] = af.phi_to_lifetime_including_error(phi_f1=dphi_f1, phi_f2=dphi_f2, err_phaseangle=error_meas,
#                                                          f1=f1, f2=f2, er=er_)
#
#     dPhi_f1_deg = np.rad2deg(Phi_f1_rad_er)
#     dPhi_f2_deg = np.rad2deg(Phi_f2_rad_er)
#     iratio_meas = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tauP_meas, dphi_f1=dPhi_f1_deg, dphi_f2=dPhi_f2_deg)
#     print('tauP ~ {:.2f}ms'.format(tauP_meas[1]*1e3))
#     print('Iratio calculated ~ {:.2f}'.format(iratio_meas[1]))
#
#     # pre-check if tau  is given in seconds
#     unit = convert_lifetime_unit_into_seconds(tau=tauP_3d)
#     tauP_3d = tauP_3d * 10 ** (switcher[unit])
#     unit_meas = convert_lifetime_unit_into_seconds(tau=tauP_meas)
#     tauP_meas = tauP_meas * 10 ** (switcher[unit_meas])
#
#     leasq_tau = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     leasq_iratio = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     difference_tau = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     difference_iratio = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     tauP_meas_3d = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     distance_tau = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     distance_int = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     iratio_meas_3d = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     y_distance = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     d_int = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     d_tau = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     x_int_temp = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     x_tau_temp = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#
#     for i, meas in enumerate(tauP_meas):
#         # intersection of two planes (calibration vs measurement plane)
#         [distance_tau[i], distance_int[i], iratio_meas_3d[i],
#         tauP_meas_3d[i]] = plane_intersection(iratio_meas=iratio_meas[i], tauP_meas=meas, iratio_calib=iratio_3d,
#                                               tauP_calib=tauP_3d, deviation=deviation)
#
#         # intersection of two lines
#         [leasq_tau[i], leasq_iratio[i], difference_tau[i], difference_iratio[i],
#          y_distance[i], d_int[i], d_tau[i], x_int_temp[i],
#          x_tau_temp[i]] = line_intersection(d_int=distance_int[i], d_tau=distance_tau[i], tauP_meas=tauP_meas_3d[i],
#                                             tauP_calib=tauP_3d, i_ratio_meas=iratio_meas_3d[i], iratio_calib=iratio_3d)
#
#     print('least square or (almost) intersection of planes')
#     print('... in the lifetime domain: {:.2e}'.format(leasq_tau['mean']))
#     print('... in the intensity domain:  {:.2e}'.format(leasq_iratio['mean']))
#
#     # ----------------------------------------------------------------------------------------------------------------
#     # calculation of resulting pO2 and T pair
#     results_calc = pd.Series([None, None, None], index=['min', 'mean', 'max'])
#     for i in results_calc.index:
#         results_calc[i] = calculating_results(difference_tau[i], difference_iratio[i], leasq_tau[i], leasq_iratio[i])
#
#     # ----------------------------------------------------------------------------------------------------------------
#     # Plotting intersection of planes
#     i = 1
#     fig = plotting_intersection_planes(tauP_meas=tauP_meas_3d[i], tauP_calib=tauP_3d, iratio_calib=iratio_3d,
#                                        distance_tau=distance_tau[i], i_ratio_meas=iratio_meas_3d[i], run=0,
#                                        distance_int=distance_int[i], zlim_tau=zlim_tau, zlim_int=zlim_int)
#
#     fig_ = None
#     fig1 = plotting_intersection_lines(results_calc=results_calc[i], distance_int=distance_int[i],  d_tau=d_tau[i],
#                                        distance_tau=distance_tau[i], x_int_temp=x_int_temp[i], x_tau_temp=x_tau_temp[i],
#                                        d_int=d_int[i], fig_=fig_)
#
#     # ----------------------------------------------------------------------------------------------------------------
#     # combining input parameter for output
#     if show_er is False:
#         input = pd.DataFrame(np.zeros(shape=(7, 2)), columns=['value', 'unit'],
#                              index=['dPhi(f1)', 'dPhi(f2)', 'A(f1)', 'A(f2)', 'tauP', 'I-ratio', 'allwoed deviation'])
#
#         input.loc['tauP'] = [tauP_meas[1]*1e3, 'ms']
#         input.loc['I-ratio'] = [iratio_meas[1], '']
#         input.loc['dPhi(f1)'] = [dPhi_f1_deg[1], 'deg']
#         input.loc['dPhi(f2)'] = [dPhi_f2_deg[1], 'deg']
#         input.loc['A(f1)'] = [ampl_f1, 'mV']
#         input.loc['A(f2)'] = [ampl_f2, 'mV']
#         input.loc['allwoed deviation'] = [deviation, '%']
#         results_calc['mean'].loc['results'] = results_calc['mean'].columns
#         results_calc['mean'].columns = input.columns
#         results_calc['mean'] = results_calc['mean'].reindex(['results', 'T [°C]', 'pO2 [hPa]'])
#         output = pd.concat([input, results_calc['mean']], axis=0)
#     else:
#         input = pd.DataFrame(np.zeros(shape=(7, 3)), columns=['value', 'std', 'unit'],
#                              index=['dPhi(f1)', 'dPhi(f2)', 'A(f1)', 'A(f2)', 'tauP', 'I-ratio', 'allwoed deviation'])
#
#         input.loc['tauP'] = [tauP_meas[1], np.array(tauP_meas).std(), 'ms']
#         input.loc['I-ratio'] = [iratio_meas[1], np.array(iratio_meas).std(), '']
#         input.loc['dPhi(f1)'] = [dPhi_f1_deg[1], np.array(dPhi_f1_deg).std(), 'deg']
#         input.loc['dPhi(f2)'] = [dPhi_f2_deg[1], np.array(dPhi_f2_deg).std(), 'deg']
#         input.loc['A(f1)'] = [ampl_f1, 'mV', '']
#         input.loc['A(f2)'] = [ampl_f2, 'mV', '']
#         input.loc['allwoed deviation'] = [deviation, '%', '']
#
#         results_calc_ = pd.concat([results_calc['min'], results_calc['mean'], results_calc['max']])
#         results_calc_.index = ['T [°C] min', 'pO2 [hPa] min', 'T [°C] mean', 'pO2 [hPa] mean', 'T [°C] max', 'pO2 [hPa] max']
#         results_calc_.loc['results'] = results_calc['mean'].columns
#         results_calc_.columns = ['value', 'std']
#         results_calc_ = results_calc_.reindex(['results', 'T [°C] min', 'T [°C] mean', 'T [°C] max', 'pO2 [hPa] min',
#                                                'pO2 [hPa] mean', 'pO2 [hPa] max'])
#
#         output = pd.concat([input, results_calc_], axis=0)
#
#     # ----------------------------------------------------------------------------------------------------------------
#     if saving is True:
#         time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
#         path_folder = saving_path
#         if saving_path is None:
#             path_folder = 'Report/'
#         if 'Report' in path_folder:
#             pass
#         else:
#             path_folder += '/Report/'
#         if not os.path.exists(path_folder):
#             os.makedirs(path_folder)
#
#         # saving names for figures and txt files
#         sav_inter_plane = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
#                           '_intersection_planes.png'
#         sav_inter_line = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
#                          '_intersection_lines.png'
#         if show_er is True:
#             sav_results = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
#                               '_intersection_measurement_err-incl.txt'
#         else:
#             sav_results = path_folder + time_now.split(' ')[0] + '_' + time_now.split(' ')[1][:-3].replace(':', '') + \
#                           '_intersection_measurement.txt'
#
#         fig.savefig(sav_inter_plane)
#         fig1.savefig(sav_inter_line)
#         output.to_csv(sav_results, sep='\t', decimal='.')
#
#     return output, tauP_meas, iratio_meas


def rearange_dataframe_and_averageing_ranges(df, ls_para1, decimal_para1, ls_para2, decimal_para2, para1='Temperature',
                                             para2='pO2.mean', para_df='tauP [ms]'):
    # round temperature and oxygen for plateaus
    temp_ = [round(t, int(decimal_para1)) for t in ls_para1]
    columns_ = sorted(pd.DataFrame(temp_).drop_duplicates()[0].values.tolist())

    # o2 = df['pO2.mean'].drop_duplicates().sort_values().values.tolist()
    # ox_ = [round(i,0) for i in o2]
    o2 = [round(t, int(decimal_para2)) for t in ls_para2]
    index_ = sorted(pd.DataFrame(o2).drop_duplicates()[0].values.tolist())

    # generate dataframe for dual sensing
    df_3d = pd.DataFrame(np.nan, index=index_, columns=columns_)

    # slit width
    deltaT = np.abs(columns_[1] - columns_[0]) / 2
    delta_ox = np.abs(index_[1] - index_[0]) / 2

    # fill dataframe
    for c in columns_:
        r = df[df[para1] <= c + deltaT]
        T_sliced = r[r[para1] >= c - deltaT]

        for n in index_:
            Tox = T_sliced[T_sliced[para2] <= n + delta_ox]
            df_3d.loc[n, c] = Tox[Tox[para2] >= n - delta_ox].mean()[para_df]

    return df_3d


def find_limits_plane_intersection(T, diff, min_direction1):
    """

    :param T:                   temperature T
    :param diff:                dataframe with T in column
    :param min_direction1:
    :return:
    """
    dist5, dist6, ox5, ox6 = af.find_closest_in_list_(list_=diff[T], value=min_direction1[T])

    if ox5 == ox6:
        y2 = diff.drop(ox5)
    else:
        if dist5 > dist6:
            y2 = diff.drop(ox5)
        else:
            y2 = diff.drop(ox6)
    dist7, dist8, ox7, ox8 = af.find_closest_in_list_(list_=y2[T], value=min_direction1[T])

    ox_to_temp = pd.DataFrame([dist5, dist6, dist7, dist8], [ox5, ox6, ox7, ox8]).dropna().drop_duplicates()
    ox_to_temp.columns = [T]

    return ox_to_temp


# --------------------------------------------------------------------------------------------------------------------
def reference_sensor(file, day, plotting=True, fontsize_=12):
    startkey = day.strftime("%d.%m.%Y")

    lines = []
    i = 0
    with open(file, mode='r') as f:
        for line in f:
            if startkey in line:
                for line in f:
                    lines.append(line.replace(',', '.').split('\t'))
            else:
                header_lines = i
                i += 1

    df = pd.DataFrame(lines[2:])
    header = ['Date', 'Time (HH:MM:SS)', 'Time (s)', 'O2 Ch1', 'Comp-temp Ch1', 'Temp. Probe', 'dphi1 raw', 'int1',
              'ambient1']

    ddf = pd.concat([df.iloc[:, :3], df.iloc[:, 4], df.iloc[:, 8], df.iloc[:, 14], df.iloc[:, 17], df.iloc[:, 21],
                     df.iloc[:, 25]], axis=1)
    ddf.columns = header
    ddf = ddf.convert_objects(convert_numeric=True)

    time_firesting = []
    for i in range(len(ddf['Time (HH:MM:SS)'])):
        t = ddf['Date'][i] + ' ' + ddf['Time (HH:MM:SS)'][i]
        time_firesting.append(datetime.datetime.strptime(t, "%d.%m.%Y %H:%M:%S"))
    ddf_output = ddf[ddf.columns[2:]]
    ddf_output.index = time_firesting

    # -----------------------------------------------------------------------------------------------
    # PLOTTING
    if plotting is True:
        f, (ax, ax1) = plt.subplots(figsize=(4.5, 4), nrows=2, sharex=True)

        ax_temp = ax.twinx()
        ax.plot(time_firesting, ddf['dphi1 raw'], color='crimson')
        ax_temp.plot(time_firesting, ddf['Temp. Probe'], lw=1., color='navy')
        ax1.plot(time_firesting, ddf['int1'], color='k')

        ax.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)
        ax_temp.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)
        ax1.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True, right=True)

        ax.grid(axis='y', lw=0.2, color='k', alpha=0.8)
        ax1.grid(axis='y', lw=0.2, color='k', alpha=0.8)

        ax.set_ylabel('dPhi [deg]', fontsize=fontsize_, color='crimson')
        ax_temp.set_ylabel('Temp [°C]', fontsize=fontsize_, color='navy')
        ax1.set_ylabel('Intensity [mV]', fontsize=fontsize_)

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax1.xaxis.set_major_formatter(myFmt)
        f.autofmt_xdate(ha='center')

        plt.tight_layout(pad=0.5)
        plt.show()

    return ddf_output, header_lines


def characterisation_sensors(timerange_2pO2, timerange_0pO2, df_dphi, fmod_fs, m_fs, f_fs, pO2_2pc, plotting=True,
                             fontsize_=13):
    # plateau extraction
    fs_2pc, fs_0pc = plateau_reference(timerange_2pO2=timerange_2pO2, timerange_0pO2=timerange_0pO2, df_dphi=df_dphi)

    # dpi to lifetime (single frequency measurement)
    dphi_2pc = [np.abs(fs_2pc['dphi1 raw']).mean() - np.abs(fs_2pc['dphi1 raw']).std(),
                np.abs(fs_2pc['dphi1 raw']).mean(), np.abs(fs_2pc['dphi1 raw']).mean() +
                np.abs(fs_2pc['dphi1 raw']).std()]
    dphi_0pc = [np.abs(fs_0pc['dphi1 raw']).mean() - np.abs(fs_0pc['dphi1 raw']).std(),
                np.abs(fs_0pc['dphi1 raw']).mean(), np.abs(fs_0pc['dphi1 raw']).mean() +
                np.abs(fs_0pc['dphi1 raw']).std()]

    tau2 = np.tan(np.deg2rad(dphi_2pc)) / (2 * np.pi * fmod_fs)
    tau0 = np.tan(np.deg2rad(dphi_0pc)) / (2 * np.pi * fmod_fs)
    print('============================================')
    print('Calibration reference sensor')
    print('\t tau0 (0%pO2): {:.3f} +/- {:.3f} µs'.format(tau0[1] * 1e6, (tau0[2] - tau0[0]) / 2 * 1e6))
    print('\t tau1 (2%pO2): {:.3f} +/- {:.3f} µs'.format(tau2[1] * 1e6, (tau2[2] - tau2[0]) / 2 * 1e6))

    # Two site model fit
    para_TSM = twoSiteModel_calib_ksv(tau0=tau0, tau1=tau2, m=m_fs, f=f_fs, pO2_calib1=pO2_2pc)
    print('\t Ksv1 fit: ', para_TSM['Ksv_fit1'][1].round(3))
    print()

    if plotting is True:
        pO2_percent = np.linspace(start=0, stop=25, num=100)
        tau_tau0_quot = [para_TSM['slope'] / (1 + k1 * pO2_percent) + (1 - para_TSM['slope']) / (1 + k2 * pO2_percent)
                         for (k1, k2) in zip(para_TSM['Ksv_fit1'], para_TSM['Ksv_fit2'])]

        df_tau_tau0_quot = pd.DataFrame(tau_tau0_quot, columns=pO2_percent, index=['min', 'mean', 'max']).T

        # ---------------------------------------------------------------------------
        f, ax = plt.subplots(figsize=(4, 3))

        ax.plot(1 / df_tau_tau0_quot['mean'], color='navy', lw=1.)
        ax.plot([0, 18.942], [1, 4.76], marker='s', color='navy', lw=0.)
        plt.title('Reference calibration')
        plt.suptitle('Ksv1 = {:.2f}, f = {:.2f}and m = {:.3f}'.format(para_TSM['Ksv_fit1'][1],para_TSM['slope'],
                                                                      para_TSM['prop Ksv']),fontsize=fontsize_ * 0.9)
        # bbox_to_anchor=(0.5, 1.25),ncol=2, loc='upper center',

        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.set_xlim(-2, 25.5)
        ax.set_ylim(0, 6)

        ax.tick_params(axis='both', direction='in', which='both', right=True, top=True, labelsize=fontsize_)
        ax.set_xlabel('pO2 (hPa)', fontsize=fontsize_)
        ax.set_ylabel('tau0/tau', fontsize=fontsize_)
        ax.grid(axis='y')

        plt.tight_layout()

    return fs_2pc, fs_0pc, para_TSM


def plateau_reference(timerange_2pO2, timerange_0pO2, df_dphi):
    # prepare timeranges
    if timerange_2pO2 is None:
        raise ValueError('Start stop time required')
    else:
        if isinstance(timerange_2pO2, str):
            if len(timerange_2pO2) <= 1:
                raise ValueError('Start stop time required')
            else:
                timerange_2pO2 = datetime.datetime.strptime(timerange_2pO2.split(',')[0], "%H:%M:%S"),\
                                 datetime.datetime.strptime(timerange_2pO2.split(',')[1][1:], "%H:%M:%S")
    if timerange_0pO2 is None:
        raise ValueError('Start stop time required')
    else:
        if isinstance(timerange_0pO2, str):
            if len(timerange_0pO2) <= 1:
                raise ValueError('Start stop time required')
            else:
                timerange_0pO2 = datetime.datetime.strptime(timerange_0pO2.split(',')[0], "%H:%M:%S"),\
                                 datetime.datetime.strptime(timerange_0pO2.split(',')[1][1:], "%H:%M:%S"),

    # ------------------------------------------------------------------
    # averaging dataframe in order to extract dphi and amplitude for 0% and 1.98% pO2
    time_stemp_0pc = []
    time_stemp_2pc = []

    for n, i in enumerate(df_dphi.index):
        if timerange_0pO2[1].strftime('%H:%M:%S') >= i.strftime('%H:%M:%S')>= timerange_0pO2[0].strftime('%H:%M:%S'):
            time_stemp_0pc.append(i)
        if timerange_2pO2[0].strftime('%H:%M:%S') <= i.strftime('%H:%M:%S') <= timerange_2pO2[1].strftime('%H:%M:%S'):
            time_stemp_2pc.append(i)
    fs_2pc = df_dphi.loc[time_stemp_2pc[0]:time_stemp_2pc[-1]]
    fs_0pc = df_dphi.loc[time_stemp_0pc[0]:time_stemp_0pc[-1]]

    print('## 0% pO2 Plateau')
    print('[Reference]')
    print('\t dphi raw {:.3f} +/- {:.3f} deg'.format(np.abs(fs_0pc['dphi1 raw']).mean(), np.abs(fs_0pc['dphi1 raw']).std()))
    print('\t Intensity {:.2f} +/- {:.2f} mV'.format(fs_0pc['int1'].mean(), fs_0pc['int1'].std()))
    print('\t Temperature {:.2f} +/- {:.2f} °C'.format(fs_0pc['Temp. Probe'].mean(), fs_0pc['Temp. Probe'].std()))
    print('[Dualsensor]')
    for i, c in enumerate(fs_0pc.columns.tolist()):
        if 'dPhi(' in c:
            print('\t dphi(f{}) {:.3f} +/- {:.3f} deg'.format(i+1, np.abs(fs_0pc['dPhi(f{}) [deg]'.format(i+1)]).mean(),
                                                              np.abs(fs_0pc['dPhi(f{}) [deg]'.format(i+1)]).std()))
            print('\t A(f{}) {:.3f} +/- {:.3f} mV'.format(i+1, np.abs(fs_0pc['A(f{}) [mV]'.format(i+1)]).mean(),
                                                              np.abs(fs_0pc['A(f{}) [mV]'.format(i+1)]).std()))

    print('--------------------------------------------------')
    print('## 1.98% pO2 Plateau')
    print('[Reference]')
    print('\t dphi raw {:.3f} +/- {:.3f} deg'.format(np.abs(fs_2pc['dphi1 raw']).mean(), np.abs(fs_2pc['dphi1 raw']).std()))
    print('\t Intensity {:.2f} +/- {:.2f} mV'.format(fs_2pc['int1'].mean(), fs_2pc['int1'].std()))
    print('\t Temperature {:.2f} +/- {:.2f} °C'.format(fs_2pc['Temp. Probe'].mean(), fs_2pc['Temp. Probe'].std()))
    print('[Dualsensor]')
    for i, c in enumerate(fs_0pc.columns.tolist()):
        if 'dPhi(' in c:
            print('\t dphi(f{}) {:.3f} +/- {:.3f} deg'.format(i+1, np.abs(fs_2pc['dPhi(f{}) [deg]'.format(i+1)]).mean(),
                                                              np.abs(fs_2pc['dPhi(f{}) [deg]'.format(i+1)]).std()))
            print('\t A(f{}) {:.3f} +/- {:.3f} mV'.format(i+1, np.abs(fs_2pc['A(f{}) [mV]'.format(i+1)]).mean(),
                                                              np.abs(fs_2pc['A(f{}) [mV]'.format(i+1)]).std()))
    print()

    return fs_2pc, fs_0pc


def dualsensor_measruement(frequencies, ddphi_f_, time_d, lock_raw_):
    f, ddphi_f, time_step = preparation_input(f_=frequencies, ddphi_f1=ddphi_f_, time_d=time_d)

    # --------------------------------------------------------------------------------
    # time_delta = 15. (?) # measurement time + waiting time + repetition (averaging)
    for i, v in enumerate(lock_raw_[0]):
        if lock_raw_[0][i + 2] - lock_raw_[0][i] > time_step - 1:
            time_start = i + 1
            break

    lock_raw = lock_raw_.loc[time_start:]
    lock_raw.index = np.arange(len(lock_raw.index))

    # ----------------------------------------------------------------------
    # split dataframe into f1 and f2 - dataframe
    df_f1 = pd.DataFrame(np.zeros(shape=(int(len(lock_raw) / 2) + 1, 3)))
    df_f2 = pd.DataFrame(np.zeros(shape=(int(len(lock_raw) / 2) + 1, 3)))
    n = 0
    m = 0
    for i in lock_raw.index:
        if (i % 2) == 0:
            df_f1.iloc[n, :] = lock_raw.iloc[i, :]
            n += 1
        else:
            df_f2.iloc[m, :] = lock_raw.iloc[i, :]
            m += 1

    df_f1.columns = ['Time (s)', 'A (mV)', 'dPhi (deg)']
    df_f2.columns = ['Time (s)', 'A (mV)', 'dPhi (deg)']

    # ----------------------------------------------------------------------
    # Time reference to f1
    amplitudes = pd.concat([df_f1[df_f1.columns[1]], df_f2[df_f2.columns[1]]], axis=1)
    amplitudes.columns = ['f1', 'f2']
    amplitudes_lockin = amplitudes.iloc[:-1]

    print('corrected phaseangle: ', ddphi_f1, ddphi_f2)

    dphi = pd.concat([np.abs(df_f1[df_f1.columns[2]] + ddphi_f1),
                      np.abs(df_f2[df_f2.columns[2]] + ddphi_f2)], axis=1)
    dphi.columns = ['f1', 'f2']
    dphi_lockin = dphi.iloc[:-1]

    return dphi_lockin, amplitudes_lockin, df_f1, df_f2, f1, f2


def synchronize_sensors(file_firesting, file_lockIn, day, f1_, f2_, ddphi_f1_, ddphi_f2_, time_d, lock_raw_,
                        plotting=True, smooth=True, mode='mirror', fontsize_=9.5):
    """
    if smooth is True - select mode of smoothing (‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’)
    :param file_firesting:
    :param file_lockIn:
    :param day:
    :param f1_:
    :param f2_:
    :param ddphi_f1_:
    :param ddphi_f2_:
    :param time_d:
    :param lock_raw_:
    :param plotting:
    :param smooth:
    :return:
    """
    # firesintg preparation
    df_firesting, header_lines = reference_sensor(file=file_firesting, day=day, plotting=plotting)

    # lockIn amplifier preparation
    [dphi_lockin, amplitudes_lockin, df_f1, df_f2, f1,
     f2] = dualsensor_measruement(f1_=f1_, f2_=f2_, ddphi_f1_=ddphi_f1_, ddphi_f2_=ddphi_f2_, time_d=time_d,
                                  lock_raw_=lock_raw_)

    time_fs = pd.read_csv(file_firesting, encoding='latin-1', skiprows=header_lines + 4, header=None, sep='\t',
                          usecols=[0, 1], skipinitialspace=True)
    start_time_fs = datetime.datetime.strptime(' '.join(time_fs.values[0]), "%d.%m.%Y %H:%M:%S")
    stop_time_fs = datetime.datetime.strptime(' '.join(time_fs.values[-1]), "%d.%m.%Y %H:%M:%S")

    # time extraction from firesting
    if start_time_fs.strftime("%d-%m-%Y") == stop_time_fs.strftime("%d-%m-%Y"):
        print('day of measurement:', start_time_fs.strftime("%d-%m-%Y"))
    else:
        print('days of measurement:', start_time_fs.strftime("%d.%m"), '-', stop_time_fs.strftime("%d.%m.%Y"))
    print('--------------------------------------------------\n')

    # time extraction LockIn
    stop_time_lockin = datetime.datetime.strptime(' '.join(file_lockIn.split('/')[-1].split('_')[0].split('-')),
                                                  "%Y%m%d %H%M%S")
    start_time_lockin = stop_time_lockin - datetime.timedelta(seconds=df_f1['Time (s)'].values[-2])

    print('start of measurement LockIn:', start_time_lockin.strftime("%d.%m.%Y %H:%M:%S"))
    print('end of measurement LockIn:', stop_time_lockin.strftime("%d.%m.%Y %H:%M:%S"))
    print('--------------------------------------')

    time_ls_full_f1 = []
    time_ls_f1 = []
    for i in df_f1['Time (s)'].iloc[:-1]:
        t_1 = start_time_lockin + datetime.timedelta(seconds=i)
        t1 = t_1 - datetime.timedelta(microseconds=t_1.microsecond)
        time_ls_full_f1.append(t1)
        time_ls_f1.append(t1.strftime('%d.%m.%Y %H:%M:%S'))

    amplitudes_lockin.index = time_ls_full_f1
    dphi_lockin.index = time_ls_full_f1

    # ==========================================================================================
    # FIRESTING
    print('start of measurement firesting:', start_time_fs.strftime("%d.%m.%Y %H:%M:%S"))
    print('end of measurement firesting:', stop_time_fs.strftime("%d.%m.%Y %H:%M:%S"))

    #time_fs = []
    #time_fs2 = []
    #for i, c in enumerate(df_firesting.index): #df_firesting['Time (HH:MM:SS)'].values
    #    d = ' '.join([df_firesting['Date'].values[i], c])
    #    t = datetime.datetime.strptime(d, "%d.%m.%Y %H:%M:%S")
    #    time_fs2.append(t.strftime("%d.%m.%Y %H:%M:%S"))
    #    time_fs2.append(t.strftime("%d.%m.%Y %H:%M:%S"))
    #    time_fs.append(t)

    ddf_slice = df_firesting[['dphi1 raw', 'int1', 'Temp. Probe']]
    ddf_slice.index = time_fs
    ddf_firesting = ddf_slice

    dphi_total = pd.merge(ddf_firesting, dphi_lockin, how='inner', left_index=True, right_index=True)
    ampl_total = pd.merge(ddf_firesting, amplitudes_lockin, how='inner', left_index=True, right_index=True)

    if smooth is True:
        y1_smooth = savgol_filter(dphi_lockin['f1'].values, 501, 2, mode=mode)
        y2_smooth = savgol_filter(dphi_lockin['f2'].values, 501, 2, mode=mode)
        dphi_ = pd.concat([pd.DataFrame(y1_smooth, index=dphi_lockin.index),
                           pd.DataFrame(y2_smooth, index=dphi_lockin.index)], axis=1, sort=False)
        dphi_.columns = ['dPhi(f1)', 'dPhi(f2)']
        dphi = pd.merge(ddf_firesting, dphi_, how='inner', left_index=True, right_index=True)
        print('1491', dphi)
    else:
        dphi = dphi_total

    if plotting is True:
        f_dphi, (ax_dphi, ax1_dphi) = plt.subplots(figsize=(5, 4), nrows=2, sharex=True)

        ax_temp = ax_dphi.twinx()
        ax_dphi.plot(time_fs, ddf_slice['dphi1 raw'], color='k')
        ax_temp.plot(time_fs, ddf_slice['Temp. Probe'], color='forestgreen', lw=.75)
        ax1_dphi.plot(dphi_total['f1'].dropna(), color='#66a5ad', lw=1., ls='--')
        ax1_dphi.plot(dphi_total['f2'].dropna(), color='navy', lw=1., ls='--')

        if smooth is True:
            ax1_dphi.plot(dphi['dPhi(f1)'].dropna(), color='darkorange', lw=1., ls='--', label='f1 fit')
            ax1_dphi.plot(dphi['dPhi(f2)'].dropna(), color='crimson', lw=1., ls='--', label='f2 fit')
            ax1_dphi.legend(['data {:.1f}Hz'.format(f1), 'data {:.1f}Hz'.format(f2), 'fit {:.1f}Hz'.format(f1),
                             'fit {:.1f}Hz'.format(f2)], fontsize=fontsize_ * 0.8)
        else:
            ax1_dphi.legend(['{:.1f}Hz'.format(f1), '{:.1f}Hz'.format(f2)], fontsize=fontsize_ * 0.8)

        ax_dphi.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)
        ax_temp.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in')
        ax1_dphi.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax_dphi.xaxis.set_major_formatter(myFmt)
        ax_temp.set_ylabel('Temperature$_{firesting}$ [°C]', fontsize=fontsize_, color='forestgreen')
        ax_dphi.set_ylabel('dPhi$_{firesting}$ [deg]', fontsize=fontsize_)
        ax1_dphi.set_ylabel('dPhi$_{lockIn}$ f1 [deg]', fontsize=fontsize_)
        f_dphi.autofmt_xdate(ha='center')

        plt.tight_layout(pad=0.75)
        plt.show()

        # ===================================================================
        # amplitudes
        f_ampl, (ax, ax1) = plt.subplots(figsize=(5, 4), nrows=2, sharex=True)

        ax.plot(dphi_total['int1'].dropna(), color='k')
        ax1.plot(ampl_total['A(f1) [mV]'].dropna(), color='#66a5ad', lw=1.)
        ax1.plot(ampl_total['A(f2) [mV]'].dropna(), color='navy', lw=1.)

        ax1.legend(['{:.1f}Hz'.format(f1), '{:.1f}Hz'.format(f2)], fontsize=9 * 0.8)
        if np.isnan(ampl_total['A(f1) [mV]'].loc[ampl_total.index[-1]]) == True:
            ymax1 = ampl_total['A(f1) [mV]'].loc[ampl_total.index[-2]]
        else:
            ymax1 = ampl_total['A(f1) [mV]'].loc[ampl_total.index[-1]]
        if np.isnan(ampl_total['A(f2) [mV]'].loc[ampl_total.index[-1]]) == True:
            ymax2 = ampl_total['A(f2) [mV]'].loc[ampl_total.index[-2]]
        else:
            ymax2 = ampl_total['A(f2) [mV]'].loc[ampl_total.index[-1]]

        if ymax1 < ymax2:
            ymax = ymax2 * 1.05
        else:
            ymax = ymax1 * 1.05
        ax1.set_ylim(0.5, ymax)
        ax.tick_params(which='both', labelsize=9 * 0.9, direction='in', top=True)
        ax1.tick_params(which='both', labelsize=9 * 0.9, direction='in')

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(myFmt)
        ax.set_ylabel('A$_{firesting}$ [mV]', fontsize=9)
        ax1.set_ylabel('A$_{lockIn}$ f1 [mV]', fontsize=9)
        f_ampl.autofmt_xdate(ha='center')

        plt.tight_layout(pad=0.75)
        plt.show()

    return dphi, ampl_total, f1, f2


def dualsensor_evalaution_plateau(fs_2pc, fs_0pc, freq_Hz, freq_pair=None):
    # preparation and frequency selection
    if freq_pair is None:
        freq_pair = [1, 2]
    print('Frequencies for evaluation f1={:.0f}Hz, f2={:.0f}Hz'.format(freq_Hz[freq_pair[0] - 1],
                                                                       freq_Hz[freq_pair[1] - 1]))
    print()
    f1 = freq_Hz[freq_pair[0] - 1]
    f2 = freq_Hz[freq_pair[1] - 1]

    column_add = []
    for i, c in enumerate(fs_2pc.columns.tolist()):
        if 'dPhi' in c:
            pass
        else:
            column_add.append(i)

    dphi_columns = ['dPhi(f{}) [deg]'.format(freq_pair[0]), 'dPhi(f{}) [deg]'.format(freq_pair[1])]
    fs_2pc = fs_2pc[dphi_columns + list(fs_2pc.columns[column_add])]
    fs_0pc = fs_0pc[dphi_columns + list(fs_0pc.columns[column_add])]

    ls_ = fs_2pc[dphi_columns].std().values.tolist() + \
          fs_0pc[dphi_columns].std().values.tolist()
    std_ = pd.DataFrame(ls_)
    z = np.abs(stats.zscore(std_))
    threshold = 1.
    outlier = []
    outlier.append(np.where(z > threshold)[0][0])
    std_new = std_.drop(outlier)
    err = std_new.mean().values[0]

    # -------------------------------------------------------------------
    # Phaseangle at 0% and 1.98% pO2 including average error
    phi0_f1 = [fs_0pc[dphi_columns[0]].mean()-err, fs_0pc[dphi_columns[0]].mean(), fs_0pc[dphi_columns[0]].mean()+err]
    phi0_f2 = [fs_0pc[dphi_columns[1]].mean()-err, fs_0pc[dphi_columns[1]].mean(), fs_0pc[dphi_columns[1]].mean()+err]
    phi2_f1 = [fs_2pc[dphi_columns[0]].mean()-err, fs_2pc[dphi_columns[0]].mean(), fs_2pc[dphi_columns[0]].mean()+err]
    phi2_f2 = [fs_2pc[dphi_columns[1]].mean()-err, fs_2pc[dphi_columns[1]].mean(), fs_2pc[dphi_columns[1]].mean()+err]

    # -------------------------------------------------------------------------
    # Calculate lifetime by 2-frequency analysis
    tau0_1, tau0_2 = af.two_frequency_lifetime(f1=f1, f2=f2, Phi_f1_rad=np.deg2rad(phi0_f1),
                                               Phi_f2_rad=np.deg2rad(phi0_f2))
    tau0_ = af.tau_selection(tau0_1, tau0_2)
    tau0 = sorted(tau0_)
    print('Characterization dualsensor')
    print('\t Lifetime 0%pO2 {:.2f} +/- {:.2f} ms'.format(tau0[1]*1e3, (tau0[0] - tau0[2])/2*1e3))
    tau2_1, tau2_2 = af.two_frequency_lifetime(f1=f1, f2=f2, Phi_f1_rad=np.deg2rad(phi2_f1),
                                               Phi_f2_rad=np.deg2rad(phi2_f2))
    tau2 = af.tau_selection(tau2_1, tau2_2)
    tau2 = sorted(tau2)
    print('\t Lifetime 1.98% pO2 {:.2f} +/- {:.2f} ms'.format(tau2[1]*1e3, (tau2[0] - tau2[2])/2*1e3))
    print()
    # -------------------------------------------------------------------------
    # Calculate intensity ratio
    int_ratio0 = af.intensity_ratio(f=f1, tau=tau0, phi=phi0_f1)
    int_ratio2 = af.intensity_ratio(f=f1, tau=tau2, phi=phi2_f1)

    print('\t I-ratio (pF/dF) 0%pO2: {:.2f} +/- {:.2f}'.format(int_ratio0[1], (int_ratio0[0] - int_ratio0[2])/2))
    print('\t I-ratio (pF/dF) 1.98%pO2: {:.2f} +/- {:.2f}'.format(int_ratio2[1], (int_ratio2[0] - int_ratio2[2])/2))

    return err, tau0, tau2, int_ratio0, int_ratio2


def combining_data(df_dphi, meas_range, err, f1, f2, fmod_fs, para_TSM):
    if len(meas_range) <= 1:
        measurement_range = df_dphi.index[0], df_dphi.index[-1]
    else:
        measurement_range = datetime.datetime.strptime(meas_range.split(',')[0], "%H:%M:%S"),\
                            datetime.datetime.strptime(meas_range.split(',')[1].split()[0], "%H:%M:%S")

    m_stemp = []
    for n, i in enumerate(df_dphi.index):
        if measurement_range[1].strftime('%H:%M:%S') >= i.strftime('%H:%M:%S')>= measurement_range[0].strftime('%H:%M:%S'):
            m_stemp.append(i)
    df_dphi_sliced = df_dphi.loc[m_stemp[0]:m_stemp[-1]]

    # -----------------------------------------------------------------
    tauP1 = []
    tauP2 = []
    for i, v in enumerate(df_dphi_sliced.index):
        if isinstance(df_dphi_sliced['dPhi(f1)'].loc[df_dphi_sliced.index[i]], np.float):
            p1 = np.deg2rad(df_dphi_sliced['dPhi(f1)'].loc[df_dphi_sliced.index[i]])
        else:
            p1 = np.deg2rad(df_dphi_sliced['dPhi(f1)'].loc[df_dphi_sliced.index[i]].mean())
        if isinstance(df_dphi_sliced['dPhi(f2)'].loc[df_dphi_sliced.index[i]], np.float):
            p2 = np.deg2rad(df_dphi_sliced['dPhi(f2)'].loc[df_dphi_sliced.index[i]])
        else:
            p2 = np.deg2rad(df_dphi_sliced['dPhi(f2)'].loc[df_dphi_sliced.index[i]].mean())

        phi1 = [p1-err, p1, p1+err]
        phi2 = [p2-err, p2, p2+err]

        tau_1, tau_2 = af.two_frequency_lifetime(f1=f1, f2=f2, Phi_f1_rad=phi1, Phi_f2_rad=phi2)
        tauP1.append(tau_1)
        tauP2.append(tau_2)

    df_tau1 = pd.DataFrame(tauP1)
    df_tau2 = pd.DataFrame(tauP2)

    # !!! TODO: select tau_dual min and max ranges
    tauP = []
    #tau_min = tau_selection(df_tau1[0], df_tau2[2])
    tau_mean = af.tau_selection(df_tau1[1], df_tau2[1])
    #tau_max = tau_selection(df_tau1[2], df_tau2[2])

    tau_mean.index=df_dphi_sliced.index
    df_tau = pd.DataFrame(tau_mean).dropna()
    df_tau.columns = ['tau_dual.mean [s]']

    # ====================================================================================
    # calculate pO2 from reference
    lifetime_ref = pd.DataFrame(np.tan(np.deg2rad(df_dphi_sliced['dphi1 raw'])) / (2*np.pi*fmod_fs))
    lifetime_ref.columns = ['tau_ref [s]']

    pO2_1 = []
    for i in lifetime_ref['tau_ref [s]']:
        x_1, x_2 = twoSiteModel_calculation(tau0=para_TSM['tauP0'], tau=i, ksv=para_TSM['Ksv_fit1'],
                                            f=para_TSM['slope'], m=para_TSM['prop Ksv'])
        pO2_1.append(x_1)
    pO2_ref = pd.DataFrame(pO2_1, index=lifetime_ref.index, columns=['pO2.min', 'pO2.mean', 'pO2.max'])

    # time vs tau0 / tau
    l = [t0 / lifetime_ref['tau_ref [s]'] for t0 in para_TSM['tauP0']]
    tau0_tau_quot = pd.concat([l[0], l[1], l[2]], axis=1)
    tau0_tau_quot.columns = ['tau0/tau_ref.min', 'tau0/tau_ref.mean', 'tau0/tau_ref.max']

    # measurement time vs lifetime and pO2
    df_phi_pO2 = pd.merge(df_dphi, pO2_ref, how='inner', left_index=True, right_index=True) #[['dphi1 raw', 'int1', 'Temp. Probe']]
    reference_phi_pO2 = pd.merge(df_phi_pO2, lifetime_ref, how='inner', left_index=True, right_index=True)
    reference_phi_pO2_quot = pd.merge(reference_phi_pO2, tau0_tau_quot, how='inner', left_index=True, right_index=True)

    # combining relevant data
    reference_sensor = reference_phi_pO2_quot[['pO2.min', 'pO2.mean', 'pO2.max', 'tau_ref [s]', 'tau0/tau_ref.min',
                                               'tau0/tau_ref.mean', 'tau0/tau_ref.max']]
    data_combined = pd.merge(reference_sensor, df_tau, how='inner', left_index=True, right_index=True)
    data_combined_all = pd.merge(reference_phi_pO2_quot, df_tau, how='inner', left_index=True, right_index=True)
    data_pO2_sorted = data_combined.sort_values(by='pO2.mean')
    data_pO2_all = data_combined_all.sort_values(by='pO2.mean')

    return data_pO2_sorted, data_pO2_all


def dualsensor_evaluation(data_all, reg_range, f1, regression=True, plotting=True, fontsize_=9.5):
    # Lifetime
    # for regression: select range 0hPa:12hPa
    data_all[['pO2.min', 'pO2.mean', 'pO2.max']] = round(data_all[['pO2.min', 'pO2.mean', 'pO2.max']], 2)
    df_averaged = data_all[['pO2.mean', 'tau_ref [s]', 'tau0/tau_ref.min', 'tau0/tau_ref.mean', 'tau0/tau_ref.max',
                            'tau_dual.mean [s]']].groupby('pO2.mean').mean()

    df_averaged['tau0 / tau'] = df_averaged['tau_dual.mean [s]'][0] / df_averaged['tau_dual.mean [s]'][reg_range[0]:reg_range[1]]

    # regression of selected range
    if regression is True:
        # Fit KernelRidge with parameter selection based on 5-fold cross validation
        param_grid = {"kernel": [Matern(l, p) for l in np.logspace(0, 200, 100) for p in np.logspace(0, 200, 100)]}
        gp_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

        gpr = GaussianProcessRegressor(kernel=gp_kernel)

        tt = df_averaged[['tau_dual.mean [s]', 'tau0 / tau']][reg_range[0]:reg_range[1]].dropna()
        ydata_tau_dual = tt['tau_dual.mean [s]'].values
        ydata_tau0tau = 1/tt['tau0 / tau'].values
        xdata = np.array([[l] for l in tt.index])

        # Fit and predict using gaussian process regressor
        gpr.fit(xdata, ydata_tau_dual)
        y_gpr1, y_std1 = gpr.predict(xdata, return_std=True)

        # Fit and predict using gaussian process regressor
        gpr.fit(xdata, ydata_tau0tau)
        y_gpr_quot, y_std_quot = gpr.predict(xdata, return_std=True)

        df_fit_ = pd.DataFrame(y_gpr1, index=tt.index, columns=['tau_dual.mean [s]'])
        df_fit = pd.concat([df_fit_, pd.DataFrame(1/y_gpr_quot, index=tt.index, columns=['tau0 / tau'])], axis=1, sort=True)
    else:
        df_fit = df_averaged

    # intensity
    int_ratio_ = af.intensity_ratio(f=f1, tau=data_all['tau_dual.mean [s]'], phi=data_all['dPhi(f1)'].values)
    df_iratio = pd.DataFrame(int_ratio_, index=data_all['pO2.mean'].values, columns=['I pF/dF'])
    df_int_inv = 1 / df_iratio
    df_int_inv.columns = ['I dF/pF']

    # ==================================================================================
    # PLOTTING
    if plotting is True:
        # Lifetime
        f_fit, (ax_fit, ax_quot_fit) = plt.subplots(figsize=(5,3), nrows=2, sharex=True)
        df_label = '{:.2f} +/- {:.2f} °C'.format(data_all['Temp. Probe'].mean(), data_all['Temp. Probe'].std())
        ax_fit.plot(df_fit.index, df_fit['tau_dual.mean [s]']*1e3, ls='--', color='crimson', label=df_label)
        #ax_fit.fill_between(df_fit.index, df_fit[df_fit.columns[0]].values*1e3,
        #                    df_fit[df_fit.columns[1]].values*1e3, alpha=0.2, color='grey', label='range')
        ax_fit.set_ylim(df_fit['tau_dual.mean [s]'].min()*1e3*0.95, df_fit['tau_dual.mean [s]'].max()*1.05*1e3)

        ax_quot_fit.plot(df_fit.index, df_fit['tau0 / tau'], ls='--', color='crimson', label='fit')

        ax_fit.legend(fontsize=9)

        ax_fit.yaxis.set_major_locator(MultipleLocator(1))
        ax_fit.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax_quot_fit.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_quot_fit.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax_fit.xaxis.set_major_locator(MultipleLocator(2))
        ax_fit.xaxis.set_minor_locator(MultipleLocator(0.5))

        ax_quot_fit.set_ylim(df_fit['tau0 / tau'].min()*0.95, df_fit['tau0 / tau'].max()*1.05)

        ax_fit.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_, top=True, right=True)
        ax_quot_fit.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_, top=True,
                                right=True)
        ax_quot_fit.set_xlabel('pO2 [hPa]', fontsize=fontsize_)
        ax_fit.set_ylabel('tauP [ms]', fontsize=fontsize_)
        ax_quot_fit.set_ylabel('tau0 / tauP', fontsize=fontsize_)

        plt.tight_layout(pad=0.5)
        plt.show()

        # Intensity ratio
        f_intFit, (ax_intFit, ax_inv) = plt.subplots(figsize=(5,3), nrows=2, sharex=True)

        ax_intFit.plot(df_iratio, color='#66a5ad', lw=1.25, label=df_label)
        # ax_intFit.fill_between(df_fit_int.index, df_fit_int[df_fit_int.columns[0]],
        #                       df_fit_int[df_fit_int.columns[1]], color='grey', alpha=0.15)

        ax_inv.plot(df_int_inv, color='navy', lw=1.25, label=df_label)
        # ax_inv.fill_between(df_fit_inv.index, df_fit_inv[df_fit_inv.columns[0]],
        #                    df_fit_inv[df_fit_inv.columns[1]], color='grey', alpha=0.15)
        ax_intFit.legend(fontsize=9)
        ax_intFit.yaxis.set_major_locator(MultipleLocator(10))
        ax_intFit.yaxis.set_minor_locator(MultipleLocator(2))
        ax_inv.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_inv.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax_inv.xaxis.set_major_locator(MultipleLocator(5))
        ax_inv.xaxis.set_minor_locator(MultipleLocator(0.5))

        ax_intFit.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_ * 0.9, right=True,
                              top=True)
        ax_inv.tick_params(axis='both', which='both', direction='in', labelsize=fontsize_ * 0.9, right=True, top=True)

        ax_inv.set_xlabel('pO2 [hPa]', fontsize=fontsize_ * 0.95)
        ax_intFit.set_ylabel('Intensity pF/dF', fontsize=fontsize_ * 0.95)
        ax_inv.set_ylabel('Intensity dF/pF', fontsize=fontsize_ * 0.95)

        plt.tight_layout(pad=0.5)

    return df_averaged, df_fit


def smooth_data(df, df_firesting, num_freq, mode, to_smooth='dPhi'):
    if to_smooth == 'dPhi':
        var = 'dPhi(f{}) [deg]'
    else:
        var = 'A(f{}) [mV]'

    dict_smooth = {}
    for i in range(num_freq):
        ydata = savgol_filter(df[var.format(i + 1)].dropna().values, 101, 2, mode=mode)
        df_smooth = pd.DataFrame(ydata, index=df[var.format(i + 1)].dropna().index, columns=[var.format(i + 1)])
        dict_smooth[i] = df_smooth

    # smooth firesting (dPhi raw, intensity, temperature and time)
    dphi_smooth = dict_smooth[0]
    for i in range(num_freq - 1):
        dphi_smooth = dphi_smooth.join(dict_smooth[i + 1], how='outer')
    l = []
    [l.append(var.format(n + 1)) for n in range(num_freq)]
    dphi_smooth.columns = l # list(df.columns[:num_freq])

    start_meas_lI = dphi_smooth.index[0]
    if df_firesting.index[0] < start_meas_lI:
        start_measurement = df_firesting.index[0]
    else:
        start_measurement = start_meas_lI

    t_lockIn = dphi_smooth.index - start_measurement
    t_firesting = df_firesting.index - start_measurement

    df_firesting['Time (s)'] = t_firesting.seconds
    dphi_smooth['delta Time'] = t_lockIn.seconds
    dphi_smooth = dphi_smooth.join(df_firesting)

    if any(dphi_smooth.index.duplicated()):
        store_row = pd.DataFrame(dphi_smooth.loc[dphi_smooth[dphi_smooth.index.duplicated()].index[0]].values[0],
                                 columns=[dphi_smooth[dphi_smooth.index.duplicated()].index[0]],
                                 index=dphi_smooth.columns).T
        dphi_smooth = dphi_smooth.drop(store_row.index[0])
        dphi_smooth.loc[store_row.index[0]] = store_row.values[0]

    dphi_smooth = dphi_smooth.sort_index()

    return dphi_smooth


def data_interpolation(num_freq, df, to_interpolate='dPhi'):
    if to_interpolate == 'dPhi':
        var = 'dPhi(f{}) [deg]'
    else:
        var = 'A(f{}) [mV]'

    dic_reg = {}
    # LockIn-measurement
    for col in range(num_freq):
        y = df[var.format(col + 1)].dropna()
        ydata = y.values
        xdata = df.loc[y.index]['delta Time'].values

        start_date = df[df['delta Time'] == xdata[0]].index[0]
        end_date = df[df['delta Time'] == xdata[-1]].index[0]

        timeindex_new = df.loc[start_date:end_date].index
        time_seconds = df.loc[start_date:end_date, 'delta Time'].values

        yinterp = np.interp(time_seconds, xdata, ydata)
        if col + 1 == num_freq:
            df_inter = pd.DataFrame([yinterp, time_seconds]).T
            df_inter.columns = [var.format(col + 1), 'delta Time']
        else:
            df_inter = pd.DataFrame(yinterp)
            df_inter.columns = [var.format(col + 1)]

        df_inter.index = timeindex_new
        dic_reg[col] = df_inter

    # Firesting-measurement
    # dphi
    xdata = df.loc[df['dphi1 raw'].dropna().index]['delta Time'].values
    ydata = df['dphi1 raw'].dropna().values
    yinterp = np.interp(time_seconds, xdata, ydata)
    df_inter = pd.DataFrame(yinterp)
    df_inter.columns = ['dphi1 raw']
    df_inter.index = timeindex_new
    dic_reg[num_freq + 1] = df_inter

    # int
    xdata = df.loc[df['int1'].dropna().index]['delta Time'].values
    ydata = df['int1'].dropna().values
    yinterp = np.interp(time_seconds, xdata, ydata)
    df_inter = pd.DataFrame(yinterp)
    df_inter.columns = ['int1']
    df_inter.index = timeindex_new
    dic_reg[num_freq + 2] = df_inter

    # temperature
    xdata = df.loc[df['Temp. Probe'].dropna().index]['delta Time'].values
    ydata = df['Temp. Probe'].dropna().values
    yinterp = np.interp(time_seconds, xdata, ydata)
    df_inter = pd.DataFrame(yinterp)
    df_inter.columns = ['Temp. Probe']
    df_inter.index = timeindex_new
    dic_reg[num_freq + 3] = df_inter

    dphi = pd.concat(dic_reg, axis=1, sort=True)

    return dphi


def synchronize_sensors2(file_firesting, day, num_freq, frequencies, ddphi, time_d, lock_raw_, plotting=True,
                         smooth=True, threshold_=1, mode='nearest', fontsize_=9.5, corr_dPhi=False):
    # firesintg preparation
    df_firesting, header_lines = reference_sensor(file=file_firesting, day=day, plotting=plotting)

    time_fs = pd.read_csv(file_firesting, encoding='latin-1', skiprows=header_lines + 4, header=None,
                          sep='\t', usecols=[0, 1], skipinitialspace=True)
    start_time_fs = datetime.datetime.strptime(' '.join(time_fs.values[0]), "%d.%m.%Y %H:%M:%S")
    stop_time_fs = datetime.datetime.strptime(' '.join(time_fs.values[-1]), "%d.%m.%Y %H:%M:%S")

    # time extraction from firesting
    if start_time_fs.strftime("%d-%m-%Y") == stop_time_fs.strftime("%d-%m-%Y"):
        print('day of measurement:', start_time_fs.strftime("%d-%m-%Y"))
    else:
        print('days of measurement:', start_time_fs.strftime("%d.%m"), '-',
              stop_time_fs.strftime("%d.%m.%Y"))
    print('--------------------------------------------------\n')
    # ==========================================================================================
    # FIRESTING
    print('start of measurement firesting:', start_time_fs.strftime("%d.%m.%Y %H:%M:%S"))
    print('end of measurement firesting:', stop_time_fs.strftime("%d.%m.%Y %H:%M:%S"))
    print('--------------------------------------------------')

    ddf_firesting_sliced = df_firesting[['dphi1 raw', 'int1', 'Temp. Probe']]

    [dphi_lockIn, amplitudes_lockIn,
     freq_Hz] = dualsensor_measurement2(file_firesting=file_firesting, num_freq=num_freq, frequencies=frequencies,
                                        ddphi=ddphi, time_d=time_d, lock_raw_=lock_raw_, corr_dPhi=corr_dPhi)

    dphi_lockIn['Time (s)'] = dphi_lockIn.index
    dphi_lockIn = dphi_lockIn.set_index('DateTime')
    dphi_total = dphi_lockIn.join(ddf_firesting_sliced)

    amplitudes_lockIn['Time (s)'] = amplitudes_lockIn.index
    amplitudes_lockIn = amplitudes_lockIn.set_index('DateTime')
    ampl_total = amplitudes_lockIn.join(ddf_firesting_sliced)

    # combine amplitude and dphi to common dataframe
    data_all = pd.concat([dphi_total, ampl_total[ampl_total.columns[:num_freq]]], axis=1, sort=True)

    # check and remove outliers with z-function for each dphi and amplitude
    z_dphi1 = np.abs(stats.zscore(dphi_total[dphi_total.columns[0]].dropna()))
    z_dphi2 = np.abs(stats.zscore(dphi_total[dphi_total.columns[1]].dropna()))
    z_ampl1 = np.abs(stats.zscore(data_all[data_all.columns[-2]].dropna()))
    z_ampl2 = np.abs(stats.zscore(data_all[data_all.columns[-1]].dropna()))
    outlier_index_dphi1 = dphi_total.index[np.where(z_dphi1 > threshold_)[0]]
    outlier_index_dphi2 = dphi_total.index[np.where(z_dphi2 > threshold_)[0]]
    outlier_index_ampl1 = data_all.index[np.where(z_ampl1 > threshold_)[0]]
    outlier_index_ampl2 = data_all.index[np.where(z_ampl2 > threshold_)[0]]

    # combining outlier indices
    outlier_index_dphi = outlier_index_dphi1.append(outlier_index_dphi2)
    outlier_index_amplitude = outlier_index_ampl1.append(outlier_index_ampl2)
    outlier_index = outlier_index_dphi.append(outlier_index_amplitude)

    dphi_total = dphi_total.drop(outlier_index_dphi)
    ampl_total = ampl_total.drop(outlier_index_amplitude)
    data_all = data_all.drop(outlier_index)

    if any(dphi_total.index.duplicated()):
        store_row = pd.DataFrame(dphi_total.loc[dphi_total[dphi_total.index.duplicated()].index[0]].values[0],
                                 columns=[dphi_total[dphi_total.index.duplicated()].index[0]],
                                 index=dphi_total.columns).T
        dphi_total = dphi_total.drop(store_row.index[0])
        dphi_total.loc[store_row.index[0]] = store_row.values[0]
    dphi_total = dphi_total.sort_index()

    # ==================================================================================
    # Smooth values
    if smooth is True:
        dphi_smooth = smooth_data(df=dphi_total, df_firesting=df_firesting, num_freq=num_freq, mode=mode,
                                  to_smooth='dPhi')
        dphi = data_interpolation(num_freq=num_freq, df=dphi_smooth, to_interpolate='dPhi')
        dphi.columns = dphi.columns.get_level_values(1)

        # smooth all data and combine them afterwards
        data_smooth1 = smooth_data(df=data_all, df_firesting=df_firesting, num_freq=num_freq, mode=mode,
                                   to_smooth='dPhi')
        data_smooth2 = smooth_data(df=data_all, df_firesting=df_firesting, num_freq=num_freq, mode=mode,
                                   to_smooth='A')
        data_ = data_smooth1.join(data_smooth2[data_smooth2.columns[:num_freq]])

        # interpolate dPhi and amplitude and re-combine them afterwards
        data_1 = data_interpolation(num_freq=num_freq, df=data_, to_interpolate='dPhi')
        data_2 = data_interpolation(num_freq=num_freq, df=data_, to_interpolate='A')

        data = data_1.join(data_2[data_2.columns[:num_freq]])
        data.columns = data.columns.get_level_values(1)
        a = ['dPhi(f{}) [deg]'.format(i+1) for i in range(num_freq)]
        b = ['A(f{}) [mV]'.format(i+1) for i in range(num_freq)]
        c = ['delta Time', 'dphi1 raw', 'int1', 'Temp. Probe']
        data = data[a+b+c]
    else:
        dphi = dphi_total
        data = data_all

    # ==================================================================================
    # Plotting
    if plotting is True:
        f_dphi, (ax_dphi, ax1_dphi) = plt.subplots(figsize=(5, 4), nrows=2, sharex=True)

        ax_temp = ax_dphi.twinx()
        ax_dphi.plot(dphi_total['dphi1 raw'].dropna(), color='k')
        ax_temp.plot(dphi_total['Temp. Probe'].dropna(), color='#fb6542', lw=.75)

        for col in range(num_freq):
            ax1_dphi.plot(dphi['dPhi(f{}) [deg]'.format(col+1)].dropna(), color=colors_freq[col], lw=0., marker='.',
                          markersize=3)

        if smooth is True:
            for col in range(num_freq):
                ax1_dphi.plot(dphi['dPhi(f{}) [deg]'.format(col+1)].dropna(), color='k', lw=.75, ls='--',
                              label='f{} fit'.format(col+1))
            ax1_dphi.legend(fontsize=fontsize_*0.7, loc=0)
        else:
            pass

        ax_dphi.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)
        ax_temp.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in')
        ax1_dphi.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)

        #ax_dphi.set_ylim(0, 60)
        #ax1_dphi.set_ylim(0, 16)

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax_dphi.xaxis.set_major_formatter(myFmt)
        ax_temp.set_ylabel('Temperature$_{firesting}$ [°C]', fontsize=fontsize_, color='#fb6542')
        ax_dphi.set_ylabel('dPhi$_{firesting}$ [deg]', fontsize=fontsize_)
        ax1_dphi.set_ylabel('dPhi$_{lockIn}$ f1 [deg]', fontsize=fontsize_)
        f_dphi.autofmt_xdate(ha='center')

        plt.tight_layout(pad=0.75)
        plt.show()

        # ----------------------------------------------------------------------
        # amplitudes
        f_ampl, (ax, ax1) = plt.subplots(figsize=(5, 4), nrows=2, sharex=True)

        ax.plot(dphi_total['int1'].dropna(), color='k')
        for col in range(num_freq):
            ax1.plot(ampl_total['A(f{}) [mV]'.format(col+1)].dropna(), color=colors_freq[col], lw=0., marker='.',
                     markersize=3)

        ax.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in', top=True)
        ax1.tick_params(which='both', labelsize=fontsize_ * 0.9, direction='in')
        #ax.set_ylim(0, 520)
        #ax1.set_ylim(10, 35)

        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(myFmt)
        ax.set_ylabel('A$_{firesting}$ [mV]', fontsize=fontsize_)
        ax1.set_ylabel('A$_{lockIn}$ f1 [mV]', fontsize=fontsize_)
        f_ampl.autofmt_xdate(ha='center')

        plt.tight_layout(pad=0.75)
        plt.show()

    return data, dphi, ampl_total, freq_Hz


def dualsensor_measurement2(file_firesting, num_freq, frequencies, ddphi, time_d, lock_raw_, corr_dPhi=False):
    """

    :param file_firesting:
    :param num_freq:
    :param frequencies:
    :param ddphi:
    :param time_d:
    :param lock_raw_:
    :return: phi_lockIn, amplitudes_lockIn, f1, f2
    """
    # input preparation
    freq_Hz = []
    ddPhi_f = []

    for i in range(num_freq):
        if len(frequencies[i]) == 0:
            pass
        else:
            [f_, ddphi_f, t_step] = preparation_input(f_=frequencies[i], ddphi_f=ddphi[i], time_d=time_d)
            freq_Hz.append(f_)
            ddPhi_f.append(ddphi_f)
            time_step = t_step

    for i, v in enumerate(lock_raw_[0]):
        if lock_raw_[0][i + 2] - lock_raw_[0][i] > time_step - 1:
            time_start = i
            break

    lock_raw = lock_raw_.loc[time_start:]
    lock_raw.index = np.arange(len(lock_raw.index))

    # ===========================================================================
    # convert 'Time (s)'' into real meausurement time before splitting
    # time extraction LockIn
    stop_time_lockin = datetime.datetime.strptime(' '.join(file_firesting.split('/')[-1].split('_')[0].split('-')),
                                                  "%Y%m%d %H%M%S")
    start_time_lockin = stop_time_lockin - datetime.timedelta(seconds=lock_raw[0].values[-1])
    print('start of measurement LockIn:', start_time_lockin.strftime("%d.%m.%Y %H:%M:%S"))
    print('end of measurement LockIn:', stop_time_lockin.strftime("%d.%m.%Y %H:%M:%S"))

    time_ls_full = []
    time_ls = []
    for i in lock_raw[0]:
        t = start_time_lockin + datetime.timedelta(seconds=i)
        t = t - datetime.timedelta(microseconds=t.microsecond)
        time_ls_full.append(t)
        time_ls.append(t.strftime('%d.%m.%Y %H:%M:%S'))

    lock_raw['DateTime'] = time_ls
    lock_raw.columns = ['Time (s)', 'Amplitude [mV]', 'dPhi [deg]', 'DateTime']

    # ----------------------------------------------------------------------
    # split dataframe into f1 and f2 - dataframe
    dic_f = {}
    for i in range(num_freq):
        d_ = split_lockin_data_into_dataframes(num_freq=num_freq, df_lockIn=lock_raw, run=i)
        d_.columns = ['Time (s)', 'A(f{}) [mV]'.format(i+1), 'dPhi(f{}) [deg]'.format(i+1),
                      'DateTime (f{})'.format(i+1)]
        dic_f[i] = d_

    # Split into phase angle and amplitude
    l_ = [pd.concat([dic_f[i].set_index('Time (s)')], axis=1, sort=True) for i in range(len(dic_f.keys()))]
    ddf_ = pd.concat(l_, axis=1, sort=True)

    drop_row = []
    for i in range(len(ddf_.index)):
        if all(v == 0 or np.isnan(v) for v in list(ddf_.loc[ddf_.index[i]].values)) is True:
            drop_row.append(ddf_.index[i])

    for c in drop_row:
        if c in ddf_.index:
            ddf_ = ddf_.drop(c)

    time_lockIn_common = []
    for i in ddf_.index:
        for r in range(num_freq):
            if isinstance(ddf_['DateTime (f{})'.format(r + 1)][i], np.float):  # is nan
                pass
            else:
                t = datetime.datetime.strptime(ddf_['DateTime (f{})'.format(r + 1)][i], "%d.%m.%Y %H:%M:%S")
                time_lockIn_common.append(t)

    amplitudes_lockIn = ddf_[[y for y in list(ddf_.columns) if 'A(f' in y]]
    amplitudes_lockIn['DateTime'] = time_lockIn_common

    dphi_lockIn = ddf_[[y for y in list(ddf_.columns) if 'dPhi(f' in y]]
    dphi_lockIn['DateTime'] = time_lockIn_common

    for i, c in enumerate(dphi_lockIn.columns):
        if i != num_freq:
            if corr_dPhi is False:
                dphi_lockIn[c] = dphi_lockIn[c]
            else:
                dphi_lockIn[c] = np.abs(dphi_lockIn[c] + ddPhi_f[i])
    # ddf_ = pd.concat([df_f1.set_index('Time (s)'), df_f2.set_index('Time (s)')],
    #                  axis=1, sort=True).iloc[1:]
    #     time_lockIn_common = []
    # for i in ddf_.index:
    #     if isinstance(ddf_['DateTime (f1)'][i], np.float):  # is nan
    #         t = datetime.datetime.strptime(ddf_['DateTime (f2)'][i], "%d.%m.%Y %H:%M:%S")
    #     else:
    #         t = datetime.datetime.strptime(ddf_['DateTime (f1)'][i], "%d.%m.%Y %H:%M:%S")
    #     time_lockIn_common.append(t)
    #
    # amplitudes_lockIn = ddf_[['A(f1) [mV]', 'A(f2) [mV]']]
    # amplitudes_lockIn['DateTime'] = time_lockIn_common
    #
    # dphi_lockIn = pd.concat([np.abs(ddf_.loc[:, 'dPhi(f1) [deg]'] + ddphi_f1),
    #                                 np.abs(ddf_.loc[:, 'dPhi(f2) [deg]'] + ddphi_f2)], axis=1, sort=True)
    # dphi_lockIn['DateTime'] = time_lockIn_common

    return dphi_lockIn, amplitudes_lockIn, freq_Hz  # df_f1, df_f2, f1, f2


def sigmoid(p, x):
    x0, y0, c, k = p
    y = c / (1 + np.exp(-k * (x - x0))) + y0
    return y


def residuals(p, x, y):
    return y - sigmoid(p, x)


def resize(arr, lower=0.0, upper=1.0):
    arr = arr.copy()
    if lower > upper: lower, upper = upper, lower
    arr -= arr.min()
    arr *= (upper - lower) / arr.max()
    arr += lower

    return arr
