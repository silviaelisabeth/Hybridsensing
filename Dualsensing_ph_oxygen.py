__author__ = 'szieger'
__project__ = 'dualsensor ph/O2 sensing'

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import additional_functions as af
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib import cm

color_pH = ['navy', '#07575B']


# ---------------------------------------------------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------------------------------------------------
def plotting_twoSiteModel(pO2_calib, df_tau_quot, df_intP, normalized=True, fontsize_=13):
    if normalized is True:
        df_intP_norm = df_intP / df_intP.loc[0]
    else:
        df_intP_norm = df_intP

    f, ax = plt.subplots()
    ax1 = ax.twinx()

    ax.plot(df_tau_quot.index, 1/df_tau_quot, color='navy', lw=1., label='tau0/tau')
    ax1.plot(df_intP_norm, color='orange', lw=1., label='$I_P$')

    ax.axvline(pO2_calib[0], color='k', lw=1., ls='--')
    ax.axvline(pO2_calib[1], color='k', lw=1., ls='--')

    # find closest value to calibration point
    if len(df_tau_quot.columns) == 3:
        data_quot = df_tau_quot[1].values
    else:
        data_quot = df_tau_quot.values
    a = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=pO2_calib[0])
    b = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=pO2_calib[1])

    if a[0] == a[1]:
        tauQ_c0 = a[2]
    else:
        arg_calib0 = stats.linregress(x=a[:2], y=a[2:])
        tauQ_c0 = arg_calib0[0] * pO2_calib[0] + arg_calib0[1]
    if b[0] == b[1]:
        tauQ_c1 = b[2]
    else:
        arg_calib1 = stats.linregress(x=b[:2], y=b[2:])
        tauQ_c1 = arg_calib1[0] * pO2_calib[1] + arg_calib1[1]

    if np.isnan(tauQ_c0):
        y_c0 = df_tau_quot.loc[pO2_calib[0]].values[1]
    else:
        y_c0 = tauQ_c0
    if np.isnan(tauQ_c1):
        y_c1 = df_tau_quot.loc[pO2_calib[1]].values[1]
    else:
        y_c1 = tauQ_c1

    ax.axhline(1/y_c0, color='k', lw=1., ls='--')
    ax.axhline(1/y_c1, color='k', lw=1., ls='--')

    ax.set_xlabel('p$O_2$ / hPa', fontsize=fontsize_)
    ax.set_ylabel('τ$_0$ / τ', color='navy', fontsize=fontsize_)
    ax1.set_ylabel('Intensity I$_P$', color='orange', fontsize=fontsize_)

    plt.tight_layout()


def plotting_boltzmann_sigmoid(pH_range, pH_calib, int_fluoro, color, type_='int', ax=None, fontsize_=13):
    if ax is None:
        f, ax = plt.subplots()

    if len(int_fluoro.columns) == 3:
        ax.plot(pH_range, int_fluoro['mean'], color=color, lw=1.)
        ax.plot(pH_range, int_fluoro['min'], color='k', ls='--', lw=.5)
        ax.plot(pH_range, int_fluoro['max'], color='k', ls='--', lw=.5)
        ax.fill_between(pH_range, int_fluoro['min'], int_fluoro['max'], color='grey', alpha=0.1, lw=0.25)
    else:
        ax.plot(pH_range, int_fluoro, color='#07575B', lw=1.)
        ax.axhline(int_fluoro.loc[pH_calib[0]].values[0], color='k', lw=1., ls='--')
        ax.axhline(int_fluoro.loc[pH_calib[1]].values[0], color='k', lw=1., ls='--')

    ax.axvline(pH_calib[0], color='k', ls='--', lw=1.)
    ax.axvline(pH_calib[1], color='k', ls='--', lw=1.)

    ax.set_xlabel('pH', fontsize=fontsize_)
    if type_ == 'int':
        ax.set_ylabel('Intensity $I_f$', fontsize=fontsize_)
    else:
        ax.set_ylabel('cot(Phi)', fontsize=fontsize_)
    return ax


def plotting_dualsensor(pO2_calib, pO2_calc, df_tau_quot, pH_range, pH_calib, pH_calc, int_fluoro, fontsize_=13,
                        method='int'):

    f, (ax_pO2, ax_pH) = plt.subplots(ncols=2, figsize=(7, 3))

    # ---------------------------------------------------------------------
    # pO2 sensing
    ax_pO2.plot(df_tau_quot.index, 1/df_tau_quot[1], color='navy', lw=1., label='tau0/tau')
    ax_pO2.plot(1/df_tau_quot[0], color='k', lw=.25, ls='--')
    ax_pO2.plot(1/df_tau_quot[2], color='k', lw=.25, ls='--')
    ax_pO2.fill_between(df_tau_quot.index, 1/df_tau_quot[0], 1/df_tau_quot[2], color='grey', alpha=0.1, lw=0.25)

    ax_pO2.set_xlabel('p$O_2$ / hPa', fontsize=fontsize_)
    ax_pO2.set_ylabel('τ$_0$ / τ', color='navy', fontsize=fontsize_)

    # calibration and measurement points
    ax_pO2.axvline(pO2_calib[0], color='k', lw=1., ls='--')
    ax_pO2.axvline(pO2_calib[1], color='k', lw=1., ls='--')
    ax_pO2.axvspan(pO2_calc[0], pO2_calc[2], color='#f0810f', alpha=0.4)

    # find closest value to calibration point
    if len(df_tau_quot.columns) == 3:
        data_quot = df_tau_quot[1].values
    else:
        data_quot = df_tau_quot.values
    a = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=pO2_calib[0])
    b = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=pO2_calib[1])

    if a[0] == a[1]:
        tauQ_c0 = a[2]
    else:
        arg_calib0 = stats.linregress(x=a[:2], y=a[2:])
        tauQ_c0 = arg_calib0[0] * pO2_calib[0] + arg_calib0[1]
    if b[0] == b[1]:
        tauQ_c1 = b[2]
    else:
        arg_calib1 = stats.linregress(x=b[:2], y=b[2:])
        tauQ_c1 = arg_calib1[0] * pO2_calib[1] + arg_calib1[1]

    if np.isnan(tauQ_c0):
        y_c0 = df_tau_quot.loc[pO2_calib[0]].values[1]
    else:
        y_c0 = tauQ_c0
    if np.isnan(tauQ_c1):
        y_c1 = df_tau_quot.loc[pO2_calib[1]].values[1]
    else:
        y_c1 = tauQ_c1

    ax_pO2.axhline(1/y_c0, color='k', lw=1., ls='--')
    ax_pO2.axhline(1/y_c1, color='k', lw=1., ls='--')

    #ax_pO2.axhline(y=1/df_tau_quot.loc[pO2_calib[0], 1], color='k', lw=1., ls='--')
    #ax_pO2.axhline(y=1/df_tau_quot.loc[pO2_calib[1], 1], color='k', lw=1., ls='--')

    # find closest value
    tauq_meas_min = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[0].values, value=pO2_calc[0])
    tauq_meas_max = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[2].values, value=pO2_calc[2])

    # linear regression to pO2 measured
    arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
    arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
    y_min = 1/(arg_min[0]*pO2_calc[0] + arg_min[1])
    y_max = 1/(arg_max[0]*pO2_calc[2] + arg_max[1])

    ax_pO2.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

    # ---------------------------------------------------------------------
    # pH sensing
    ax_pH.plot(pH_range, int_fluoro['mean'], color='#07575B', lw=1.)
    ax_pH.plot(pH_range, int_fluoro['min'], color='k', ls='--', lw=.5)
    ax_pH.plot(pH_range, int_fluoro['max'], color='k', ls='--', lw=.5)
    ax_pH.fill_between(pH_range, int_fluoro['min'], int_fluoro['max'], color='grey', alpha=0.1, lw=0.25)

    # calibration and measurement points
    ax_pH.axvline(pH_calib[0], color='k', ls='--', lw=1.)
    ax_pH.axvline(pH_calib[1], color='k', ls='--', lw=1.)
    ax_pH.axvspan(pH_calc[0], pH_calc[2], color='#f0810f', alpha=0.4)

    # find closest values
    iF_calib0 = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pH_calib[0])
    iF_calib1 = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pH_calib[1])
    iF_meas_min = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pH_calc[0])
    iF_meas_max = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pH_calc[2])

    # linear regression to pH measured
    arg_min = stats.linregress(x=pH_calib, y=iF_meas_min[2:])
    arg_max = stats.linregress(x=pH_calib, y=iF_meas_max[2:])
    y_min = arg_min[0]*pH_calc[0] + arg_min[1]
    y_max = arg_max[0]*pH_calc[2] + arg_max[1]

    ax_pH.axhline(y=sum(iF_calib0[2:])/2, color='k', lw=1., ls='--')
    ax_pH.axhline(y=sum(iF_calib1[2:])/2, color='k', lw=1., ls='--')
    ax_pH.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

    ax_pH.set_xlabel('pH', fontsize=fontsize_)
    if method == 'int':
        ax_pH.set_ylabel('Intensity $I_f$', color='#07575B', fontsize=fontsize_)
    else:
        ax_pH.set_ylabel('cot(Phi)', color='#07575B', fontsize=fontsize_)

    plt.tight_layout()

    return ax_pO2, ax_pH


# ---------------------------------------------------------------------------------------------------------------------
# Additional functions
# ---------------------------------------------------------------------------------------------------------------------
def twoSiteModel_fit(pO2_range, f, m, Ksv, tau_phos0, conv_tau_int, int_phos0, plotting=True, pO2_calib=None,
                     normalized_int=True):
    if isinstance(Ksv, np.float):
        # tau/tau0
        tau_quot = (f / (1 + Ksv*pO2_range)) + (1 - f) / (1 + Ksv*m*pO2_range)
        tauP = tau_quot * tau_phos0
        # conversion into dataFrame
        tau_quot = pd.DataFrame(tau_quot, index=pO2_range)
    else:
        # tau/tau0
        tau_quot = [(f / (1 + p*pO2_range)) + (1 - f) / (1 + p*m*pO2_range) for p in Ksv]
        tauP = [t_quot * t0 for (t_quot, t0) in zip(tau_quot, tau_phos0)]

        # conversion into dataFrame
        tau_quot = pd.DataFrame(tau_quot, columns=pO2_range).T

    # convert lifetime in seconds
    if type(tauP) == list:
        if tauP[1][1] < 1:
            tauP = pd.DataFrame(tauP).T
        else:
            tauP = pd.DataFrame(tauP).T*1E-6
    else:
        if tauP[0] < 1:
            tauP = pd.DataFrame(tauP).T
        else:
            if len(pd.DataFrame(tauP).columns) == 1:
                tauP = pd.DataFrame(tauP)*1E-6
            else:
                tauP = pd.DataFrame(tauP).T*1E-6
    tauP.index = pO2_range

    # pre-check is each index of df_tau is in df_conv_tau_int
    if isinstance(pO2_range, np.float):
        if pO2_range in conv_tau_int.index:
            intP = [i * t * conv_tau_int.loc[pO2_range].values[0] for (i, t) in zip(int_phos0, tauP)]
        else:
            print('find closest values')
    else:
        if set(pO2_range) >= set(conv_tau_int.index) is True:
            pass
        else:
            xdata = np.array(conv_tau_int.index)
            ydata = np.array(conv_tau_int[conv_tau_int.columns[0]].values)

            try:
                popt, pcov = curve_fit(af.func_exp, xdata, ydata)
                conv_tau_int = pd.DataFrame(af.func_exp(pO2_range, *popt), index=pO2_range)
            except RuntimeError:
                popt, pcov = curve_fit(af.func_exp_2, xdata, ydata, maxfev=1000)
                conv_tau_int = pd.DataFrame(af.func_exp_2(pO2_range, *popt), index=pO2_range)

        intP = tauP.copy()
        if len(tauP.columns) == 3:
            for j in tauP.columns:
                for i in tauP.index:
                    intP.loc[i, j] = tauP.loc[i, j] * conv_tau_int.loc[i, 0] * int_phos0
        else:
            for i in tauP.index:
                intP.loc[i] = tauP.loc[i] * conv_tau_int.loc[i] * int_phos0

    # plotting lifetime quotient and normalized intensity
    if plotting is True:
        if pO2_calib is None:
            raise ValueError('pO2 calibration points are required')
        else:
            plotting_twoSiteModel(pO2_calib=pO2_calib, df_tau_quot=tau_quot, df_intP=intP, normalized=normalized_int)

    return tau_quot, tauP, intP


def twoSiteModel_calibration(tau0, tau1, m, f, pO2_calib1):
    # preparation
    tau_quot = [x1/x2 for (x1, x2) in zip(tau1, tau0)]

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
        raise ValueError('decide about Ksv')

    # combining all (fit) parameter for two-site-model
    para_TSM = pd.Series({'prop Ksv': m, 'slope': f, 'Ksv_fit1': Ksv_fit1, 'Ksv_fit2': [k*m for k in Ksv_fit1]})

    return para_TSM


def twoSiteModel_evaluation(tau0, tau, m, f, ksv, pO2_range):
    # preparation pq equation
    quot = [t1/t0 for (t1, t0) in zip(tau, tau0)]

    c = [1-1/q for q in quot]
    b = [k*(m*(1-f/t) + 1/t*(f-1)+1) for (k, t) in zip(ksv, quot)]
    a = [m*(p**2) for p in ksv]

    sqrt = [np.sqrt(b_**2 - 4*a_*c_) for (b_, a_, c_) in zip(b, a,c)]
    z1 = [-1*b_ + s for (b_, s) in zip(b, sqrt)]
    z2 = [-1*b_ - s for (b_, s) in zip(b, sqrt)]
    n = [2*a_ for a_ in a]

    pO2_1 = sorted([z / n for (z, n) in zip(z1, n)])
    pO2_2 = sorted([z / n for (z, n) in zip(z2, n)])

    # select reasonable pO2 value
    if pO2_1[1] < -1. or np.abs(pO2_1[1]) > np.abs(pO2_range[-1]):
        print('select pO2_2', pO2_2)
        pO2_calc = pO2_2
    elif pO2_1[1] == 0 or pO2_2[1] == 0:
        if pO2_1[1] >= 0:
            pO2_calc = pO2_1
        else:
            pO2_calc = pO2_2
    else:
        pO2_calc = pO2_1

    pO2_calc = np.array(pO2_calc)

    return pO2_calc


# --------------------------------------------------------------------------
# pH sensing
def boltzmann_fit_cotPhi(dPhi_ph0, dPhi_ph1, v50, pH_range, pH_calib):
    bottom_z = af.cot(np.deg2rad(dPhi_ph1)) * (1 + 10**(pH_calib[1] - v50)) / (1 + 10**(pH_calib[0] - v50)) - \
               af.cot(np.deg2rad(dPhi_ph0))
    bottom_n = (1 + 10**(pH_calib[1] - v50)) / (1 + 10**(pH_calib[0] - v50)) - 1
    bottom = bottom_z / bottom_n
    top = (1 + 10**(pH_calib[0] - v50)) * (af.cot(np.deg2rad(dPhi_ph0)) - bottom) + bottom

    cotPhi = [bottom + (top - bottom) / (1 + 10**(pH_range - v50)) for (top, bottom) in zip(top, bottom)]
    df_cotPhi = pd.DataFrame(cotPhi, columns=pH_range, index=['min', 'mean', 'max']).T

    para = pd.Series({'top': top, 'bottom': bottom, 'cotPhi': df_cotPhi})
    return para


# linear regression to pO2 calculated
def reg_boltzmann_top_bottom(pO2_calib, pO2_calc, para_c0, para_c1):
    reg_top0 = stats.linregress(x=pO2_calib, y=[para_c0['top'][0], para_c1['top'][0]])
    reg_top1 = stats.linregress(x=pO2_calib, y=[para_c0['top'][1], para_c1['top'][1]])
    reg_top2 = stats.linregress(x=pO2_calib, y=[para_c0['top'][2], para_c1['top'][2]])

    reg_bottom0 = stats.linregress(x=pO2_calib, y=[para_c0['bottom'][0], para_c1['bottom'][0]])
    reg_bottom1 = stats.linregress(x=pO2_calib, y=[para_c0['bottom'][1], para_c1['bottom'][1]])
    reg_bottom2 = stats.linregress(x=pO2_calib, y=[para_c0['bottom'][2], para_c1['bottom'][2]])

    top0 = [reg_top0[0]*pO2_calc[0] + reg_top0[1], reg_top0[0]*pO2_calc[1] + reg_top0[1], reg_top0[0]*pO2_calc[2] +
            reg_top0[1]]
    top1 = [reg_top1[0]*pO2_calc[0] + reg_top1[1], reg_top1[0]*pO2_calc[1] + reg_top1[1], reg_top1[0]*pO2_calc[2] +
            reg_top1[1]]
    top2 = [reg_top2[0]*pO2_calc[0] + reg_top2[1], reg_top2[0]*pO2_calc[1] + reg_top2[1], reg_top2[0]*pO2_calc[2] +
            reg_top2[1]]

    bottom0 = [reg_bottom0[0]*pO2_calc[0] + reg_bottom0[1], reg_bottom0[0]*pO2_calc[1] + reg_bottom0[1],
               reg_bottom0[0]*pO2_calc[2] + reg_bottom0[1]]
    bottom1 = [reg_bottom1[0]*pO2_calc[0] + reg_bottom1[1], reg_bottom1[0]*pO2_calc[1] + reg_bottom1[1],
               reg_bottom1[0]*pO2_calc[2] + reg_bottom1[1]]
    bottom2 = [reg_bottom2[0]*pO2_calc[0] + reg_bottom2[1], reg_bottom2[0]*pO2_calc[1] + reg_bottom2[1],
               reg_bottom2[0]*pO2_calc[2] + reg_bottom2[1]]

    top = pd.Series({'min': top0, 'mean': top1, 'max': top2})
    bottom = pd.Series({'min': bottom0, 'mean': bottom1, 'max': bottom2})
    return top, bottom


def linreg_intensity_to_pO2(pO2_calc, x, i_f_c0, i_f_c1):
    y_min_ = [i_f_c0[0], i_f_c1[0]]
    y_mean_ = [i_f_c0[1], i_f_c1[1]]
    y_max_ = [i_f_c0[2], i_f_c1[2]]

    arg_min = stats.linregress(x=x, y=y_min_)
    arg_mean = stats.linregress(x=x, y=y_mean_)
    arg_max = stats.linregress(x=x, y=y_max_)
    y_min = [arg_min[0] * p + arg_min[1] for p in pO2_calc]
    y_mean = [arg_mean[0] * p + arg_mean[1] for p in pO2_calc]
    y_max = [arg_max[0] * p + arg_max[1] for p in pO2_calc]

    y = pd.concat([pd.DataFrame(y_min), pd.DataFrame(y_mean), pd.DataFrame(y_max)], axis=1)
    y.columns = ['min', 'mean', 'max']
    return y


# boltzmann sigmoid (pH vs I_f)
def boltzmann_sigmoid(pH_range, top, bottom, v50, slope):
    if isinstance(top, np.float):
        cot_dPhi = bottom + (top - bottom) / (1 + 10**((-v50+pH_range)/slope))
    else:
        cot_dPhi = [(b1 + (t-b2) / (1 + 10**((-v50 + pH_range)/slope))) for (b1, t, b2) in zip(bottom, top, bottom)]
    return cot_dPhi


def boltzmann_sigmoid_int(pH_range, v50, slope, int_f_max):
    if isinstance(int_f_max, np.float):
        i_f = 10**((v50 - pH_range)/slope) / (1 + 10**((v50 - pH_range)/slope)) * int_f_max
        df_i_f = pd.DataFrame(i_f, index=pH_range)
    else:
        i_f = [(10**((v50 - pH_range)/slope)) / (1 + 10**((v50 - pH_range)/slope)) * i for i in int_f_max]
        df_i_f = pd.DataFrame(i_f, columns=pH_range, index=['min', 'mean', 'max']).T
    return df_i_f


def boltzmann_fit_calibration_intensity(pH0, pH1, int_f0, int_f1, v50, slope, plot=True, fontsize_=13):
    # cot(dPhi) fitting
    t0 = 1 + 10**((-v50 + pH0) / slope)
    t1 = 1 + 10**((-v50 + pH1) / slope)

    if isinstance(int_f0, np.float):
        top = int_f0 + (t0 - 1)*t1*(int_f0 - int_f1) / (t1 - t0)
        bottom = top - ((int_f0 - int_f1) * t0 * t1) / (t1 - t0)
    else:
        top = [i00 + (t0 - 1)*t1/(t1 - t0) * (i0 - i1) for (i00, i0, i1) in zip(int_f0, int_f0, int_f1)]
        bottom = [top_ - ((i0 - i1) * t0 * t1) / (t1 - t0) for (top_, i0, i1) in zip(top, int_f0, int_f1)]

    pH_range = np.linspace(0, 14, num=int((14/0.01 + 1)))
    int_fluoro = boltzmann_sigmoid(pH_range=pH_range, top=top, bottom=bottom, v50=v50, slope=slope)

    int_fluoro = pd.DataFrame(int_fluoro)
    if len(int_fluoro.index) > 3:
        int_fluoro.index = pH_range
    else:
        int_fluoro = int_fluoro.T
        int_fluoro.index = pH_range
        int_fluoro.columns = ['min', 'mean', 'max']

    # -------------------------------------------------------------------------------------------------
    fit_parameter = pd.Series({'top': top, 'bottom': bottom, 'slope': slope, 'v50': v50, 'int_fluoro': int_fluoro})

    if plot is True:
        ax = plotting_boltzmann_sigmoid(pH_range=pH_range, pH_calib=[pH0, pH1], int_fluoro=int_fluoro,
                                        color=color_pH[1], fontsize_=fontsize_, type_='int')

    return fit_parameter


def pH_calculation(i_f_fit, i_fluoro_meas, type_='moderate'):
    # minimal fluorescence range
    a0 = af.find_closest_value_(index=i_f_fit['min'].values, data=i_f_fit.index, value=i_fluoro_meas[0])
    a1 = af.find_closest_value_(index=i_f_fit['min'].values, data=i_f_fit.index, value=i_fluoro_meas[1])
    a2 = af.find_closest_value_(index=i_f_fit['min'].values, data=i_f_fit.index, value=i_fluoro_meas[2])
    # mean fluorescence range
    b0 = af.find_closest_value_(index=i_f_fit['mean'].values, data=i_f_fit.index, value=i_fluoro_meas[0])
    b1 = af.find_closest_value_(index=i_f_fit['mean'].values, data=i_f_fit.index, value=i_fluoro_meas[1])
    b2 = af.find_closest_value_(index=i_f_fit['mean'].values, data=i_f_fit.index, value=i_fluoro_meas[2])
    # maximal fluorescence range
    c0 = af.find_closest_value_(index=i_f_fit['max'].values, data=i_f_fit.index, value=i_fluoro_meas[0])
    c1 = af.find_closest_value_(index=i_f_fit['max'].values, data=i_f_fit.index, value=i_fluoro_meas[1])
    c2 = af.find_closest_value_(index=i_f_fit['max'].values, data=i_f_fit.index, value=i_fluoro_meas[2])

    # linear regression
    arg_a0 = stats.linregress(x=a0[:2], y=a0[2:])
    arg_a1 = stats.linregress(x=a1[:2], y=a1[2:])
    arg_a2 = stats.linregress(x=a2[:2], y=a2[2:])
    arg_b0 = stats.linregress(x=b0[:2], y=b0[2:])
    arg_b1 = stats.linregress(x=b1[:2], y=b1[2:])
    arg_b2 = stats.linregress(x=b2[:2], y=b2[2:])
    arg_c0 = stats.linregress(x=c0[:2], y=c0[2:])
    arg_c1 = stats.linregress(x=c1[:2], y=c1[2:])
    arg_c2 = stats.linregress(x=c2[:2], y=c2[2:])

    pH_min_ = [arg_a0[0] * i_fluoro_meas[0] + arg_a0[1], arg_a1[0] * i_fluoro_meas[0] + arg_a1[1],
               arg_a2[0] * i_fluoro_meas[0] + arg_a2[1], arg_a0[0] * i_fluoro_meas[1] + arg_a0[1],
               arg_a1[0] * i_fluoro_meas[1] + arg_a1[1], arg_a2[0] * i_fluoro_meas[1] + arg_a2[1],
               arg_a0[0] * i_fluoro_meas[2] + arg_a0[1], arg_a1[0] * i_fluoro_meas[2] + arg_a1[1],
               arg_a2[0] * i_fluoro_meas[2] + arg_a2[1]]

    pH_mean_ = [arg_b0[0] * i_fluoro_meas[0] + arg_b0[1], arg_b1[0] * i_fluoro_meas[0] + arg_b1[1],
                arg_b2[0] * i_fluoro_meas[0] + arg_b2[1], arg_b0[0] * i_fluoro_meas[1] + arg_b0[1],
                arg_b1[0] * i_fluoro_meas[1] + arg_b1[1], arg_b2[0] * i_fluoro_meas[1] + arg_b2[1],
                arg_b0[0] * i_fluoro_meas[2] + arg_b0[1], arg_b1[0] * i_fluoro_meas[2] + arg_b1[1],
                arg_b2[0] * i_fluoro_meas[2] + arg_b2[1]]

    pH_max_ = [arg_c0[0] * i_fluoro_meas[0] + arg_c0[1], arg_c1[0] * i_fluoro_meas[0] + arg_c1[1],
              arg_c2[0] * i_fluoro_meas[0] + arg_c2[1], arg_c0[0] * i_fluoro_meas[1] + arg_c0[1],
              arg_c1[0] * i_fluoro_meas[1] + arg_c1[1], arg_c2[0] * i_fluoro_meas[1] + arg_c2[1],
              arg_c0[0] * i_fluoro_meas[2] + arg_c0[1], arg_c1[0] * i_fluoro_meas[2] + arg_c1[1],
              arg_c2[0] * i_fluoro_meas[2] + arg_c2[1]]

    # -----------------------------------------------------------------------------------------------
    # re-check plausibility of pH calculated
    pH_min = []
    for i in range(len(pH_min_)):
        if pH_min_[i] < 0 or pH_min_[i] > 14.:
            pass
        else:
            pH_min.append(pH_min_[i])

    pH_max = []
    for i in range(len(pH_max_)):
        if pH_max_[i] < 0 or pH_max_[i] > 14.:
            pass
        else:
            pH_max.append(pH_max_[i])

    pH_mean = []
    for i in range(len(pH_mean_)):
        if pH_mean_[i] < 0 or pH_mean_[i] > 14.:
            pass
        else:
            pH_mean.append(pH_mean_[i])

    if type_ == 'moderate' or type_ == 'optimistic':
        pH = np.array([np.array(pH_min).mean(), np.array(pH_mean).mean(), np.array(pH_max).mean()])
    elif type_ == 'pessimistic':
        mean_ = np.array(pH_min + pH_mean + pH_max).mean()
        std_ = np.array(pH_min + pH_mean + pH_max).std()
        pH = np.array([mean_ - std_, mean_, mean_ + std_])
    else:
        raise ValueError('Choose type of error propagation (optimistic / moderate or pessimistic)')

    for i, j in enumerate(pH):
        if np.isnan(j) == True:
            if i != 2:
                pH[i] = 0.
            else:
                pH[i] = 14.

    return pH.mean(), pH.std(), pH


def pH_selection(pH_calc0, pH_calc1, pH_calc2, type_='moderate'):
    pH_all = []
    for i in pH_calc0:
        if np.isnan(i) == False:
            pH_all.append(i)
    for i in pH_calc1:
        if np.isnan(i) == False:
            pH_all.append(i)
    for i in pH_calc2:
        if np.isnan(i) == False:
            pH_all.append(i)

    if type_ == 'moderate':
        pH_calc = [pH_calc0[1], pH_calc1[1], pH_calc2[1]]
    elif type_ == 'optimistic':
        pH_calc = [max(pH_calc0), pH_calc1[1], min(pH_calc2)]
    else:
        pH_calc = [min(pH_calc0), pH_calc1[1], max(pH_calc2)]

    return pH_calc


def regression_boltzmann_to_pO2(pO2_calc, x, fit_boltzmann_c0, fit_boltzmann_c1):

    # linear regression of the top values
    [slope_top0, intercept_top0, r_value, p_value, std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['top'][0],
                                                                                       fit_boltzmann_c1['top'][0]])
    [slope_top1, intercept_top1, r_value, p_value, std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['top'][1],
                                                                                       fit_boltzmann_c1['top'][1]])
    [slope_top2, intercept_top2, r_value, p_value, std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['top'][2],
                                                                                       fit_boltzmann_c1['top'][2]])
    # linear regression of the bottom values
    [slope_bottom0, intercept_bottom0, r_value, p_value,
     std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['bottom'][0], fit_boltzmann_c1['bottom'][0]])
    [slope_bottom1, intercept_bottom1, r_value, p_value,
     std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['bottom'][1], fit_boltzmann_c1['bottom'][1]])
    [slope_bottom2, intercept_bottom2, r_value, p_value,
     std_err] = stats.linregress(x=x, y=[fit_boltzmann_c0['bottom'][2], fit_boltzmann_c1['bottom'][2]])

    # ----------------------------
    # boltzmann parameters at calculated pO2 measurement point
    # top values
    top_meas0 = slope_top0 * pO2_calc[0] + intercept_top0
    top_meas1 = slope_top1 * pO2_calc[0] + intercept_top1
    top_meas2 = slope_top2 * pO2_calc[0] + intercept_top2

    top_meas3 = slope_top0 * pO2_calc[1] + intercept_top0
    top_meas4 = slope_top1 * pO2_calc[1] + intercept_top1
    top_meas5 = slope_top2 * pO2_calc[1] + intercept_top2

    top_meas6 = slope_top0 * pO2_calc[2] + intercept_top0
    top_meas7 = slope_top1 * pO2_calc[2] + intercept_top1
    top_meas8 = slope_top2 * pO2_calc[2] + intercept_top2

    top_meas_all = [top_meas0, top_meas1, top_meas2, top_meas3, top_meas4, top_meas5, top_meas6, top_meas7, top_meas8]
    top_meas = [min(top_meas_all), sum(top_meas_all) / np.float(len(top_meas_all)), max(top_meas_all)]

    # ----------------------------
    # bottom values
    bottom_meas0 = slope_bottom0 * pO2_calc[0] + intercept_bottom0
    bottom_meas1 = slope_bottom1 * pO2_calc[0] + intercept_bottom1
    bottom_meas2 = slope_bottom2 * pO2_calc[0] + intercept_bottom2

    bottom_meas3 = slope_bottom0 * pO2_calc[1] + intercept_bottom0
    bottom_meas4 = slope_bottom1 * pO2_calc[1] + intercept_bottom1
    bottom_meas5 = slope_bottom2 * pO2_calc[1] + intercept_bottom2

    bottom_meas6 = slope_bottom0 * pO2_calc[2] + intercept_bottom0
    bottom_meas7 = slope_bottom1 * pO2_calc[2] + intercept_bottom1
    bottom_meas8 = slope_bottom2 * pO2_calc[2] + intercept_bottom2

    bottom_meas_all = [bottom_meas0, bottom_meas1, bottom_meas2, bottom_meas3, bottom_meas4, bottom_meas5, bottom_meas6,
                       bottom_meas7, bottom_meas8]
    bottom_meas = [min(bottom_meas_all), sum(bottom_meas_all) / np.float(len(bottom_meas_all)), max(bottom_meas_all)]

    return top_meas, bottom_meas


# ---------------------------------------------------------------------------------------------------------------------
# Simulation INPUT pO2, pH
# ---------------------------------------------------------------------------------------------------------------------
def simulate_phaseangle_pO2_pH(pO2_range, pO2_calib, ox_meas, curv_O2, prop_ksv, K_sv1, tau_phos0, df_conv_tau_int,
                               int_phosphor_c0, pH_range, pH_calib, pH_meas, pk_a, slope, int_fluoro_max, f1, f2,
                               er_phase, plotting=True, normalize_phosphor=False, fontsize_=13, decimal=4):
    # calibration
    # determine lifetime of phosphorescent according to given pO2-level
    tau_quot, tauP, intP = twoSiteModel_fit(pO2_range=pO2_range, f=curv_O2, m=prop_ksv, Ksv=K_sv1, tau_phos0=tau_phos0,
                                            conv_tau_int=df_conv_tau_int, int_phos0=int_phosphor_c0, plotting=plotting,
                                            pO2_calib=pO2_calib, normalized_int=normalize_phosphor)

    # determine i_f according to given pH value
    df_i_f = boltzmann_sigmoid_int(pH_range=pH_range, v50=pk_a, slope=slope, int_f_max=int_fluoro_max)
    if plotting is True:
        plotting_boltzmann_sigmoid(pH_range=pH_range, pH_calib=pH_calib, int_fluoro=df_i_f, color='navy', type_='int',
                                   ax=None, fontsize_=fontsize_)

    para_simulation = pd.Series({'tau_quot': tau_quot, 'tauP': tauP, 'intP': intP, 'intF': df_i_f})

    # -------------------------------------------------------------------------------------------
    # discrete values for calibration and simulation point - find closest values for simulation points
    # -------------------------------------
    # fitting intensity of the fluorophor
    iF_0, iF_1, iF_meas = af.fitting_to_measpoint(index_=df_i_f.index, data_df=df_i_f[0], value_meas=pH_meas,
                                                  value_cal0=pH_calib[0], value_cal1=pH_calib[1])
    # -------------------------------------
    # fitting intensity of the phosphor
    iP_0, iP_1, iP_meas = af.fitting_to_measpoint(index_=intP.index, data_df=intP, value_meas=ox_meas,
                                                  value_cal0=pO2_calib[0], value_cal1=pO2_calib[1])

    # -------------------------------------
    # fitting lifetime of the phosphor
    tauP_0, tauP_1, tauP_meas = af.fitting_to_measpoint(index_=tauP.index, data_df=tauP, value_meas=ox_meas,
                                                        value_cal0=pO2_calib[0], value_cal1=pO2_calib[1])

    intP_discret = pd.Series({'phosphor0': iP_0, 'phosphor1': iP_1, 'meas': iP_meas})
    tauP_discret = pd.Series({'phosphor0': tauP_0, 'phosphor1': tauP_1, 'meas': tauP_meas})
    i_f_discret = pd.Series({'fluoro0': iF_0, 'fluoro1': iF_1, 'meas': iF_meas})

    int_ratio = pd.Series({'fluoro0, phosphor0': i_f_discret['fluoro0'] / intP_discret['phosphor0'],
                           'fluoro1, phosphor0': i_f_discret['fluoro1'] / intP_discret['phosphor0'],
                           'fluoro0, phosphor1': i_f_discret['fluoro0'] / intP_discret['phosphor1'],
                           'fluoro1, phosphor1': i_f_discret['fluoro1'] / intP_discret['phosphor1'],
                           'meas': i_f_discret['meas'] / intP_discret['meas']})

    # -------------------------------------------------------------------
    print('pO2 for simulation: {:.2f} hPa'.format(ox_meas))
    print('pH for simulation: {:.2f}'.format(pH_meas))
    # -------------------------------------------------------------------
    # amplitude ratio at 2 different modulation frequencies
    ampl_ratio_f1, ampl_ratio_f2 = af.amplitude_ratio(intP_discret=intP_discret, tauP_discret=tauP_discret,
                                                      i_f_discret=i_f_discret, f1=f1, f2=f2)

    Phi_f1_deg, Phi_f1_deg_er = af.superimposed_phaseangle_er(tauP_discret=tauP_discret, ampl_ratio=ampl_ratio_f1, f=f1,
                                                              er_phase=er_phase, decimal=decimal)
    Phi_f2_deg, Phi_f2_deg_er = af.superimposed_phaseangle_er(tauP_discret=tauP_discret, ampl_ratio=ampl_ratio_f2, f=f2,
                                                              er_phase=er_phase, decimal=decimal)

    return Phi_f1_deg, Phi_f2_deg, Phi_f1_deg_er, Phi_f2_deg_er, int_ratio, para_simulation


# ---------------------------------------------------------------------------------------------------------------------
# Simulation pO2, pH dualsensing
# ---------------------------------------------------------------------------------------------------------------------
# Individual pO2 sensing
def oxygen_sensing(pO2_range, pO2_calib, curv_O2, prop_ksv, Phi_f1_deg, Phi_f2_deg, phi_f1_meas, phi_f2_meas,
                   error_phaseangle, f1, f2, method_):

    # phosphorescence lifetime from 2-Frequency measurement
    [tau_c0, tau_c1, Phi_f1_deg_er, Phi_f2_deg_er,
     dev_tau] = af.preparation_lifetime(phi_f1_deg=Phi_f1_deg, phi_f2_deg=Phi_f2_deg, err_phase=error_phaseangle,
                                        f1=f1, f2=f2, method_=method_, er=True)
    [tau_meas, Phi_f1_meas_er,
     Phi_f2_meas_er] = af.phi_to_lifetime_including_error(phi_f1=phi_f1_meas, f1=f1, f2=f2, er=True, phi_f2=phi_f2_meas,
                                                          err_phaseangle=error_phaseangle)
    Phi_f1_deg_er = Phi_f1_deg_er.append(pd.Series({'meas': np.rad2deg(Phi_f1_meas_er)}))
    Phi_f2_deg_er = Phi_f2_deg_er.append(pd.Series({'meas': np.rad2deg(Phi_f2_meas_er)}))
    tau = pd.Series({'phosphor0': tau_c0, 'phosphor1': tau_c1, 'meas': tau_meas})

    # ----------------------------------
    print('Deviation of lifetimes calculated: pO2(0) ~ {:.2e}, pO2(1) ~ {:.2e}'.format(dev_tau['tau_c0'],
                                                                                       dev_tau['tau_c1']))

    # ---------------------------------------------------------------------------------
    para_TSM = twoSiteModel_calibration(tau0=tau_c0, tau1=tau_c1, m=prop_ksv, f=curv_O2, pO2_calib1=pO2_calib[1])

    # Calculation of pO2 at measurement point
    pO2_calc = twoSiteModel_evaluation(tau0=tau_c0, tau=tau_meas, m=prop_ksv, f=curv_O2, ksv=para_TSM['Ksv_fit1'],
                                       pO2_range=pO2_range)

    # ---------------------------------------------------------------------------------
    print('Calculated pO2: {:.2f} ± {:.2e} hPa'.format(pO2_calc.mean(), pO2_calc.std()))

    return pO2_calc, para_TSM, tau, Phi_f1_deg_er, Phi_f2_deg_er


# Individual pH sensing
def pH_sensing(pH_range, pH_calib, pk_a, slope, pO2_calib, pO2_calc, i_fluoro, type_='moderate', plotting=True,
               fontsize_=13):

    # fit parameter at both calibration points for the phosphor
    fit_parameter_c0 = boltzmann_fit_calibration_intensity(pH0=pH_calib[0], pH1=pH_calib[1], fontsize_=13,
                                                           int_f0=i_fluoro['fluoro0, phosphor0'], plot=False,
                                                           int_f1=i_fluoro['fluoro1, phosphor0'], v50=pk_a, slope=slope)

    fit_parameter_c1 = boltzmann_fit_calibration_intensity(pH0=pH_calib[0], pH1=pH_calib[1], fontsize_=13,
                                                           int_f0=i_fluoro['fluoro0, phosphor1'], plot=False,
                                                           int_f1=i_fluoro['fluoro1, phosphor1'], v50=pk_a, slope=slope)

    i_f_fit_c0 = boltzmann_sigmoid_int(pH_range=pH_range, v50=pk_a, slope=slope,
                                       int_f_max=fit_parameter_c0['int_fluoro'].loc[0].values)
    i_f_fit_c1 = boltzmann_sigmoid_int(pH_range=pH_range, v50=pk_a, slope=slope,
                                       int_f_max=fit_parameter_c1['int_fluoro'].loc[0].values)

    # -------------------------------------------------------------------
    # linear regression
    i_f_max_meas = linreg_intensity_to_pO2(pO2_calc=pO2_calc, x=pO2_calib, i_f_c0=i_f_fit_c0.loc[0],
                                           i_f_c1=i_f_fit_c1.loc[0])

    i_f_fit_meas = boltzmann_sigmoid_int(pH_range=pH_range, v50=pk_a, slope=slope, int_f_max=i_f_max_meas.loc[0])

    # -------------------------------------------------------------------
    # pH calculation --> find closest value for fluorescence intensity
    ind_meas = []
    ls_pH_mean = {}
    ls_pH_std = {}
    ls_pH = {}
    for i in i_fluoro.index:
        if 'meas' in i:
            ind_meas.append(i)
    for ind in ind_meas:
        pH_mean, pH_std, pH = pH_calculation(i_f_fit=i_f_fit_meas, i_fluoro_meas=i_fluoro[ind], type_=type_)
        ls_pH_mean[ind] = pH_mean
        ls_pH_std[ind] = pH_std
        ls_pH[ind] = pH

    # -------------------------------------------------------------------
    # plotting
    if plotting is True:
        ax = plotting_boltzmann_sigmoid(pH_range=pH_range, pH_calib=pH_calib, int_fluoro=i_f_fit_meas,
                                        color=color_pH[1], type_='int', fontsize_=fontsize_)
    print('Calculated pH: {:.2f} ± {:.2e}'.format(pH_mean, pH_std))

    return pH, i_f_fit_meas


# --------------------------------------------------------
# Dual-sensing
def pH_oxygen_dualsensor(phi_f1_deg, phi_f2_deg, phi_f1_meas, phi_f2_meas, error_phaseangle, pO2_range, pO2_calib,
                         curv_O2, prop_ksv, df_conv_tau_int, intP_c0, pH_range, pH_calib, v50, slope_pH, f1,
                         f2, plotting=True, fontsize_=13, type_='moderate', method_='std'):
    # phosphorescence sensor
    [pO2_calc, para_TSM, tau, Phi_f1_deg_er,
     Phi_f2_deg_er] = oxygen_sensing(pO2_range=pO2_range, pO2_calib=pO2_calib, curv_O2=curv_O2, prop_ksv=prop_ksv,
                                     Phi_f1_deg=phi_f1_deg, phi_f1_meas=phi_f1_meas, Phi_f2_deg=phi_f2_deg,
                                     method_=method_, phi_f2_meas=phi_f2_meas, error_phaseangle=error_phaseangle,
                                     f1=f1, f2=f2)

    # From lifetime to intensity of the phosphorescent
    [tau_quot, tauP, intP] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM['slope'], m=para_TSM['prop Ksv'],
                                              Ksv=para_TSM['Ksv_fit1'], tau_phos0=tau['phosphor0'],
                                              conv_tau_int=df_conv_tau_int, int_phos0=intP_c0, plotting=False,
                                              pO2_calib=pO2_calib, normalized_int=False)

    # ------------------------------------------------------------------------------------
    # intensity ratio via equation
    int_ratio = af.intensity_ratio_calculation(f1=f1, f2=f2, tau=tau, phi_f1_deg=Phi_f1_deg_er,
                                               phi_f2_deg=Phi_f2_deg_er)

    # conversion intensity ratio into total amplitude based on fitted intP
    ampl_f1, ampl_f2, intP_max_calc = af.int_ratio_to_ampl(f1=f1, f2=f2, pO2_calib=pO2_calib, pO2_calc=pO2_calc,
                                                           int_ratio=int_ratio, intP=intP, tauP=tauP)

    # intensity based conversion - i_f = int_ratio * i_p - find closest value
    intF = af.intP_to_intF(intP=intP, calibP=pO2_calib, calcP=pO2_calc, int_ratio=int_ratio)

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    [pH_all, i_f_fit_meas] = pH_sensing(pH_range=pH_range, pH_calib=pH_calib, pk_a=v50, slope=slope_pH, type_=type_,
                                        pO2_calib=pO2_calib, pO2_calc=pO2_calc, i_fluoro=intF, plotting=False,
                                        fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pH = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                            pH_range=pH_range, pH_calib=pH_calib, int_fluoro=i_f_fit_meas,
                                            pH_calc=pH_all, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pH = None

    return pO2_calc, tau_quot, tauP, intP, pH_all, i_f_fit_meas, para_TSM, ampl_f1, ampl_f2, ax_pO2, ax_pH


def pH_oxygen_dualsensor_meas(phi_f1_deg, phi_f2_deg, phi_f1_meas, phi_f2_meas, error_phaseangle, pO2_range,
                              pO2_calib, curv_O2, prop_ksv, df_conv_tau_int, intP_c0, pH_range, pH_calib, v50, slope_pH,
                              f1, f2, ampl_total_f1=None, ampl_total_f2=None, int_ratio=None, plotting=True,
                              fontsize_=13, type_='moderate', method_='std'):
    # pre-check: either int_ratio or total amplitude at two modulation frequencies are required!
    if int_ratio is None and ampl_total_f1 is None or int_ratio is None and ampl_total_f2 is None:
        raise ValueError('Either int_ratio or the total amplitude are required for evaluation!')
    else:
        pass

    # phosphorescence sensor
    ls_pO2_calc = {}
    ls_tau = {}
    ls_para_TSM = {}
    ls_tau_quot = {}
    ls_tauP = {}
    ls_intP = {}
    for i in range(len(phi_f1_meas)):
        [pO2_calc, para_TSM, tau, Phi_f1_deg_er,
        Phi_f2_deg_er] = oxygen_sensing(pO2_range=pO2_range, pO2_calib=pO2_calib, curv_O2=curv_O2, prop_ksv=prop_ksv,
                                        Phi_f1_deg=phi_f1_deg, phi_f1_meas=phi_f1_meas[i], Phi_f2_deg=phi_f2_deg,
                                        method_=method_, phi_f2_meas=phi_f2_meas[i], error_phaseangle=error_phaseangle,
                                        f1=f1, f2=f2)

        # From lifetime to intensity of the phosphorescent
        [tau_quot, tauP, intP] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM['slope'], m=para_TSM['prop Ksv'],
                                                  Ksv=para_TSM['Ksv_fit1'], tau_phos0=tau['phosphor0'],
                                                  conv_tau_int=df_conv_tau_int, int_phos0=intP_c0, plotting=False,
                                                  pO2_calib=pO2_calib, normalized_int=False)
        ls_pO2_calc['meas {}'.format(i)] = pO2_calc
        ls_tau[i] = tau
        ls_para_TSM[i] = para_TSM
        ls_tau_quot[i] = tau_quot
        ls_tauP[i] = tauP
        ls_intP[i] = intP

    # ------------------------------------------------------------------------------------
    # in case only the total amplitude is given -> conversion into intensity ratio
    if ampl_total_f1 is None:
        pass
    else:
        int_ratio_F0P0 = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau['phosphor0'],
                                              dphi_f1=phi_f1_deg['fluoro0, phosphor0'],
                                              dphi_f2=phi_f2_deg['fluoro0, phosphor0'])
        int_ratio_F1P0 = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau['phosphor0'],
                                              dphi_f1=phi_f1_deg['fluoro1, phosphor0'],
                                              dphi_f2=phi_f2_deg['fluoro1, phosphor0'])
        int_ratio_F0P1 = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau['phosphor1'],
                                              dphi_f1=phi_f1_deg['fluoro0, phosphor1'],
                                              dphi_f2=phi_f2_deg['fluoro0, phosphor1'])
        int_ratio_F1P1 = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau['phosphor1'],
                                              dphi_f1=phi_f1_deg['fluoro1, phosphor1'],
                                              dphi_f2=phi_f2_deg['fluoro1, phosphor1'])
        int_ratio_meas = {}
        for i in range(len(phi_f1_meas)):
            int_ratio_meas['meas {}'.format(i)] = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=ls_tau[i]['meas'],
                                                                       dphi_f1=phi_f1_meas[i], dphi_f2=phi_f2_meas[i])

        i_ratio = pd.Series({'fluoro0, phosphor0': int_ratio_F0P0, 'fluoro1, phosphor0': int_ratio_F1P0,
                             'fluoro0, phosphor1': int_ratio_F0P1, 'fluoro1, phosphor1': int_ratio_F1P1})
        int_ratio = pd.concat([i_ratio, int_ratio_meas], axis=0)

    # intensity based conversion - i_f = int_ratio * i_p - find closest value
    ls_intF = {}
    for i in range(len(phi_f1_meas)):
        print(i)
        intF = af.intP_to_intF(intP=ls_intP[i], calibP=pO2_calib, calcP=ls_pO2_calc['meas {}'.format(i)],
                               int_ratio=int_ratio)
        ls_intF[i] = intF

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    ls_pH = {}
    ls_If_fit_meas = {}
    for i in range(len(phi_f1_meas)):
        [pH_all, i_f_fit_meas] = pH_sensing(pH_range=pH_range, pH_calib=pH_calib, pk_a=v50, slope=slope_pH, type_=type_,
                                            pO2_calib=pO2_calib, pO2_calc=ls_pO2_calc['meas {}'.format(i)],
                                            i_fluoro=ls_intF[i], plotting=False,fontsize_=fontsize_)
        ls_pH[i] = pH_all
        ls_If_fit_meas[i] = i_f_fit_meas

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pH = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                            pH_range=pH_range, pH_calib=pH_calib, int_fluoro=i_f_fit_meas,
                                            pH_calc=pH_all, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pH = None

    return pd.Series(ls_pO2_calc), pd.Series(ls_tau_quot), pd.Series(ls_tauP), pd.Series(ls_intP), pd.Series(ls_pH), \
           pd.Series(ls_If_fit_meas), pd.Series(ls_para_TSM), ax_pO2, ax_pH

