__author__ = 'szieger'
__project__ = 'dualsensor CO2/O2 sensing'

import matplotlib
import additional_functions as af
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib import cm
from scipy import stats
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------------------------------------------------------------
def plotting_twoSiteModel(calib_point, df_tau_quot, df_int, analyt='pO2', calibration='1point', normalized=True,
                          fontsize_=13):
    if normalized is True:
        df_int_norm = df_int / df_int.loc[0]
    else:
        df_int_norm = df_int

    f, ax = plt.subplots()
    ax1 = ax.twinx()

    ax.plot(df_tau_quot.index, 1/df_tau_quot, color='navy', lw=1., label='tau0/tau')
    ax1.plot(df_int_norm, color='orange', lw=1., label='$I_P$')

    if calibration == '1point':
        ax.axvline(calib_point, color='k', lw=1., ls='--')
    else:
        ax.axvline(calib_point[0], color='k', lw=1., ls='--')
        ax.axvline(calib_point[1], color='k', lw=1., ls='--')

    # find closest value to calibration point
    if len(df_tau_quot.columns) == 3:
        data_quot = df_tau_quot[1].values
    else:
        data_quot = df_tau_quot.values

    if calibration == '1point':
        a = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=calib_point)
        b = None
    else:
        a = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=calib_point[0])
        b = af.find_closest_value_(index=df_tau_quot.index, data=data_quot, value=calib_point[1])

    if a[0] == a[1]:
        tauQ_c0 = a[2]
    else:
        arg_calib0 = stats.linregress(x=a[:2], y=a[2:])
        tauQ_c0 = arg_calib0[0] * calib_point[0] + arg_calib0[1]
    if np.isnan(tauQ_c0):
        y_c0 = df_tau_quot.loc[calib_point[0]].values[1]
    else:
        y_c0 = tauQ_c0
    ax.axhline(1/y_c0, color='k', lw=1., ls='--')

    if b is None:
        pass
    else:
        if b[0] == b[1]:
            tauQ_c1 = b[2]
        else:
            arg_calib1 = stats.linregress(x=b[:2], y=b[2:])
            tauQ_c1 = arg_calib1[0] * calib_point[1] + arg_calib1[1]
        if np.isnan(tauQ_c1):
            y_c1 = df_tau_quot.loc[calib_point[1]].values[1]
        else:
            y_c1 = tauQ_c1
        ax.axhline(1/y_c1, color='k', lw=1., ls='--')

    if analyt == 'pO2':
        ax.set_xlabel('p$O_2$ / hPa', fontsize=fontsize_)
        ax1.set_ylabel('Rel. Intensity I$_P$', color='orange', fontsize=fontsize_)
    else:
        ax.set_xlabel('p$CO_2$ / hPa', fontsize=fontsize_)
        ax1.set_ylabel('Rel. Intensity I$_F$', color='orange', fontsize=fontsize_)

    # x limits
    if df_tau_quot.index[0] <= 0:
        if df_tau_quot.index[0] == 0:
            xmin = -5.
        else:
            xmin = df_tau_quot.index[0] * 1.05
    else:
        xmin = df_tau_quot.index[0] * 0.95

    if df_tau_quot.index[-1] <= 0:
        if df_tau_quot.index[-1] == 0:
            xmax = -0.05
        else:
            xmax = df_tau_quot.index[-1] * 1.05
    else:
        xmax = df_tau_quot.index[-1] * 1.05
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('τ$_0$ / τ', color='navy', fontsize=fontsize_)

    plt.tight_layout()


def plotting_dualsensor(pO2_calib, pO2_calc, df_tau_quot, tauP_c0, pCO2_range, pCO2_calib, pCO2_calc, int_fluoro,
                        fontsize_=13):

    f, (ax_pO2, ax_pCO2) = plt.subplots(ncols=2, figsize=(7, 3))

    # ---------------------------------------------------------------------
    # pO2 sensing
    if isinstance(tauP_c0, np.float):
        tauP_c0 = tauP_c0 * 1E6
    else:
        tauP_c0 = tauP_c0[1] * 1E6

    if len(df_tau_quot.columns) == 3:
        ax_pO2.plot(df_tau_quot.index, tauP_c0*df_tau_quot[1], color='navy', lw=1., label='tau0/tau')
        ax_pO2.plot(tauP_c0*df_tau_quot[0], color='k', lw=.25, ls='--')
        ax_pO2.plot(tauP_c0*df_tau_quot[2], color='k', lw=.25, ls='--')
        ax_pO2.fill_between(df_tau_quot.index, tauP_c0*df_tau_quot[0], tauP_c0*df_tau_quot[2], color='grey', alpha=0.1,
                            lw=0.25)
    else:
        ax_pO2.plot(df_tau_quot.index, tauP_c0*df_tau_quot[0], color='navy', lw=1., label='tau0/tau')

    ax_pO2.set_xlabel('p$O_2$ / hPa', fontsize=fontsize_)
    ax_pO2.set_ylabel('τ [µs]', color='navy', fontsize=fontsize_)

    # calibration and measurement points
    if isinstance(pO2_calib, np.float):
        ax_pO2.axvline(pO2_calib, color='k', lw=1., ls='--')
    else:
        ax_pO2.axvline(pO2_calib[0], color='k', lw=1., ls='--')
        ax_pO2.axvline(pO2_calib[1], color='k', lw=1., ls='--')
    ax_pO2.axvspan(pO2_calc[0], pO2_calc[2], color='#f0810f', alpha=0.4)

    # find closest values for calibration points
    if isinstance(pO2_calib, np.float):
        if len(df_tau_quot.columns) == 3:
            tauq_c1 = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[1].values, value=pO2_calib)
        else:
            tauq_c1 = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[0].values, value=pO2_calib)

        # averaging values
        if (tauq_c1[2:][0] == tauq_c1[2:][1]) == True:
            y_c1 = tauq_c1[2:][0]
        else:
            y_c1 = (tauq_c1[2:][0] + tauq_c1[2:][1]) / 2
        ax_pO2.axhline(y=y_c1, color='k', lw=1., ls='--')
    else:
        tauq_c0 = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[1].values, value=pO2_calib[0])
        tauq_c1 = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[1].values, value=pO2_calib[1])

        # linear regression to pO2 measured
        arg_c0 = stats.linregress(x=pO2_calib, y=tauq_c0[2:])
        arg_c1 = stats.linregress(x=pO2_calib, y=tauq_c1[2:])
        y_c0 = tauP_c0*(arg_c0[0]*pO2_calib[0] + arg_c0[1])
        y_c1 = tauP_c0*(arg_c1[0]*pO2_calib[1] + arg_c1[1])
        ax_pO2.axhline(y=y_c0, color='k', lw=1., ls='--')
        ax_pO2.axhline(y=y_c1, color='k', lw=1., ls='--')

    # find closest value for measurement point
    if len(df_tau_quot.columns) == 3:
        tauq_meas_min = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[0].values, value=pO2_calc[0])
        tauq_meas_max = af.find_closest_value_(index=df_tau_quot.index, data=df_tau_quot[2].values, value=pO2_calc[2])

        # linear regression to pO2 measured or averaging values when 1-Point-Calibration is chosen
        if isinstance(pO2_calib, np.float):
            if (tauq_meas_min[2:][0] == tauq_meas_min[2:][1]) == True:
                y_min = tauq_meas_min[2:][0]
            else:
                y_min = (tauq_meas_min[2:][0] + tauq_meas_min[2:][1]) / 2
            if (tauq_meas_max[2:][0] == tauq_meas_max[2:][1]) == True:
                y_max = tauq_meas_max[2:][0]
            else:
                y_max = (tauq_meas_max[2:][0] + tauq_meas_max[2:][1]) / 2
        else:
            arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
            arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
            y_min = tauP_c0*(arg_min[0]*pO2_calc[0] + arg_min[1])
            y_max = tauP_c0*(arg_max[0]*pO2_calc[2] + arg_max[1])

        ax_pO2.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

    # ---------------------------------------------------------------------
    # pCO2 sensing
    ax_pCO2.plot(pCO2_range, int_fluoro['mean'], color='#07575B', lw=1.)
    ax_pCO2.plot(pCO2_range, int_fluoro['min'], color='k', ls='--', lw=.5)
    ax_pCO2.plot(pCO2_range, int_fluoro['max'], color='k', ls='--', lw=.5)
    ax_pCO2.fill_between(pCO2_range, int_fluoro['min'], int_fluoro['max'], color='grey', alpha=0.1, lw=0.25)

    # calibration and measurement points
    ax_pCO2.axvline(pCO2_calib[0], color='k', ls='--', lw=1.)
    ax_pCO2.axvline(pCO2_calib[1], color='k', ls='--', lw=1.)
    ax_pCO2.axvspan(pCO2_calc[0], pCO2_calc[2], color='#f0810f', alpha=0.4)

    # find closest values
    iF_calib0 = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pCO2_calib[0])
    iF_calib1 = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pCO2_calib[1])
    iF_meas_min = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pCO2_calc[0])
    iF_meas_max = af.find_closest_value_(index=int_fluoro.index, data=int_fluoro['mean'].values, value=pCO2_calc[2])

    # linear regression to pH measured
    arg_min = stats.linregress(x=pCO2_calib, y=iF_meas_min[2:])
    arg_max = stats.linregress(x=pCO2_calib, y=iF_meas_max[2:])
    y_min = arg_min[0]*pCO2_calc[0] + arg_min[1]
    y_max = arg_max[0]*pCO2_calc[2] + arg_max[1]

    ax_pCO2.axhline(y=sum(iF_calib0[2:])/2, color='k', lw=1., ls='--')
    ax_pCO2.axhline(y=sum(iF_calib1[2:])/2, color='k', lw=1., ls='--')
    ax_pCO2.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

    # ax_pCO2.set_ylim(50, 120)
    ax_pCO2.set_xlabel('pCO2 / hPa', fontsize=fontsize_)
    ax_pCO2.set_ylabel('Rel. Intensity $I_f$', color='#07575B', fontsize=fontsize_)

    plt.tight_layout()

    return ax_pO2, ax_pCO2


# --------------------------------------------------------------------------------------------------------------------
# Additional functions
# --------------------------------------------------------------------------------------------------------------------
# Helping hands
def reduced_euqation_pCO2(linear, abscissa, int_quot):
    if isinstance(linear, np.float):
        if isinstance(abscissa, np.float):
            pCO2 = [(i - abscissa) / linear for i in int_quot]
        else:
            pCO2 = [(i - abs) / linear for (i, abs) in zip(int_quot, abscissa)]
    else:
        if isinstance(abscissa, np.float):
            pCO2 = [(i - abscissa) / lin for (i, lin) in zip(int_quot, linear)]
        else:
            pCO2 = [(i - abs) / lin for (i, abs, lin) in zip(int_quot, abscissa, linear)]
    return pCO2


def pq_equation_pCO2(quadratic, linear, abscissa, int_quot):
    """
    pq-equation for quadratic equation y = y0 / (a*x**2 + b*x + c)
    :param quadratic:
    :param linear:
    :param abscissa:
    :param int_quot:
    :return:
    """
    # preparation
    a = quadratic
    b = linear
    if isinstance(abscissa, np.float):
        c = [abscissa - i for i in int_quot]
    else:
        c = [abs - i for (abs, i) in zip(abscissa, int_quot)]

    if isinstance(b, np.float):
        if isinstance(a, np.float):
            sqrt = [np.sqrt(b**2 - 4*a*c_) for c_ in c]
            n = 2*a
        else:
            sqrt = [np.sqrt(b**2 - 4*a_*c_) for (a_, c_) in zip(a, c)]
            n = [2*a_ for a_ in a]
        z1 = [-1*b - sqrt_ for sqrt_ in sqrt]
        z2 = [-1*b + sqrt_ for sqrt_ in sqrt]
    else:
        if isinstance(a, np.float):
            sqrt = [np.sqrt(b_**2 - 4*a*c_) for (b_, c_) in zip(b, c)]
            n = 2*a
        else:
            sqrt = [np.sqrt(b_**2 - 4*a_*c_) for (a_, b_, c_) in zip(a, b, c)]
            n = [2*a_ for a_ in a]
        z1 = [-1*b_ - sqrt_ for (b_, sqrt_) in zip(b, sqrt)]
        z2 = [-1*b_ + sqrt_ for (b_, sqrt_) in zip(b, sqrt)]

    # Division of (de-)nominator
    if isinstance(a, np.float):
        x1 = [z_ / n for z_ in z1]
        x2 = [z_ / n for z_ in z2]
    else:
        x1 = [z_ / n_ for (z_, n_) in zip(z1, n)]
        x2 = [z_ / n_ for (z_, n_) in zip(z2, n)]

    return x1, x2


# --------------------------------------------------------------
def twoSiteModel_calib_ksv(tau0, tau1, m, f, pO2_calib1):
    # preparation
    if isinstance(tau0, np.float):
        tau_quot = [x1/tau0 for x1 in tau1]
    else:
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
        print(ksv_1, ksv_2)
        raise ValueError('decide about Ksv')
        # Ksv_fit1 = ksv_1.append(ksv_2)

    # combining all (fit) parameter for two-site-model
    para_TSM = pd.Series({'tauP0': tau0, 'tauP1': tau1, 'prop Ksv': m, 'slope': f, 'Ksv_fit1': Ksv_fit1,
                          'Ksv_fit2': [k*m for k in Ksv_fit1]})

    return para_TSM


def twoSiteModel_calib_tauP0(ksv, tau1, m, f, pO2_calib1):
    # preparation
    quot = f / (1 + ksv*pO2_calib1) + (1-f) / (1 + ksv*pO2_calib1*m)

    if isinstance(tau1, np.float):
        tau0 = tau1 / quot
    else:
        tau0 = [x1 / quot for x1 in tau1]

    # combining all (fit) parameter for two-site-model
    para_TSM = pd.Series({'tauP0': tau0, 'tauP1': tau1, 'prop Ksv': m, 'slope': f, 'Ksv_fit1': ksv,
                          'Ksv_fit2': ksv*m})

    return para_TSM


def twoSiteModel_evaluation(tau0, tau, m, f, ksv, pO2_range):
    # preparation pq equation
    if isinstance(tau0, np.float):
        if isinstance(tau, np.float):
            quot = tau / tau0
        else:
            quot = [t1/tau0 for t1 in tau]
    else:
        if isinstance(tau, np.float):
            quot = [tau / t0 for t0 in tau0]
        else:
            quot = [t1/t0 for (t1, t0) in zip(tau, tau0)]

    c = [1-1/q for q in quot]
    if isinstance(ksv, np.float):
        b = [ksv*(m*(1-f/t) + 1/t*(f-1)+1) for t in quot]
        a = m*(ksv**2)
    else:
        b = [k*(m*(1-f/t) + 1/t*(f-1)+1) for (k, t) in zip(ksv, quot)]
        a = [m*(p**2) for p in ksv]

    if isinstance(a, np.float):
        sqrt = [np.sqrt(b_**2 - 4*a*c_) for (b_, c_) in zip(b, c)]
        n = 2*a
    else:
        sqrt = [np.sqrt(b_**2 - 4*a_*c_) for (b_, a_, c_) in zip(b, a, c)]
        n = [2*a_ for a_ in a]
    z1 = [-1*b_ + s for (b_, s) in zip(b, sqrt)]
    z2 = [-1*b_ - s for (b_, s) in zip(b, sqrt)]

    if isinstance(n, np.float):
        pO2_1 = sorted([z / n for z in z1])
        pO2_2 = sorted([z / n for z in z2])
    else:
        pO2_1 = sorted([z / n for (z, n) in zip(z1, n)])
        pO2_2 = sorted([z / n for (z, n) in zip(z2, n)])

    # select reasonable pO2 value
    if pO2_1[1] < 0 or pO2_1[1] > pO2_range[-1]:
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


def twoSiteModel_fit(pO2_range, f, m, Ksv, tau_phos0, conv_tau_int, int_phos0, plotting=True, pO2_calib=None,
                     normalized_int=True, calibration='1point', fontsize_=13):

    if isinstance(Ksv, np.float):
        # tau/tau0
        tau_quot = (f / (1 + Ksv*pO2_range)) + (1 - f) / (1 + Ksv*m*pO2_range)
        if isinstance(tau_phos0, np.float):
            tauP = tau_quot * tau_phos0
        else:
            tauP = [tau_quot * t for t in tau_phos0]
        # conversion into dataFrame
        tau_quot = pd.DataFrame(tau_quot, index=pO2_range)
    else:
        # tau/tau0
        tau_quot = [(f / (1 + p*pO2_range)) + (1 - f) / (1 + p*m*pO2_range) for p in Ksv]
        if isinstance(tau_phos0, np.float):
            tauP = [t_quot * tau_phos0 for t_quot in tau_quot]
        else:
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
            tauP = pd.DataFrame(tauP)
        else:
            if len(pd.DataFrame(tauP).columns) == 1:
                tauP = pd.DataFrame(tauP)*1E-6
            else:
                tauP = pd.DataFrame(tauP).T*1E-6
    tauP.index = pO2_range

    # pre-check if each index of df_tau is in df_conv_tau_int
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
    intP_norm = intP/intP.max()

    # plotting lifetime quotient and normalized intensity
    if plotting is True:
        if pO2_calib is None:
            raise ValueError('pO2 calibration points are required')
        else:
            if isinstance(pO2_calib, np.float):
                pO2_calib = list([pO2_calib])
            plotting_twoSiteModel(calib_point=pO2_calib, df_tau_quot=tau_quot, df_int=intP, analyt='pO2',
                                  calibration=calibration, normalized=normalized_int, fontsize_=fontsize_)

    return tau_quot, tauP, intP, intP_norm


def pCO2_fit_tsm(x, pCO2_calib, f, m, Ksv, intF_max, plotting=True, normalize=True, fontsize_=13):
    if isinstance(Ksv, np.float):
        if isinstance(intF_max, np.float):
            intF_ = ((f / (1 + Ksv*x)) + (1 - f) / (1 + Ksv*m*x)) * intF_max
        else:
            intF_ = ((f / (1 + Ksv*x)) + (1 - f) / (1 + Ksv*m*x)) * intF_max[1]
        intF = pd.DataFrame(intF_, index=x)
    else:
        intF_ = [((f / (1 + p*x)) + (1 - f) / (1 + p*m*x)) * intF_max for p in Ksv]
        intF = pd.DataFrame(intF_, columns=x, index=['min', 'mean', 'max']).T

    intF_norm = intF / intF.max()

    if plotting is True:
        f, ax = plt.subplots()

        if normalize is True:
            ax.plot(x, intF_norm, color='navy', lw=1.)
            # find closest value to calibration point
            a = af.find_closest_value_(index=intF.index, data=intF_norm.values, value=pCO2_calib[0])
            b = af.find_closest_value_(index=intF.index, data=intF_norm.values, value=pCO2_calib[1])
        else:
            ax.plot(x, intF, color='navy', lw=1.)
            # find closest value to calibration point
            a = af.find_closest_value_(index=intF.index, data=intF.values, value=pCO2_calib[0])
            b = af.find_closest_value_(index=intF.index, data=intF.values, value=pCO2_calib[1])

        ax.axvline(pCO2_calib[0], lw=0.75, ls='--', color='k')
        ax.axvline(pCO2_calib[1], lw=0.75, ls='--', color='k')

        if a[0] == a[1]:
            y_c0 = a[2]
        else:
            arg_calib0 = stats.linregress(x=a[:2], y=a[2:])
            y_c0 = arg_calib0[0] * pCO2_calib[0] + arg_calib0[1]
        if b[0] == b[1]:
            y_c1 = b[2]
        else:
            arg_calib1 = stats.linregress(x=b[:2], y=b[2:])
            y_c1 = arg_calib1[0] * pCO2_calib[1] + arg_calib1[1]

        ax.axhline(y_c0, color='k', lw=1., ls='--')
        ax.axhline(y_c1, color='k', lw=1., ls='--')

        ax.set_xlabel('p$CO_2$ / hPa', fontsize=fontsize_)
        ax.set_ylabel('Rel. Intensity I$_F$', fontsize=fontsize_)

    return intF, intF_norm


def pCO2_fit_empiric(x, pCO2_calib, abs, linear_CO2, quadratic_CO2, intF_max, plotting=True, x_correction=True,
                     normalize=True, fontsize_=13):
    # pCO2_correction x_tilde = x - pCO2_calib0 to define intF0
    if x_correction is True:
        x_tilde = x - pCO2_calib[0]
    else:
        x_tilde = x

    if isinstance(linear_CO2, np.float):
        intF_ = intF_max / (abs + linear_CO2 * x_tilde + quadratic_CO2 * x_tilde**2)
        intF = pd.DataFrame(intF_, index=x)
    else:
        intF_ = [intF_max / (ab + li * x_tilde + q * x_tilde**2) for (ab, li, q) in zip(abs, linear_CO2, quadratic_CO2)]
        intF = pd.DataFrame(intF_, columns=x, index=['min', 'mean', 'max']).T

    intF_norm = intF / intF.max()

    if plotting is True:
        f, ax = plt.subplots()

        if normalize is True:
            if len(intF_norm.columns) == 3:
                ax.plot(x, intF_norm['mean'], color='navy', lw=1.)
                ax.fill_between(x, intF_norm['min'], intF_norm['max'], color='grey', alpha=0.2)

                # find closest value to calibration point
                a = af.find_closest_value_(index=intF.index, data=intF_norm['mean'].values, value=pCO2_calib[0])
                b = af.find_closest_value_(index=intF.index, data=intF_norm['mean'].values, value=pCO2_calib[1])
            else:
                ax.plot(x, intF_norm, color='navy', lw=1.)

                # find closest value to calibration point
                a = af.find_closest_value_(index=intF.index, data=intF_norm.values, value=pCO2_calib[0])
                b = af.find_closest_value_(index=intF.index, data=intF_norm.values, value=pCO2_calib[1])
        else:
            if len(intF.columns) == 3:
                ax.plot(x, intF['mean'], color='navy', lw=1.)
                ax.fill_between(x, intF['min'], intF['max'], color='grey', alpha=0.2)

                # find closest value to calibration point
                a = af.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2_calib[0])
                b = af.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2_calib[1])
            else:
                ax.plot(x, intF, color='navy', lw=1.)

                # find closest value to calibration point
                a = af.find_closest_value_(index=intF.index, data=intF.values, value=pCO2_calib[0])
                b = af.find_closest_value_(index=intF.index, data=intF.values, value=pCO2_calib[1])

        ax.axvline(pCO2_calib[0], lw=0.75, ls='--', color='k')
        ax.axvline(pCO2_calib[1], lw=0.75, ls='--', color='k')

        if a[0] == a[1]:
            y_c0 = a[2]
        else:
            arg_calib0 = stats.linregress(x=a[:2], y=a[2:])
            y_c0 = arg_calib0[0] * pCO2_calib[0] + arg_calib0[1]
        if b[0] == b[1]:
            y_c1 = b[2]
        else:
            arg_calib1 = stats.linregress(x=b[:2], y=b[2:])
            y_c1 = arg_calib1[0] * pCO2_calib[1] + arg_calib1[1]

        ax.axhline(y_c0, color='k', lw=1., ls='--')
        ax.axhline(y_c1, color='k', lw=1., ls='--')

        ax.set_xlabel('p$CO_2$ / hPa', fontsize=fontsize_)
        ax.set_ylabel('Rel. Intensity I$_F$', fontsize=fontsize_)

    return intF, intF_norm


def pCO2_calib_TSM(intF_max, intF_calib1, m, f, pCO2_calib1, pCO2_range):
    # preparation
    if isinstance(intF_max, np.float):
        int_quot = [i1 / intF_max for i1 in intF_calib1]

        # parts of pq equation
        a = [(pCO2_calib1**2)*(m*i) for i in int_quot]
        b = [pCO2_calib1*((m + 1)*i - (m*f - f + 1)) for i in int_quot]
        c = [i - 1 for i in int_quot]

        sqrt1 = [b_**2 for b_ in b]
        sqrt2 = [4*a_*c_ for (a_, c_) in zip(a, c)]

        z1 = [-1*b_ + np.sqrt(sqrt1_ - sqrt2_) for (b_, sqrt1_, sqrt2_) in zip(b, sqrt1, sqrt2)]
        z2 = [-1*b_ - np.sqrt(sqrt1_ - sqrt2_) for (b_, sqrt1_, sqrt2_) in zip(b, sqrt1, sqrt2)]
        n = [2*a_ for a_ in a]

        ksv_1 = [z_/n_ for (z_, n_) in zip(z1, n)]
        ksv_2 = [z_/n_ for (z_, n_) in zip(z2, n)]
    else:
        int_quot = [x1/x2 for (x1, x2) in zip(intF_calib1, intF_max)]

        # parts of pq equation
        a = [(pCO2_calib1**2)*(m*t) for t in int_quot]
        b = [pCO2_calib1*((m + 1)*t - (m*f - f + 1)) for t in int_quot]
        c = [t - 1 for t in int_quot]

        sqrt1 = [b_**2 for b_ in b]
        sqrt2 = [4*a_*c_ for (a_, c_) in zip(a, c)]

        z_ = [np.sqrt(s1 - s2) for (s1, s2) in zip(sqrt1, sqrt2)]
        z1 = [-1*b_ + z1_ for (b_, z1_) in zip(b, z_)]
        z2 = [-1*b_ - z1_ for (b_, z1_) in zip(b, z_)]
        n = [2*a_ for a_ in a]

        ksv_1 = [x/y for (x, y) in zip(z1, n)]
        ksv_2 = [x/y for (x, y) in zip(z2, n)]

    if len(ksv_1) == 1:
        ksv_t1 = ksv_1[0]
        ksv_t2 = ksv_2[0]
    else:
        ksv_t1 = ksv_1[1]
        ksv_t2 = ksv_2[1]

    if ksv_t1 < 0 and ksv_t2 >=0:
        Ksv_fit1 = ksv_2
    elif ksv_t1 >= 0 and ksv_t2 < 0:
        Ksv_fit1 = ksv_1
    else:
        raise ValueError('decide about Ksv')

    if isinstance(Ksv_fit1, np.float):
        if Ksv_fit1 < 0:
            Ksv_fit1 = 0.
        Ksv_fit2 = Ksv_fit1*m
    else:
        for i, j in enumerate(Ksv_fit1):
            if j < 0:
                Ksv_fit1[i] = 0.
        Ksv_fit2 = [k*m for k in Ksv_fit1]

    # -----------------------------------------------------------------------------
    # calculating maximal intensity at pCO2 = 0
    if isinstance(f, np.float):
        if isinstance(Ksv_fit1, np.float):
            if isinstance(intF_max, np.float):
                df_iF_ = (f / (1 + Ksv_fit1 * pCO2_range) + (1 - f) / (1 + Ksv_fit2 * pCO2_range)) * intF_max
            else:
                df_iF_ = [(f / (1 + Ksv_fit1 * pCO2_range) + (1 - f) / (1 + Ksv_fit2 * pCO2_range) * i)
                          for i in intF_max]
        else:
            if isinstance(intF_max, np.float):
                df_iF_ = [(f / (1 + k1 * pCO2_range) + (1 - f) / (1 + k2 * pCO2_range)) * intF_max
                          for (k1, k2) in zip(Ksv_fit1, Ksv_fit2)]
            else:
                df_iF_ = [(f / (1 + k1 * pCO2_range) + (1 - f) / (1 + k2 * pCO2_range) * i)
                          for (k1, k2, i) in zip(Ksv_fit1, Ksv_fit2, intF_max)]
    else:
        if isinstance(Ksv_fit1, np.float):
            if isinstance(intF_max, np.float):
                df_iF_ = [(f_ / (1 + Ksv_fit1 * pCO2_range) + (1 - f_) / (1 + Ksv_fit2 * pCO2_range)) * intF_max
                          for f_ in f]
            else:
                df_iF_ = [(f_ / (1 + Ksv_fit1 * pCO2_range) + (1 - f_) / (1 + Ksv_fit2 * pCO2_range)) * i
                          for (f_, i) in zip(f, intF_max)]
        else:
            if isinstance(intF_max, np.float):
                df_iF_ = [(f_ / (1 + k1 * pCO2_range) + (1 - f_) / (1 + k2 * pCO2_range)) * intF_max
                          for (f_, k1, k2) in zip(f, Ksv_fit1, Ksv_fit2)]
            else:
                df_iF_ = [(f_ / (1 + k1 * pCO2_range) + (1 - f_) / (1 + k2 * pCO2_range)) * i
                          for (f_, k1, k2, i) in zip(f, Ksv_fit1, Ksv_fit2, intF_max)]

    df_iF = pd.DataFrame(df_iF_, columns=pCO2_range, index=['min', 'mean', 'max']).T

    if len(Ksv_fit1) == 1:
        Ksv_fit1 = Ksv_fit1*3
    if len(Ksv_fit2) == 1:
        Ksv_fit2 = Ksv_fit2*3

    # -----------------------------------------------------------------------------
    # combining all (fit) parameter for two-site-model
    para_TSM = pd.Series({'prop Ksv': m, 'slope': f, 'Ksv_fit1': Ksv_fit1, 'Ksv_fit2': Ksv_fit2, 'intF': df_iF})

    return para_TSM


def pCO2_calib_empiric(intF_max, intF_calib0, intF_calib1, quadratic_CO2, pCO2_calib, pCO2_range, x_correction=True):
    # pCO2_correction x_tilde = x - pCO2_calib0 to define intF0
    if x_correction is True:
        x_tilde = pCO2_range - pCO2_calib[0]
        pCO2_calib = [pCO2_calib[0] - pCO2_calib[0], pCO2_calib[1] - pCO2_calib[0]]
    else:
        x_tilde = pCO2_range

    # preparation - relative intensity
    if isinstance(intF_max, np.float):
        y1 = [intF_max / i for i in intF_calib0]
        y2 = [intF_max / i for i in intF_calib1]
    else:
        y1 = [m / i for (m, i) in zip(intF_max, intF_calib0)]
        y2 = [m / i for (m, i) in zip(intF_max, intF_calib1)]

    # -----------------------------------------------------------------------------
    if x_correction is True:
        abscissa_CO2 = y1
        if isinstance(y2, np.float):
            if isinstance(quadratic_CO2, np.float):
                linear_CO2 = (y2 - quadratic_CO2 * pCO2_calib[1]**2 - 1) / pCO2_calib[1]
            else:
                linear_CO2 = [(y2 - quad * pCO2_calib[1]**2 - 1) / pCO2_calib[1] for quad in quadratic_CO2]
        else:
            if isinstance(quadratic_CO2, np.float):
                linear_CO2 = [(y2_ - quadratic_CO2 * pCO2_calib[1]**2 - 1) / pCO2_calib[1] for y2_ in y2]
            else:
                linear_CO2 = [(y2_ - quad * pCO2_calib[1]**2 - 1) / pCO2_calib[1]
                              for (y2_, quad) in zip(y2, quadratic_CO2)]

    else:
        abscissa_CO2 = 1.
        x2 = 1 / (pCO2_calib[1] * (pCO2_calib[1] - pCO2_calib[0]))
        x1 = 1 / (pCO2_calib[0] * (pCO2_calib[1] - pCO2_calib[0]))
        quadratic_CO2 = [(y2_ * x2 - y1_ * x1 + 1/ (pCO2_calib[0] * pCO2_calib[1])) for (y1_, y2_) in zip(y1, y2)]

        if isinstance(quadratic_CO2, np.float):
            linear_CO2 = [(y2_ - y1_) / (pCO2_calib[1] - pCO2_calib[0]) - quadratic_CO2 * (pCO2_calib[1] + pCO2_calib[0])
                          for (y1_, y2_) in zip(y1, y2)]
        else:
            linear_CO2 = [(y2_ - y1_) / (pCO2_calib[1] - pCO2_calib[0]) - quad * (pCO2_calib[1] + pCO2_calib[0])
                          for (y1_, y2_, quad) in zip(y1, y2, quadratic_CO2)]

    # -----------------------------------------------------------------------------
    # calculating maximal intensity at pCO2 = 0
    if isinstance(quadratic_CO2, np.float):
        if quadratic_CO2 == 0.:
            # reziprok linear y0 / (b*x + c) = y
            if isinstance(linear_CO2, np.float):
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = intF_max / (linear_CO2 * x_tilde + abscissa_CO2)
                    else:
                        intF = [i / (linear_CO2 * x_tilde + abscissa_CO2) for i in intF_max]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (linear_CO2 * x_tilde + abs) for abs in abscissa_CO2]
                    else:
                        intF = [i / (linear_CO2 * x_tilde + abs) for (i, abs) in zip(intF_max, abscissa_CO2)]
            else:
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (lin * x_tilde + abscissa_CO2) for lin in linear_CO2]
                    else:
                        intF = [i / (lin * x_tilde + abscissa_CO2) for (i, lin) in zip(intF_max, linear_CO2)]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (lin * x_tilde + abs) for (lin, abs) in zip(linear_CO2, abscissa_CO2)]
                    else:
                        intF = [i / (lin * x_tilde + abs)
                                for (i, lin, abs) in zip(intF_max, linear_CO2, abscissa_CO2)]
        else:
            # reziprok quadratic y0 / (a*x**2 + b*x + c) = y
            if isinstance(linear_CO2, np.float):
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = intF_max / (quadratic_CO2 * x_tilde**2 + linear_CO2 * x_tilde + abscissa_CO2)
                    else:
                        intF = [i / (quadratic_CO2 * x_tilde**2 + linear_CO2 * x_tilde + abscissa_CO2)
                                for i in intF_max]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quadratic_CO2 * x_tilde**2 + linear_CO2 * x_tilde + abs)
                                for abs in abscissa_CO2]
                    else:
                        intF = [i / (quadratic_CO2 * x_tilde**2 + linear_CO2 * x_tilde + abs)
                                for (i, abs) in zip(intF_max, abscissa_CO2)]
            else:
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quadratic_CO2 * x_tilde**2 + lin * x_tilde + abscissa_CO2)
                                for lin in linear_CO2]
                    else:
                        intF = [i / (quadratic_CO2 * x_tilde**2 + lin * x_tilde + abscissa_CO2)
                                for (i, lin) in zip(intF_max, linear_CO2)]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quadratic_CO2 * x_tilde**2 + lin * x_tilde + abs)
                                for (lin, abs) in zip(linear_CO2, abscissa_CO2)]
                    else:
                        intF = [i / (quadratic_CO2 * x_tilde**2 + lin * x_tilde + abs)
                                for (i, lin, abs) in zip(intF_max, linear_CO2, abscissa_CO2)]
    else:
        if quadratic_CO2[1] == 0.:
            # reziprok linear y0 / (b*x + c) = y
            if isinstance(linear_CO2, np.float):
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = intF_max / (linear_CO2 * x_tilde + abscissa_CO2)
                    else:
                        intF = [i / (linear_CO2 * x_tilde + abscissa_CO2) for i in intF_max]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (linear_CO2 * x_tilde + abs) for abs in abscissa_CO2]
                    else:
                        intF = [i / (linear_CO2 * x_tilde + abs) for (i, abs) in zip(intF_max, abscissa_CO2)]
            else:
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (lin * x_tilde + abscissa_CO2) for lin in linear_CO2]
                    else:
                        intF = [i / (lin * x_tilde + abscissa_CO2) for (i, lin) in zip(intF_max, linear_CO2)]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (lin * x_tilde + abs) for (lin, abs) in zip(linear_CO2, abscissa_CO2)]
                    else:
                        intF = [i / (lin * x_tilde + abs)
                                for (i, lin, abs) in zip(intF_max, linear_CO2, abscissa_CO2)]
        else:
            # reziprok quadratic y0 / (a*x**2 + b*x + c) = y
            if isinstance(linear_CO2, np.float):
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quad * x_tilde**2 + linear_CO2 * x_tilde + abscissa_CO2)
                                for quad in quadratic_CO2]
                    else:
                        intF = [i / (quad * x_tilde**2 + linear_CO2 * x_tilde + abscissa_CO2)
                                for (i, quad) in zip(intF_max, quadratic_CO2)]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quad * x_tilde**2 + linear_CO2 * x_tilde + abs)
                                for (quad, abs) in zip(quadratic_CO2, abscissa_CO2)]
                    else:
                        intF = [i / (quad * x_tilde**2 + linear_CO2 * x_tilde + abs)
                                for (i, quad, abs) in zip(intF_max, quadratic_CO2, abscissa_CO2)]
            else:
                if isinstance(abscissa_CO2, np.float):
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quad * x_tilde**2 + lin * x_tilde + abscissa_CO2)
                                for (quad, lin) in zip(quadratic_CO2, linear_CO2)]
                    else:
                        intF = [i / (quad * x_tilde**2 + lin * x_tilde + abscissa_CO2)
                                for (i, quad, lin) in zip(intF_max, quadratic_CO2, linear_CO2)]
                else:
                    if isinstance(intF_max, np.float):
                        intF = [intF_max / (quad * x_tilde**2 + lin * x_tilde + abs)
                                for (quad, lin, abs) in zip(quadratic_CO2, linear_CO2, abscissa_CO2)]
                    else:
                        intF = [i / (quadratic_CO2 * x_tilde**2 + lin * x_tilde + abs)
                                for (i, lin, abs) in zip(intF_max, linear_CO2, abscissa_CO2)]
    df_iF = pd.DataFrame(intF, index=['min', 'mean', 'max'], columns=x_tilde).T

    # -----------------------------------------------------------------------------
    # combining all (fit) parameter for two-site-model
    para_empiric = pd.Series({'quadratic': quadratic_CO2, 'linear': linear_CO2, 'abscissa': abscissa_CO2,
                              'intF': df_iF, 'pCO2_range': x_tilde, 'pCO2_calib': pCO2_calib})

    return para_empiric


def pCO2_eval_TSM(intF_max, intF_calib1, m, f, ksv_CO2):
    # preparation
    if isinstance(intF_max, np.float):
        int_quot = [intF_max/x2 for x2 in intF_calib1]
    else:
        int_quot = [x1/x2 for (x1, x2) in zip(intF_calib1, intF_max)]

    c = [1 - 1/q for q in int_quot]
    b = [k*(m*(1-f/t) + 1/t*(f-1)+1) for (k, t) in zip(ksv_CO2, int_quot)]
    a = [m*(p**2) for p in ksv_CO2]

    sqrt = [np.sqrt(b_**2 - 4*a_*c_) for (b_, a_, c_) in zip(b, a,c)]

    z1 = [-1*b_ + s for (b_, s) in zip(b, sqrt)]
    z2 = [-1*b_ - s for (b_, s) in zip(b, sqrt)]
    n = [2*a_ for a_ in a]

    pCO2_1 = sorted([z / n for (z, n) in zip(z1, n)])
    pCO2_2 = sorted([z / n for (z, n) in zip(z2, n)])

    # select reasonable pO2 value
    if np.abs(pCO2_1[1]) > 1E3:
        if np.abs(pCO2_2[1]) < 1E3:
            pCO2_calc = pCO2_2
        else:
            raise ValueError('Revise pCO2 evaluation! pCO2 was calculated to {:.2e}hPa or {:.2e}hPa'.format(pCO2_1[1],
                                                                                                            pCO2_2[1]))
    else:
        if all(np.array(pCO2_1) < 0) is True:
            print('All solutions are below zero! --> Due to physical reasonableness set the solution to zero.')
            pCO2_calc = [0., 0., 0.]
        else:
            pCO2_calc = pCO2_1

    pCO2_calc = np.array(pCO2_calc)

    return pCO2_calc


def pCO2_eval_empiric(intF_max, intF_meas, abscissa, linear, quadratic, pCO2_calib0, x_correction=True):
    # preparation
    if isinstance(intF_max, np.float):
        int_quot = [intF_max/x2 for x2 in intF_meas]
    else:
        int_quot = [x1/x2 for (x1, x2) in zip(intF_max, intF_meas)]

    if isinstance(quadratic, np.float) or isinstance(quadratic, int):
        if np.float(quadratic) == 0.:
            # reduced equation y0 / (b*x + c) = y
            pCO2_1 = reduced_euqation_pCO2(linear=linear, abscissa=abscissa, int_quot=int_quot)
            pCO2_2 = None
        else:
            # quadratic equation y = y0 / (a*x**2 + b*x + c)
            pCO2_1, pCO2_2 = pq_equation_pCO2(quadratic=quadratic, linear=linear, abscissa=abscissa, int_quot=int_quot)
    else:
        # quadratic is list
        if quadratic[1] == 0.:
            # reduced equation y0 / (b*x + c) = y
            pCO2_1 = reduced_euqation_pCO2(linear=linear, abscissa=abscissa, int_quot=int_quot)
            pCO2_2 = None
        else:
            # quadratic equation y = y0 / (a*x**2 + b*x + c)
            pCO2_1, pCO2_2 = pq_equation_pCO2(quadratic=quadratic, linear=linear, abscissa=abscissa, int_quot=int_quot)

    if x_correction is True:
        pCO2_1 = [pCO21 + pCO2_calib0 for pCO21 in pCO2_1]
        if pCO2_2 is None:
            pass
        else:
            pCO2_2 = [pCO22 + pCO2_calib0 for pCO22 in pCO2_2]
    else:
        pass

    # select reasonable pO2 value
    if np.abs(pCO2_1[1]) > 1E3:
        if np.abs(pCO2_2[1]) < 1E3:
            pCO2_calc = pCO2_2
        else:
            raise ValueError('Revise pCO2 evaluation! pCO2 was calculated to {:.2e}hPa or {:.2e}hPa'.format(pCO2_1[1],
                                                                                                            pCO2_2[1]))
    else:
        if all(np.array(pCO2_1) < 0) is True:
            print('All solutions are below zero! --> Due to physical reasonableness set the solution to zero.')
            pCO2_calc = [0., 0., 0.]
        else:
            pCO2_calc = pCO2_1
    pCO2_calc = np.array(pCO2_calc)

    return pCO2_calc


def pCO2_intensity_TSM(curv_CO2, pCO2_range, prop_ksv_CO2, intF_max, Ksv_fit1):
    if isinstance(Ksv_fit1, np.float):
        intF = (curv_CO2 / (1 + Ksv_fit1 * pCO2_range) +
                (1 - curv_CO2) / (1 + Ksv_fit1 * prop_ksv_CO2 * pCO2_range)) * intF_max
        df_intF = pd.DataFrame(intF, index=pCO2_range)
    else:
        if isinstance(intF_max, np.float):
            intF = [(curv_CO2 / (1 + p * pCO2_range) +
                     (1 - curv_CO2) / (1 + p * prop_ksv_CO2 * pCO2_range)) * intF_max for p in Ksv_fit1]
        else:
            intF = [(curv_CO2 / (1 + p * pCO2_range) + (1 - curv_CO2) / (1 + p * prop_ksv_CO2 * pCO2_range)) * Int
                    for (p, Int) in zip(Ksv_fit1, intF_max)]

        df_intF = pd.DataFrame(intF, columns=pCO2_range, index=['min', 'mean', 'max']).T

    return df_intF


def pCO2_intensity_empiric(pCO2_range, intF_max, quadratic, linear, abscissa):
    if isinstance(intF_max, np.float):
        # fluorescence float
        if isinstance(quadratic, np.float):
            # quadratic term float
            if isinstance(linear, np.float):
                # linear term float
                if isinstance(abscissa, np.float):
                    # all floats
                    intF = intF_max / (abscissa + pCO2_range * linear + (pCO2_range**2) * quadratic)
                else:
                    # abscissa is list
                    intF = [intF_max / (abs + pCO2_range * linear + (pCO2_range**2) * quadratic) for abs in abscissa]
            else:
                if isinstance(abscissa, np.float):
                    # all floats except linear
                    intF = [intF_max / (abscissa + pCO2_range * lin + (pCO2_range**2) * quadratic) for lin in linear]
                else:
                    # all floats except linear and abscissa
                    intF = [intF_max / (abs + pCO2_range * lin + (pCO2_range**2) * quadratic)
                            for (abs, lin) in zip(abscissa, linear)]
        else:
            # quadratic term list
            if isinstance(linear, np.float):
                # linear term float
                if isinstance(abscissa, np.float):
                    # all floats except quadratic
                    intF = [intF_max / (abscissa + pCO2_range * linear + (pCO2_range**2) * quad) for quad in quadratic]
                else:
                    # all float except quadratic and abscissa
                    intF = [intF_max / (abs + pCO2_range*linear + (pCO2_range**2) * quad)
                            for (abs, quad) in zip(abscissa, quadratic)]
            else:
                if isinstance(abscissa, np.float):
                    # all floats except quadratic and abscissa
                    intF = [intF_max / (abs + pCO2_range * linear + (pCO2_range**2) * quad)
                            for (abs, quad) in zip(abscissa, quadratic)]
                else:
                    # quadratic, linear and abscissa lists
                    intF = [intF_max / (abs + pCO2_range * lin + (pCO2_range**2) * quad)
                            for (abs, lin, quad) in (abscissa, linear, quadratic)]
        #df_intF = pd.DataFrame(intF, index=pCO2_range)
    else:
        # fluorescence list
        if isinstance(quadratic, np.float):
            # quadratic term float
            if isinstance(linear, np.float):
                # linear term float
                if isinstance(abscissa, np.float):
                    # all floats excpet intF
                    intF = [intF / (abscissa + pCO2_range * linear + (pCO2_range**2) * quadratic) for intF in intF_max]
                else:
                    # all floats excpet intF and abscissa
                    intF = [intF / (abs + pCO2_range * linear + (pCO2_range**2) * quadratic)
                            for (intF, abs) in zip(intF_max, abscissa)]
            else:
                # intF and linear term list
                if isinstance(abscissa, np.float):
                    # all floats except intF and linear term
                    intF = [intF / (abscissa + pCO2_range * lin + (pCO2_range**2) * quadratic)
                            for (intF, lin) in zip(intF_max, linear)]
                else:
                    # all floats except intF, linear and abscissa
                    intF = [intF / (abs + pCO2_range * lin + (pCO2_range**2) * quadratic)
                            for (intF, abs, lin) in zip(intF_max, abscissa, linear)]
        else:
            # quadratic term list
            if isinstance(linear, np.float):
                # linear term float
                if isinstance(abscissa, np.float):
                    # all floats except intF and quadratic term
                    intF = [intF / (abscissa + pCO2_range * linear + (pCO2_range**2) * quad)
                            for (intF, quad) in zip(intF_max, quadratic)]
                else:
                    # all floats except intF, quadratic term and abscissa
                    intF = [intF / (abs + pCO2_range * linear + (pCO2_range**2) * quad)
                            for (intF, abs, quad) in zip(intF_max, abscissa, quadratic)]
            else:
                # linear term list
                if isinstance(abscissa, np.float):
                    # all list except abscissa
                    intF = [intF / (abscissa + pCO2_range * lin + (pCO2_range**2) * quad)
                            for (intF, lin, quad) in zip(intF_max, linear, quadratic)]
                else:
                    # all list
                    intF = [intF / (abs + pCO2_range * lin + (pCO2_range**2) * quad)
                            for (intF, abs, lin, quad) in zip(intF_max, abscissa, linear, quadratic)]

        # convert intF into DataFrame
        df_intF = pd.DataFrame(intF, columns=pCO2_range, index=['min', 'mean', 'max']).T

    return df_intF


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


def linreg_intensity_to_intFmax_empiric(df_iF_p0, value):
    a0 = af.find_closest_value_(index=df_iF_p0.index, value=value, data=df_iF_p0['min'])
    a1 = af.find_closest_value_(index=df_iF_p0.index, value=value, data=df_iF_p0['mean'])
    a2 = af.find_closest_value_(index=df_iF_p0.index, value=value, data=df_iF_p0['max'])

    if a0[0] == a0[1]:
        iF_min = a0[2]
    else:
        arg_calib0 = stats.linregress(x=a0[:2], y=a0[2:])
        iF_min = arg_calib0[0] * value + arg_calib0[1]
    if a1[0] == a1[1]:
        iF_mean = a1[2]
    else:
        arg_calib0 = stats.linregress(x=a1[:2], y=a1[2:])
        iF_mean = arg_calib0[0] * value + arg_calib0[1]
    if a2[0] == a2[1]:
        iF_max = a2[2]
    else:
        arg_calib0 = stats.linregress(x=a2[:2], y=a2[2:])
        iF_max = arg_calib0[0] * value + arg_calib0[1]

    intF_max = [iF_min, iF_mean, iF_max]

    return intF_max


def linreg_parameter_int_to_pO2calc_TSM(pO2_calib, pO2_calc, para_TSM_p0, para_TSM_p1, intF):
    # linear regression to pO2 measured
    fit1_min = stats.linregress(x=pO2_calib, y=[para_TSM_p0['Ksv_fit1'][0], para_TSM_p1['Ksv_fit1'][0]])
    fit1_mean = stats.linregress(x=pO2_calib, y=[para_TSM_p0['Ksv_fit1'][1], para_TSM_p1['Ksv_fit1'][1]])
    fit1_max = stats.linregress(x=pO2_calib, y=[para_TSM_p0['Ksv_fit1'][2], para_TSM_p1['Ksv_fit1'][2]])

    Ksv_fit1_meas = [fit1_min[0] * pO2_calc[1] + fit1_min[1], fit1_mean[0] * pO2_calc[1] + fit1_mean[1],
                     fit1_max[0] * pO2_calc[1] + fit1_max[1]]
    para_TSM_meas = pd.Series({'prop Ksv': para_TSM_p0['prop Ksv'], 'slope': para_TSM_p0['slope'],
                               'Ksv_fit1': Ksv_fit1_meas,
                               'Ksv_fit2': [i*para_TSM_p0['prop Ksv'] for i in Ksv_fit1_meas]})

    return para_TSM_meas


def fit_parameter_int_to_pO2calc_empiric(pO2_calib, pO2_calc, para_lin_p0, para_lin_p1):

    # linear regression to pO2 measured
    if isinstance(para_lin_p0, np.float):
        q_mean = stats.linregress(x=pO2_calib, y=[para_lin_p0, para_lin_p1])
        if q_mean[0] == q_mean[1]:
            fit1_meas = q_mean[2]
        else:
            fit1_meas = q_mean[0] * pO2_calc[1] + q_mean[1]
    else:
        q_min = stats.linregress(x=pO2_calib, y=[para_lin_p0[0], para_lin_p1[0]])
        q_mean = stats.linregress(x=pO2_calib, y=[para_lin_p0[1], para_lin_p1[1]])
        q_max = stats.linregress(x=pO2_calib, y=[para_lin_p0[2], para_lin_p1[2]])
        if q_min[0] == q_min[1]:
            q_min_ = q_min[2]
        else:
            q_min_ = q_min[0] * pO2_calc[1] + q_min[1]
        if q_mean[0] == q_mean[1]:
            q_mean_ = q_mean[2]
        else:
            q_mean_ = q_mean[0] * pO2_calc[1] + q_mean[1]
        if q_max[0] == q_max[1]:
            q_max_ = q_max[2]
        else:
            q_max_ = q_max[0] * pO2_calc[1] + q_max[1]
        fit1_meas = [q_min_, q_mean_, q_max_]

    return fit1_meas


def linreg_parameter_int_to_pO2calc_empiric(pO2_calib, pO2_calc, para_lin_p0, para_lin_p1, intF):

    # linear regression to pO2 measured
    quad_fit1_meas = fit_parameter_int_to_pO2calc_empiric(pO2_calib=pO2_calib, pO2_calc=pO2_calc,
                                                          para_lin_p0=para_lin_p0['quadratic'],
                                                          para_lin_p1=para_lin_p1['quadratic'])

    linear_fit1_meas = fit_parameter_int_to_pO2calc_empiric(pO2_calib=pO2_calib, pO2_calc=pO2_calc,
                                                            para_lin_p0=para_lin_p0['linear'],
                                                            para_lin_p1=para_lin_p1['linear'])

    abs_fit1_meas = fit_parameter_int_to_pO2calc_empiric(pO2_calib=pO2_calib, pO2_calc=pO2_calc,
                                                         para_lin_p0=para_lin_p0['abscissa'],
                                                         para_lin_p1=para_lin_p1['abscissa'])

    para_lin_meas = pd.Series({'quadratic': quad_fit1_meas, 'linear': linear_fit1_meas, 'abscissa': abs_fit1_meas})


    return para_lin_meas


# --------------------------------------------------------------------------------------------------------------------
# INPUT parameter / Simulation
# --------------------------------------------------------------------------------------------------------------------
def simulate_phaseangle_pO2_pCO2(pO2_range, pO2_calib, pO2_meas, curv_O2, prop_ksv_O2, K_sv1_O2, tauP_c0,
                                 df_conv_tau_int_O2, intP_max, pCO2_range, pCO2_calib, pCO2_meas, intF_max, f1, f2,
                                 er_phase, curv_CO2=None, prop_ksv_CO2=None, K_sv1_CO2=None, plotting=True, decimal=4,
                                 normalize=False, fontsize_=13):

    # calibration
    # determine lifetime of phosphorescent according to given pO2-level
    tau_quotP, tauP, intP, intP_norm = twoSiteModel_fit(pO2_range=pO2_range, f=curv_O2, m=prop_ksv_O2, Ksv=K_sv1_O2,
                                                        tau_phos0=tauP_c0, conv_tau_int=df_conv_tau_int_O2,
                                                        calibration='2point', int_phos0=intP_max, plotting=plotting,
                                                        pO2_calib=pO2_calib, normalized_int=normalize)

    # determine i_f according to given pCO2 value
    if curv_CO2 is None or prop_ksv_CO2 is None or K_sv1_CO2 is None:
        raise ValueError('curv_CO2, prop_ksv_CO2 and K_sv1_CO2 are required for two site model fit!')
    intF, intF_norm = pCO2_fit_tsm(x=pCO2_range, pCO2_calib=pCO2_calib, f=curv_CO2, m=prop_ksv_CO2, Ksv=K_sv1_CO2,
                                   intF_max=intF_max, plotting=plotting, normalize=normalize,
                                   fontsize_=fontsize_)

    para_simulation = pd.Series({'tau_quot': tau_quotP, 'tauP': tauP, 'intP': intP, 'intF': intF})

    # -------------------------------------------------------------------------------------------
    # discrete values for calibration and simulation point - find closest values for simulation points
    # -------------------------------------
    # fitting intensity of the fluorophor
    iF_0, iF_1, iF_meas = af.fitting_to_measpoint(index_=intF.index, data_df=intF[intF.columns[0]],
                                                  value_meas=pCO2_meas, value_cal0=pCO2_calib[0],
                                                  value_cal1=pCO2_calib[1])

    # -------------------------------------
    # fitting intensity of the phosphor
    iP_0, iP_1, iP_meas = af.fitting_to_measpoint(index_=intP.index, data_df=intP[intP.columns[0]], value_meas=pO2_meas,
                                                  value_cal0=pO2_calib[0], value_cal1=pO2_calib[1])

    # -------------------------------------
    # fitting lifetime of the phosphor
    tauP_0, tauP_1, tauP_meas = af.fitting_to_measpoint(index_=tauP.index, data_df=tauP[0], value_meas=pO2_meas,
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
    print('pO2 for simulation: {:.2f} hPa'.format(pO2_meas))
    print('pCO2 for simulation: {:.2f} hPa'.format(pCO2_meas))
    # -------------------------------------------------------------------
    # amplitude ratio at 2 different modulation frequencies
    ampl_ratio_f1, ampl_ratio_f2 = af.amplitude_ratio(intP_discret=intP_discret, tauP_discret=tauP_discret,
                                                      i_f_discret=i_f_discret, f1=f1, f2=f2)

    Phi_f1_deg, Phi_f1_deg_er = af.superimposed_phaseangle_er(tauP_discret=tauP_discret, ampl_ratio=ampl_ratio_f1, f=f1,
                                                              er_phase=er_phase, decimal=decimal)
    Phi_f2_deg, Phi_f2_deg_er = af.superimposed_phaseangle_er(tauP_discret=tauP_discret, ampl_ratio=ampl_ratio_f2, f=f2,
                                                              er_phase=er_phase, decimal=decimal)

    # preparation of output (decimal digits)
    Phi_f1_deg = Phi_f1_deg.round(decimal)
    Phi_f2_deg = Phi_f2_deg.round(decimal)
    int_ratio = int_ratio.round(decimal)

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
    print('Deviation of lifetimes calculated: pO2(0) ~ {:.2e}µs, pO2(1) ~ {:.2e}µs'.format(dev_tau['tau_c0']*1E6,
                                                                                           dev_tau['tau_c1']*1E6))

    # ---------------------------------------------------------------------------------
    para_TSM = twoSiteModel_calib_ksv(tau0=tau_c0, tau1=tau_c1, m=prop_ksv, f=curv_O2, pO2_calib1=pO2_calib[1])

    # Calculation of pO2 at measurement point
    pO2_calc = twoSiteModel_evaluation(tau0=tau_c0, tau=tau_meas, m=prop_ksv, f=curv_O2, ksv=para_TSM['Ksv_fit1'],
                                       pO2_range=pO2_range)

    # ---------------------------------------------------------------------------------
    print('Calculated pO2: {:.2f} ± {:.2e} hPa'.format(pO2_calc.mean(), pO2_calc.std()))

    return pO2_calc, para_TSM, tau, Phi_f1_deg_er, Phi_f2_deg_er


def oxygen_sensing_1cp(pO2_range, pO2_calib, curv_O2, prop_ksv, Phi_c1_f1_deg, Phi_c1_f2_deg, phi_f1_meas,
                       phi_f2_meas, error_phaseangle, f1, f2, tau_c0=None, Ksv=None):
    if isinstance(pO2_calib, np.float):
        pass
    else:
        pO2_calib = pO2_calib[1]

    # phosphorescence lifetime from 2-Frequency measurement
    [Phi_c1_f1_deg_er, Phi_c1_f2_deg_er,
     para_tau] = af.preparation_lifetime_2cp(phi_f1_deg=Phi_c1_f1_deg, phi_f2_deg=Phi_c1_f2_deg, f1=f1, f2=f2, er=True,
                                             err_phase=error_phaseangle)

    [tau_meas, Phi_f1_meas_er,
     Phi_f2_meas_er] = af.phi_to_lifetime_including_error(phi_f1=phi_f1_meas, f1=f1, f2=f2, er=True, phi_f2=phi_f2_meas,
                                                          err_phaseangle=error_phaseangle)

    Phi_c1_f1_deg_er = Phi_c1_f1_deg_er.append(pd.Series({'meas': np.rad2deg(Phi_f1_meas_er)}))
    Phi_c1_f2_deg_er = Phi_c1_f2_deg_er.append(pd.Series({'meas': np.rad2deg(Phi_f2_meas_er)}))

    # ---------------------------------------------------------------------------------
    # 1-Point-Calibration
    if tau_c0 is None:
        if Ksv is None:
            raise ValueError('Error! either Ksv or tauP_c0 is required for 1-Point-Calibration!')
        else:
            para_TSM = twoSiteModel_calib_tauP0(ksv=Ksv, tau1=para_tau['fluoro0, phosphor1'], m=prop_ksv, f=curv_O2,
                                                pO2_calib1=pO2_calib)
    else:
        if Ksv is None:
            para_TSM = twoSiteModel_calib_ksv(tau0=tau_c0, tau1=para_tau['fluoro0, phosphor1'], m=prop_ksv, f=curv_O2,
                                              pO2_calib1=pO2_calib)
        else:
            calib = input('For 1-Point-Calibration decide wether Ksv or tauP ought to be calibrated.')
            if calib == 'tauP' or calib == 'tau' or calib == 'tauP_c0' or calib == 'Lifetime' or calib == 'lifetime':
                para_TSM = twoSiteModel_calib_tauP0(ksv=Ksv, tau1=para_tau['fluoro0, phosphor1'], m=prop_ksv,
                                                    f=curv_O2, pO2_calib1=pO2_calib)
            else:
                para_TSM = twoSiteModel_calib_ksv(tau0=tau_c0, tau1=para_tau['fluoro0, phosphor1'], m=prop_ksv,
                                                  f=curv_O2, pO2_calib1=pO2_calib)

    # Calculation of pO2 at measurement point
    pO2_calc = twoSiteModel_evaluation(tau0=para_TSM['tauP0'], tau=tau_meas, m=para_TSM['prop Ksv'],
                                       f=para_TSM['slope'], ksv=para_TSM['Ksv_fit1'], pO2_range=pO2_range)

    # ---------------------------------------------------------------------------------
    print('Calculated pO2: {:.2f} ± {:.2e} hPa'.format(pO2_calc.mean(), pO2_calc.std()))

    # ---------------------------------------------------------------------------------
    tau = pd.Series({'phosphor0': para_TSM['tauP0'], 'phosphor1': para_TSM['tauP1'], 'meas': tau_meas})

    return pO2_calc, para_TSM, tau, Phi_c1_f1_deg_er, Phi_c1_f2_deg_er


# Individual pCO2 sensing
def pCO2_sensing(pCO2_calib, pCO2_range, intF, intF_max, pO2_calc, pO2_calib, curv_CO2=None, prop_ksv_CO2=None,
                 plotting=True, fontsize_=13):

    # fit_parameter at both calibration points for the phosphor
    if curv_CO2 is None or prop_ksv_CO2 is None:
        raise ValueError('curve_CO2 and prop_ksv_CO2 are required for two site model fit!')

    # re-calibration of fit-parameters at pO2_calib1 (as intP is minimal in this case)
    para_p0 = pCO2_calib_TSM(intF_max=intF_max, intF_calib1=intF['fluoro1, phosphor0'], m=prop_ksv_CO2, f=curv_CO2,
                             pCO2_calib1=pCO2_calib[1], pCO2_range=pCO2_range)
    para_p1 = pCO2_calib_TSM(intF_max=intF_max, intF_calib1=intF['fluoro1, phosphor1'], m=prop_ksv_CO2, f=curv_CO2,
                             pCO2_calib1=pCO2_calib[1], pCO2_range=pCO2_range)

    # linear fit to calculated pO2 value
    para_meas = linreg_parameter_int_to_pO2calc_TSM(pO2_calib=pO2_calib, intF=intF, pO2_calc=pO2_calc,
                                                    para_TSM_p0=para_p0, para_TSM_p1=para_p1)

    # ----------------------------------------------------------------------
    # linear regression of fluorescence intensity to measurement point
    intF_max_meas = linreg_intensity_to_pO2(pO2_calc=pO2_calc, x=pO2_calib, i_f_c0=para_p0['intF'].loc[0],
                                            i_f_c1=para_p1['intF'].loc[0])

    intF_fit_meas = pCO2_intensity_TSM(pCO2_range=pCO2_range, curv_CO2=para_meas['slope'],
                                       Ksv_fit1=para_meas['Ksv_fit1'], prop_ksv_CO2=para_meas['prop Ksv'],
                                       intF_max=intF_max_meas.loc[0])

    # ----------------------------------------------------------------------
    # pCO2_calculation
    pCO2_calc = pCO2_eval_TSM(intF_max=intF_fit_meas.loc[0], intF_calib1=intF['meas'], m=para_meas['prop Ksv'],
                              f=para_meas['slope'], ksv_CO2=para_meas['Ksv_fit1'])

    print('Calculated pCO2: {:.2f} ± {:.2e} hPa'.format(pCO2_calc.mean(), pCO2_calc.std()))

    # ----------------------------------------------------------------------
    # plotting
    if plotting is True:
        f, ax = plt.subplots()
        ax.plot(intF_fit_meas.index, intF_fit_meas['min'], color='navy', lw=0.75)
        ax.plot(intF_fit_meas.index, intF_fit_meas['mean'], color='navy', lw=1.)
        ax.plot(intF_fit_meas.index, intF_fit_meas['max'], color='navy', lw=0.75)

        ax.axvline(pCO2_calib[0], lw=0.75, ls='--', color='k')
        ax.axvline(pCO2_calib[1], lw=0.75, ls='--', color='k')

        # find closest value to calculated measurement point
        a = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['min'].values, value=pCO2_calc[0])
        b = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['mean'].values, value=pCO2_calc[1])
        c = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['max'].values, value=pCO2_calc[2])

        if a[0] == a[1]:
            intF_min = a[2]
        else:
            arg_calc_min = stats.linregress(x=a[:2], y=a[2:])
            intF_min = arg_calc_min[0] * pCO2_calc[0] + arg_calc_min[1]
        if b[0] == b[1]:
            intF_mean = b[2]
        else:
            arg_calc_mean = stats.linregress(x=b[:2], y=b[2:])
            intF_mean = arg_calc_mean[0] * pCO2_calc[1] + arg_calc_mean[1]
        if c[0] == c[1]:
            intF_max = c[2]
        else:
            arg_calc_max = stats.linregress(x=c[:2], y=c[2:])
            intF_max = arg_calc_max[0] * pCO2_calc[2] + arg_calc_max[1]

        if np.isnan(intF_min):
            y_c0 = intF.loc[pCO2_calc[0]].values[0]
        else:
            y_c0 = intF_min
        if np.isnan(intF_max):
            y_c1 = intF.loc[pCO2_calc[2]].values[0]
        else:
            y_c1 = intF_max

        ax.set_ylim(0, 130)
        ax.set_xlim(pCO2_range[0], pCO2_range[-1])
        ax.axhline(y_c0, color='k', lw=1., ls='--')
        ax.axhline(y_c1, color='k', lw=1., ls='--')

        ax.set_xlabel('p$CO_2$ / hPa', fontsize=fontsize_)
        ax.set_ylabel('Rel. Intensity I$_F$', fontsize=fontsize_)

    return pCO2_calc, intF_fit_meas, para_meas


def pCO2_sensing_1cp(pCO2_calib, pCO2_range, intF, intF_max, curv_CO2=None, prop_ksv_CO2=None, plotting=True,
                     fontsize_=13):
    # fit_parameter at both calibration points for the phosphor
    if curv_CO2 is None or prop_ksv_CO2 is None:
        raise ValueError('curve_CO2 and prop_ksv_CO2 are required for two site model fit!')

    # find correct key of dict
    s = 'fluoro1'
    key_extracted = []
    for k in intF.keys().tolist():
        if s in k:
            key_extracted.append(k)

    # average intF between phosphor0 and phosphor1 at fluoro1
    if len(intF[key_extracted].keys()) > 1:
        intF_ = [(a1 + a2)/2 for (a1, a2) in zip(intF[key_extracted[0]], intF[key_extracted[1]])]
    else:
        intF_ = intF[key_extracted[0]]

    # re-calibration of fit-parameters at pO2_calib1 (as intP is minimal in this case)
    para_meas = pCO2_calib_TSM(intF_max=intF_max, intF_calib1=intF_, m=prop_ksv_CO2, f=curv_CO2,
                               pCO2_calib1=pCO2_calib[1], pCO2_range=pCO2_range)

    # ----------------------------------------------------------------------
    # linear regression of fluorescence intensity to measurement point
    intF_fit_meas = pCO2_intensity_TSM(pCO2_range=pCO2_range, curv_CO2=para_meas['slope'],
                                       Ksv_fit1=para_meas['Ksv_fit1'], prop_ksv_CO2=para_meas['prop Ksv'],
                                       intF_max=intF_max)

    # ----------------------------------------------------------------------
    # pCO2_calculation
    ls_meas_ind = []
    for i in intF.index:
        if 'meas' in i:
            ls_meas_ind.append(i)
    pCO2_calc = pCO2_eval_TSM(intF_max=intF_fit_meas.loc[0], intF_calib1=intF[ls_meas_ind[0]], m=para_meas['prop Ksv'],
                              f=para_meas['slope'], ksv_CO2=para_meas['Ksv_fit1'])
    print('Calculated pCO2: {:.2f} ± {:.2e} hPa'.format(pCO2_calc.mean(), pCO2_calc.std()))

    # ----------------------------------------------------------------------
    # plotting
    if plotting is True:
        f, ax = plt.subplots()
        ax.plot(intF_fit_meas.index, intF_fit_meas['min'], color='navy', lw=0.75)
        ax.plot(intF_fit_meas.index, intF_fit_meas['mean'], color='navy', lw=1.)
        ax.plot(intF_fit_meas.index, intF_fit_meas['max'], color='navy', lw=0.75)

        ax.axvline(pCO2_calib[0], lw=0.75, ls='--', color='k')
        ax.axvline(pCO2_calib[1], lw=0.75, ls='--', color='k')

        # find closest value to calculated measurement point
        a = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['min'].values, value=pCO2_calc[0])
        b = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['mean'].values, value=pCO2_calc[1])
        c = af.find_closest_value_(index=intF_fit_meas.index, data=intF_fit_meas['max'].values, value=pCO2_calc[2])

        if a[0] == a[1]:
            intF_min = a[2]
        else:
            arg_calc_min = stats.linregress(x=a[:2], y=a[2:])
            intF_min = arg_calc_min[0] * pCO2_calc[0] + arg_calc_min[1]
        if b[0] == b[1]:
            intF_mean = b[2]
        else:
            arg_calc_mean = stats.linregress(x=b[:2], y=b[2:])
            intF_mean = arg_calc_mean[0] * pCO2_calc[1] + arg_calc_mean[1]
        if c[0] == c[1]:
            intF_max = c[2]
        else:
            arg_calc_max = stats.linregress(x=c[:2], y=c[2:])
            intF_max = arg_calc_max[0] * pCO2_calc[2] + arg_calc_max[1]

        if np.isnan(intF_min):
            y_c0 = intF.loc[pCO2_calc[0]].values[0]
        else:
            y_c0 = intF_min
        if np.isnan(intF_max):
            y_c1 = intF.loc[pCO2_calc[2]].values[0]
        else:
            y_c1 = intF_max

        ax.set_ylim(0, 130)
        ax.set_xlim(pCO2_range[0], pCO2_range[-1])
        ax.axhline(y_c0, color='k', lw=1., ls='--')
        ax.axhline(y_c1, color='k', lw=1., ls='--')

        ax.set_xlabel('p$CO_2$ / hPa', fontsize=fontsize_)
        ax.set_ylabel('Rel. Intensity I$_F$', fontsize=fontsize_)

    return pCO2_calc, intF_fit_meas, para_meas


# =====================================================================================================================
# Dualsensing
# =====================================================================================================================
# simulation with 4 (CO2_oxygen_dualsensor) or 2 (CO2_O2_hybrid_2cp) calibration points
def CO2_oxygen_dualsensor(pO2_range, pO2_calib, curv_O2, prop_ksv_O2, phi_f1_deg, phi_f1_meas, phi_f2_deg, phi_f2_meas,
                          er_phase, df_conv_tau_intP, intP_max, intF_max, pCO2_range, pCO2_calib, f1, f2, method_,
                          curv_CO2=None, prop_ksv_CO2=None, plotting=True, fontsize_=13):

    # ----------------------------------------------------------------------------------------------------------------
    # Oxygen sensing - phosphorescence sensor
    [pO2_calc, para_TSM_O2, tau, Phi_f1_deg_er,
     Phi_f2_deg_er] = oxygen_sensing(pO2_range=pO2_range, pO2_calib=pO2_calib, curv_O2=curv_O2, prop_ksv=prop_ksv_O2,
                                     Phi_f1_deg=phi_f1_deg, phi_f1_meas=phi_f1_meas, Phi_f2_deg=phi_f2_deg,
                                     phi_f2_meas=phi_f2_meas, error_phaseangle=er_phase, f1=f1, f2=f2, method_=method_)

    # From lifetime to intensity of the phosphorescent
    [tau_quot, tauP, intP, intP_norm] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM_O2['slope'],
                                                         m=para_TSM_O2['prop Ksv'], Ksv=para_TSM_O2['Ksv_fit1'],
                                                         tau_phos0=tau['phosphor0'], conv_tau_int=df_conv_tau_intP,
                                                         int_phos0=intP_max, plotting=False, pO2_calib=pO2_calib,
                                                         normalized_int=False, fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # intensity ratio via equation
    int_ratio = af.intensity_ratio_calculation(f1=f1, f2=f2, tau=tau, phi_f1_deg=Phi_f1_deg_er,
                                               phi_f2_deg=Phi_f2_deg_er)

    # conversion intensity ratio into total amplitude based on fitted intP
    ampl_total_f1, ampl_total_f2, intP_max_calc = af.int_ratio_to_ampl(f1=f1, f2=f2, pO2_calib=pO2_calib,
                                                                       pO2_calc=pO2_calc, int_ratio=int_ratio,
                                                                       intP=intP, tauP=tauP)

    # conversion intP to intF
    intF = af.intP_to_intF(intP=intP, calibP=pO2_calib, calcP=pO2_calc, int_ratio=int_ratio)

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    pCO2_calc, intF_meas, para_TSM_CO2 = pCO2_sensing(pCO2_calib=pCO2_calib, pCO2_range=pCO2_range, intF=intF,
                                                      intF_max=intF_max, prop_ksv_CO2=prop_ksv_CO2, curv_CO2=curv_CO2,
                                                      pO2_calc=pO2_calc, pO2_calib=pO2_calib, plotting=False,
                                                      fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pCO2 = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                              pCO2_range=pCO2_range, pCO2_calib=pCO2_calib, int_fluoro=intF_meas,
                                              tauP_c0=tau['phosphor0'][1], pCO2_calc=pCO2_calc, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pCO2 = None

    return pO2_calc, pCO2_calc, tau_quot, tauP, para_TSM_O2, intP, intF_meas, para_TSM_CO2, ax_pO2, ax_pCO2,\
           ampl_total_f1, ampl_total_f2


def CO2_O2_hybrid_2cp(pO2_range, pO2_calib, curv_O2, prop_ksv_O2, phi_f1_c1, phi_f1_meas, phi_f2_c1,
                      phi_f2_meas, er_phase, df_conv_tau_intP, intP_max, intF_max, pCO2_range, pCO2_calib, f1, f2,
                      curv_CO2, prop_ksv_CO2, ksv_O2=None, tau_c0=None, plotting=True, fontsize_=13,
                      calibration='1point'):
    # ----------------------------------------------------------------------------------------------------------------
    # Oxygen sensing - phosphorescence sensor
    [pO2_calc, para_TSM_O2, tau, Phi_f1_deg_er,
     Phi_f2_deg_er] = oxygen_sensing_1cp(pO2_range=pO2_range, pO2_calib=pO2_calib, tau_c0=tau_c0, curv_O2=curv_O2,
                                         prop_ksv=prop_ksv_O2, Phi_c1_f1_deg=phi_f1_c1, phi_f1_meas=phi_f1_meas,
                                         Phi_c1_f2_deg=phi_f2_c1, phi_f2_meas=phi_f2_meas, error_phaseangle=er_phase,
                                         f1=f1, f2=f2, Ksv=ksv_O2)

    # From lifetime to intensity of the phosphorescent
    [tau_quot, tauP, intP, intP_norm] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM_O2['slope'],
                                                         m=para_TSM_O2['prop Ksv'], Ksv=para_TSM_O2['Ksv_fit1'],
                                                         tau_phos0=tau['phosphor0'], conv_tau_int=df_conv_tau_intP,
                                                         int_phos0=intP_max, plotting=False, pO2_calib=pO2_calib,
                                                         normalized_int=False, fontsize_=fontsize_,
                                                         calibration=calibration)

    # ------------------------------------------------------------------------------------
    # intensity ratio via equation
    int_ratio = af.intensity_ratio_calculation(f1=f1, f2=f2, tau=tau, phi_f1_deg=Phi_f1_deg_er,
                                               phi_f2_deg=Phi_f2_deg_er)

    # conversion intensity ratio into total amplitude based on fitted intP
    ampl_total_f1, ampl_total_f2, intP_max_calc = af.int_ratio_to_ampl_1cp(f1=f1, f2=f2, pO2_calib=pO2_calib,
                                                                           pO2_calc=pO2_calc, int_ratio=int_ratio,
                                                                           intP=intP, tauP=tauP)

    # conversion intP to intF
    intF = af.intP_to_intF_1cp(intP=intP, calibP=pO2_calib, calcP=pO2_calc, int_ratio=int_ratio)

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    pCO2_calc, intF_meas, para_TSM_CO2 = pCO2_sensing_1cp(pCO2_calib=pCO2_calib, pCO2_range=pCO2_range, intF=intF,
                                                          intF_max=intF_max, prop_ksv_CO2=prop_ksv_CO2,
                                                          curv_CO2=curv_CO2, plotting=False, fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pCO2 = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                              pCO2_range=pCO2_range, pCO2_calib=pCO2_calib, int_fluoro=intF_meas,
                                              tauP_c0=tau['phosphor0'], pCO2_calc=pCO2_calc, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pCO2 = None

    return pO2_calc, pCO2_calc, tau_quot, tauP, para_TSM_O2, intP, intF_meas, para_TSM_CO2, ax_pO2, ax_pCO2,\
           ampl_total_f1, ampl_total_f2


# ---------------------------------------------------------------------------------
# measurement evaluation with 4 (CO2_oxygen_dualsensor_meas) or 2 (CO2_O2_hybrid_2cp_meas) calibration points

def CO2_oxygen_dualsensor_meas(pO2_range, pO2_calib, curv_O2, prop_ksv_O2, phi_f1_deg, phi_f1_meas, phi_f2_deg,
                               phi_f2_meas, er_phase, df_conv_tau_intP, intP_max, intF_max, pCO2_range, pCO2_calib,
                               f1, f2, method_, int_ratio=None, ampl_total_f2=None, ampl_total_f1=None, curv_CO2=None,
                               prop_ksv_CO2=None, plotting=True, fontsize_=13):
    # pre-check: either int_ratio or total amplitude at two modulation frequencies are required!
    if int_ratio is None and ampl_total_f1 is None or int_ratio is None and ampl_total_f2 is None:
        raise ValueError('Either int_ratio or the total amplitude are required for evaluation!')
    else:
        pass

    # ----------------------------------------------------------------------------------------------------------------
    # Oxygen sensing - phosphorescence sensor
    [pO2_calc, para_TSM_O2, tau, Phi_f1_deg_er,
     Phi_f2_deg_er] = oxygen_sensing(pO2_range=pO2_range, pO2_calib=pO2_calib, curv_O2=curv_O2, prop_ksv=prop_ksv_O2,
                                     Phi_f1_deg=phi_f1_deg, phi_f1_meas=phi_f1_meas, Phi_f2_deg=phi_f2_deg,
                                     phi_f2_meas=phi_f2_meas, error_phaseangle=er_phase, f1=f1, f2=f2, method_=method_)

    # From lifetime to intensity of the phosphorescent
    [tau_quot, tauP, intP, intP_norm] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM_O2['slope'],
                                                         m=para_TSM_O2['prop Ksv'], Ksv=para_TSM_O2['Ksv_fit1'],
                                                         tau_phos0=tau['phosphor0'], conv_tau_int=df_conv_tau_intP,
                                                         int_phos0=intP_max, plotting=False, pO2_calib=pO2_calib,
                                                         normalized_int=False, fontsize_=fontsize_)

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
        int_ratio_meas = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau['meas'], dphi_f1=phi_f1_meas,
                                              dphi_f2=phi_f2_meas)
        int_ratio = pd.Series({'fluoro0, phosphor0': int_ratio_F0P0, 'fluoro1, phosphor0': int_ratio_F1P0,
                               'fluoro0, phosphor1': int_ratio_F0P1, 'fluoro1, phosphor1': int_ratio_F1P1,
                               'meas': int_ratio_meas})

    # conversion intP to intF
    intF = af.intP_to_intF(intP=intP, calibP=pO2_calib, calcP=pO2_calc, int_ratio=int_ratio)

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    pCO2_calc, intF_meas, para_TSM_CO2 = pCO2_sensing(pCO2_calib=pCO2_calib, pCO2_range=pCO2_range, intF=intF,
                                                      intF_max=intF_max, pO2_calc=pO2_calc, pO2_calib=pO2_calib,
                                                      prop_ksv_CO2=prop_ksv_CO2, curv_CO2=curv_CO2, plotting=False,
                                                      fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pCO2 = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                              pCO2_range=pCO2_range, pCO2_calib=pCO2_calib, int_fluoro=intF_meas,
                                              tauP_c0=tau['phosphor0'][1], pCO2_calc=pCO2_calc, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pCO2 = None

    return pO2_calc, pCO2_calc, tau_quot, tauP, para_TSM_O2, intP, intF_meas, para_TSM_CO2, ax_pO2, ax_pCO2


def CO2_O2_hybrid_2cp_meas(pO2_range, pO2_calib, curv_O2, prop_ksv_O2, phi_f1_c1, phi_f1_meas, phi_f2_c1, phi_f2_meas,
                           er_phase, df_conv_tau_intP, intP_max, intF_max, pCO2_range, pCO2_calib, f1, f2,
                           int_ratio=None, ampl_total_f2=None, ampl_total_f1=None, curv_CO2=None, prop_ksv_CO2=None,
                           ksv_O2=None, tau_c0=None, plotting=True, fontsize_=13, calibration='1point'):
    # pre-check: either int_ratio or total amplitude at two modulation frequencies are required!
    if int_ratio is None and ampl_total_f1 is None or int_ratio is None and ampl_total_f2 is None:
        raise ValueError('Either int_ratio or the total amplitude are required for evaluation!')
    else:
        pass

    if isinstance(pO2_calib, np.float):
        pass
    else:
        pO2_calib = np.array(pO2_calib).max()

    # ----------------------------------------------------------------------------------------------------------------
    # Oxygen sensing - phosphorescence sensor
    [pO2_calc, para_TSM_O2, tau, Phi_f1_deg_er,
     Phi_f2_deg_er] = oxygen_sensing_1cp(pO2_range=pO2_range, pO2_calib=pO2_calib, tau_c0=tau_c0, curv_O2=curv_O2,
                                         prop_ksv=prop_ksv_O2, Phi_c1_f1_deg=phi_f1_c1, phi_f1_meas=phi_f1_meas,
                                         Phi_c1_f2_deg=phi_f2_c1, phi_f2_meas=phi_f2_meas, error_phaseangle=er_phase,
                                         f1=f1, f2=f2, Ksv=ksv_O2)

    # From lifetime to intensity of the phosphorescent
    [tau_quot, tauP, intP, intP_norm] = twoSiteModel_fit(pO2_range=pO2_range, f=para_TSM_O2['slope'],
                                                         m=para_TSM_O2['prop Ksv'], Ksv=para_TSM_O2['Ksv_fit1'],
                                                         tau_phos0=tau['phosphor0'], conv_tau_int=df_conv_tau_intP,
                                                         int_phos0=intP_max, plotting=False, pO2_calib=pO2_calib,
                                                         normalized_int=False, fontsize_=fontsize_,
                                                         calibration=calibration)

    # ------------------------------------------------------------------------------------
    # in case only the total amplitude is given -> conversion into intensity ratio
    if ampl_total_f1 is None:
        pass
    else:
        keys = phi_f1_c1.keys().tolist()
        keys.append('meas')
        int_ratio = pd.Series({key: None for key in keys})
        for i in keys:
            if len(i.split(' ')) > 1:
                phos = i.split(' ')[1]
                int_ratio[i] = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau[phos], dphi_f1=phi_f1_c1[i],
                                                    dphi_f2=phi_f2_c1[i])
            else:
                phos = i.split(' ')[0]
                int_ratio[i] = af.ampl_to_int_ratio(f1=f1, f2=f2, tauP=tau[phos], dphi_f1=phi_f1_meas,
                                                    dphi_f2=phi_f2_meas)

    # conversion intP to intF
    intF = af.intP_to_intF_1cp(intP=intP, calibP=pO2_calib, calcP=pO2_calc, int_ratio=int_ratio)

    # ------------------------------------------------------------------------------------
    # fluorescence sensor
    pCO2_calc, intF_meas, para_TSM_CO2 = pCO2_sensing_1cp(pCO2_calib=pCO2_calib, pCO2_range=pCO2_range, intF=intF,
                                                          intF_max=intF_max, prop_ksv_CO2=prop_ksv_CO2,
                                                          curv_CO2=curv_CO2, plotting=False, fontsize_=fontsize_)

    # ------------------------------------------------------------------------------------
    # Plotting
    if plotting is True:
        ax_pO2, ax_pCO2 = plotting_dualsensor(pO2_calib=pO2_calib, pO2_calc=pO2_calc, df_tau_quot=tau_quot,
                                              pCO2_range=pCO2_range, pCO2_calib=pCO2_calib, int_fluoro=intF_meas,
                                              tauP_c0=tau['phosphor0'], pCO2_calc=pCO2_calc, fontsize_=fontsize_)
    else:
        ax_pO2 = None
        ax_pCO2 = None

    return pO2_calc, pCO2_calc, tau_quot, tauP, para_TSM_O2, intP, intF_meas, para_TSM_CO2, ax_pO2, ax_pCO2

