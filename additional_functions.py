__author__ = 'szieger'
__project__ = 'dualsensor'

import matplotlib.pyplot as plt
import matplotlib.pylab
import matplotlib.gridspec as gspec
import numpy as np
import math
import pandas as pd
import cmath
from termcolor import colored
from tabulate import tabulate
from scipy import stats
import itertools
import datetime
from mpmath import *
import os
from glob import glob
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon, QDoubleValidator, QColor
from PyQt5.QtCore import Qt

today = datetime.date.today()

# ----------------------------------------------------------------------------------------------------------------------
# trigonometric functions
# ----------------------------------------------------------------------------------------------------------------------
def cot(phi):
    return 1/np.tan(phi)


def arccot(cot):
    return np.arctan(1/cot)


# ----------------------------------------------------------------------------------------------------------------------
# general fitting functions
# ----------------------------------------------------------------------------------------------------------------------
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


def function_empiric(x, abs_, lin, quad):
    return 1/(abs_ + lin*x + quad*x**2)


def func_exp(x, a, b, c):
    return np.array(a * np.exp(-b * x) + c, dtype=np.float64)


def func_exp_2(x, b, c):
    return np.array(np.exp(-b * x) + c, dtype=np.float64)


def fitting_intensity(p, xdata_reg, steps, method, m, k, f, abs_, lin, quad, tsm, empiric, type_, path=None,
                      plotting=True, fontsize_=13, saving=True):
    # load data
    df_ = pd.read_csv(p, sep='\t', decimal=',', dtype=np.float64, encoding='latin-1')

    # Preparing
    frequency = p.split('/')[-1].split('_')[-1].split('.')[0]
    print('Analyzing ', frequency)
    number_runs = int(len(df_.index)/int(steps))

    df = pd.DataFrame(np.zeros(shape=(int(steps), len(df_.columns)-2)))
    df_std = pd.DataFrame(np.zeros(shape=(int(steps), number_runs)))

    partial_pressure = df_['Partial pressure [hPa]'].loc[:steps-1].values
    print('[1] Pressure correction and normalization')

    x_tilde = partial_pressure - np.float(partial_pressure[0])
    df.index = x_tilde
    df_std.index = x_tilde

    for i in range(number_runs):
        df[i] = df_.loc[:steps-1, df_.columns[-2]].values
        df_std[i] = df_.loc[:steps-1, df_.columns[-1]].values

    if type_ == 'phaseangle' or type_ == 'phase angle' or type_ == 'lifetime':
        df = 1/df

    # **********************************************************************************
    # Saving
    if saving is True:
        if path is None:
            raise ValueError('Saving path is required!')
        else:
            pass
        # preparation
        directory0 = Path(path + '/FitReport/')
        directory1 = Path(path + '/Report/')
        directory2 = Path(path + '/Graph/')
        if not os.path.exists(directory0):
            os.makedirs(directory0)
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        if not os.path.exists(directory2):
            os.makedirs(directory2)

    # **********************************************************************************
    # Fitting
    # starting point
    if method == 'TSM' or method == 'Two-site model':
        try:
            tsm
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Chosen method requires an adequate model!")
            calibration_para_failed.setInformativeText("As you chose the two-site-model for fitting, provide the "
                                                       "corresponding model as tsm = Model(function_TSM).")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        method = 'TSM'
        if m is None or k is None or f is None:
            m = 0.1
            k = 0.3
            f = 1.
            print('#1: Warning! No starting point defined. Default values are: m=0.6, k=0.4, f=1.')
        else:
            pass
        params_tsm = tsm.make_params(m=m, k=k, f=f)
        params_tsm['f'].vary = False
        params_tsm['f'].value = 0.85
        params_tsm['m'].min = 0.09
        params_tsm['m'].max = 0.22
        statistics = pd.DataFrame(np.zeros(shape=(number_runs, len(tsm.param_names)*2 + 1)),
                                  columns=['m', 'm std', 'k', 'k std', 'f', 'f std', 'X2'])
        for i in range(number_runs):
            result_tsm = tsm.fit(df[i], x=x_tilde, params=params_tsm)
            if saving is True:
                name0 = path + '/FitReport/' + method + '_rel-Intensity_' + frequency + '.txt'
                fh = open(name0, 'w')
                fh.write(result_tsm.fit_report())
                fh.close()

            s0 = result_tsm.fit_report().split('\n')[12]
            s1 = result_tsm.fit_report().split('\n')[13]
            s2 = result_tsm.fit_report().split('\n')[14]

            if s0.find('+') == -1:
                if s0.find('.') == -1:
                    statistics.loc[i, 'm'] = np.float(s0[s0.find(':')+1 : s0.find('(')-1])
                    statistics.loc[i, 'm std'] = np.nan
                else:
                    statistics.loc[i, 'm'] = np.float(s0[s0.find('.')-1 : s0.find('(')-1])
                    statistics.loc[i, 'm std'] = np.nan
            else:
                statistics.loc[i, 'm'] = np.float(s0[s0.find('.')-1 : s0.find('+')-1])
                statistics.loc[i, 'm std'] = s0[s0.find('.', s0.find('.')+1)-1 : s0.find('(')]

            if s1.find('+') == -1:
                if s1.find('.') == -1:
                    statistics.loc[i, 'k'] = np.float(s1[s1.find(':')+1 : s1.find('(')-1])
                    statistics.loc[i, 'k std'] = np.nan
                else:
                    statistics.loc[i, 'k'] = np.float(s1[s1.find('.')-1 : s1.find('(')-1])
                    statistics.loc[i, 'k std'] = np.nan
            else:
                statistics.loc[i, 'k'] = np.float(s1[s1.find('.')-1 : s1.find('+')-1])
                statistics.loc[i, 'k std'] = s1[s1.find('.', s1.find('.')+1)-1 : s1.find('(')]

            if s2.find('+') == -1:
                if s2.find('.') == -1:
                    statistics.loc[i, 'f'] = np.float(s2[s2.find(':')+1 : s2.find('(')-1])
                    statistics.loc[i, 'f std'] = np.nan
                else:
                    statistics.loc[i, 'f'] = np.float(s2[s2.find('.')-1 : s2.find('(')-1] )
                    statistics.loc[i, 'f std'] = np.nan
            else:
                statistics.loc[i, 'f'] = np.float(s2[s2.find('.')-1 : s2.find('+')-1])
                statistics.loc[i, 'f std'] = s2[s2.find('.', s2.find('.')+1)-1 : s2.find('(')]

            # goodness of fit (chi square)
            statistics.loc[i, 'X2'] = np.float(result_tsm.fit_report().split('\n')[7].split('=')[1])

            # ----------------------------------------------------------------
            # regression
            y_regression = pd.DataFrame(np.zeros(shape=(len(xdata_reg), number_runs)),
                                        index=xdata_reg+partial_pressure[0])
            for i in range(number_runs):
                y_regression[i] = function_TSM(x=xdata_reg, f=statistics.loc[i, 'f'], m=statistics.loc[i, 'm'],
                                               k=statistics.loc[i, 'k'])

    # ------------------------------------------------------------------------------
    elif method == 'empiric':
        try:
            empiric
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Chosen method requires an adequate model!")
            calibration_para_failed.setInformativeText("As you chose the empiric function for fitting, provide the "
                                                       "corresponding model as empiric = Model(function_empiric).")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        if abs_ is None or lin is None or quad is None:
            abs_ = 1.
            lin = .13
            quad = 1.
            print('#1: Warning! No starting point defined. Default values are: abscissa=1., linear=0.13, quadratic=1..')
        else:
            pass

        # ------------------------------------------------
        print(' [2] Fitparameter abscissa = 1')

        params_empiric = empiric.make_params(abs_=abs_, lin=lin, quad=quad)
        params_empiric['abs_'].vary = False
        params_empiric['abs_'].value = 1.
        params_empiric['quad'].min = 1e-4
        params_empiric['quad'].max = 3e-4
        statistics = pd.DataFrame(np.zeros(shape=(number_runs, len(tsm.param_names)*2 + 1)),
                                  columns=['quadratic', 'quadratic std', 'linear', 'linear std', 'abscissa',
                                           'abscissa std', 'X2'])

        # ------------------------------------------------
        for i in range(number_runs):
            result_empiric = empiric.fit(df[i], x=df.index, params=params_empiric)
            if saving is True:
                name0 = path + '/FitReport/' + method + '_rel-Intensity_' + frequency + '.txt'
                fh = open(name0, 'w')
                fh.write(result_empiric.fit_report())
                fh.close()

            s0 = result_empiric.fit_report().split('\n')[12]
            s1 = result_empiric.fit_report().split('\n')[13]
            s2 = result_empiric.fit_report().split('\n')[14]
            if s0.find('+') == -1:
                if s0.find('.') == -1:
                    statistics.loc[i, 'abscissa'] = np.float(s0[s0.find(':')+1 : s0.find('(')-1])
                    statistics.loc[i, 'abscissa std'] = np.nan
                else:
                    statistics.loc[i, 'abscissa'] = np.float(s0[s0.find('.')-1 : s0.find('(')-1])
                    statistics.loc[i, 'abscissa std'] = np.nan
            else:
                statistics.loc[i, 'abscissa'] = np.float(s0[s0.find('.')-1 : s0.find('+')-1])
                statistics.loc[i, 'abscissa std'] = s0[s0.find('.', s0.find('.')+1)-1 : s0.find('(')]

            if s1.find('+') == -1:
                if s1.find('.') == -1:
                    statistics.loc[i, 'linear'] = np.float(s1[s1.find(':')+1 : s1.find('(')-1])
                    statistics.loc[i, 'linear std'] = np.nan
                else:
                    statistics.loc[i, 'linear'] = np.float(s1[s1.find('.')-1 : s1.find('(')-1])
                    statistics.loc[i, 'linear std'] = np.nan
            else:
                statistics.loc[i, 'linear'] = np.float(s1[s1.find('.')-1 : s1.find('+')-1])
                statistics.loc[i, 'linear std'] = s1[s1.find('.', s1.find('.')+1)-1 : s1.find('(')]

            if s2.find('+') == -1:
                if s2.find('.') == -1:
                    statistics.loc[i, 'quadratic'] = np.float(s2[s2.find(':')+1 : s2.find('(')-1])
                    statistics.loc[i, 'quadratic std'] = np.nan
                else:
                    statistics.loc[i, 'quadratic'] = np.float(s2[s2.find('.')-1 : s2.find('(')-1] )
                    statistics.loc[i, 'quadratic std'] = np.nan
            else:
                statistics.loc[i, 'quadratic'] = np.float(s2[s2.find('.')-1 : s2.find('+')-1])
                statistics.loc[i, 'quadratic std'] = s2[s2.find('.', s2.find('.')+1)-1 : s2.find('(')]

            # goodness of fit (chi square)
            statistics.loc[i, 'X2'] = np.float(result_empiric.fit_report().split('\n')[7].split('=')[1])

            # ----------------------------------------------------------------
            # regression
            y_regression = pd.DataFrame(np.zeros(shape=(len(xdata_reg), number_runs)),
                                        index=xdata_reg+partial_pressure[0])
            for i in range(number_runs):
                y_regression[i] = function_empiric(x=xdata_reg, abs_=statistics.loc[i, 'abscissa'],
                                                   lin=statistics.loc[i, 'linear'], quad=statistics.loc[i, 'quadratic'])
    else:
        raise ValueError('Choose a fitting method: TSM or empiric')

    # **********************************************************************************
    # Plotting data points vs. fit
    if plotting is True:
        alpha = []
        i = 1.
        while i > 0:
            alpha.append(i)
            i -= 1/number_runs

        # -------------------------------------
        fig, ax = plt.subplots()
        ax.errorbar(df.index, df[0], yerr=df_std[0], fmt='s', color='k', alpha=alpha[0], markersize=4, capsize=5,
                    label='data ' + frequency)
        for i in range(number_runs):
            ax.plot(xdata_reg+partial_pressure[0], y_regression[i], color='navy', alpha=alpha[i], lw=1.,
                    label='{} $Χ^2$ = {:.2e}'.format(method, statistics.loc[i, 'X2']))

        ax.legend()
        ax.tick_params(labelsize=fontsize_)
        if type_ == 'intensity':
            ax.set_ylabel('Rel. Intensity pCO2', fontsize=fontsize_)
        elif type_ == 'dphi' or type_ == 'phaseangle' or type_ == 'phase angle' or type_ == 'lifetime':
            ax.set_ylabel('τ0 / τ', fontsize=fontsize_)
        else:
            ax.set_ylabel('signal', fontsize=fontsize_)
        ax.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
        plt.tight_layout()

    # **********************************************************************************
    # Saving
    if saving is True:
        name1 = path + '/Report/' + 'Regression_' + method + '_rel-Intensity_' + frequency + '.txt'
        name2 = path + '/Report/' + 'Statistics_' + method + '_rel-Intensity_' + frequency + '.txt'
        name3 = path + '/Graph/' + 'Fit_plot_' + method + '_' + frequency + '.png'

        if plotting is True:
            fig.savefig(name3, dpi=300)
        y_regression.to_csv(name1, sep='\t', decimal='.')
        statistics.to_csv(name2, sep='\t', decimal='.')

    return statistics, y_regression


def find_closest_value_(index, data, value):
    """
    Find closest value in index
    :param index:
    :param data:
    :param value:
    :return:
    """
    df_ = pd.DataFrame(data, index=index)
    # find closest value and its position
    x_closest = min(df_.index, key=lambda x:abs(x-value))

    # position of closest value
    x_closest_pos = [i for i, x in enumerate(df_.index) if x == x_closest]

    # find next (higher or lower) value
    if x_closest-value < 0:
        x_min = x_closest
        y_min = df_.loc[x_closest].values[0]

        if x_closest_pos[0]+1 >= len(df_.index):
            x_max = df_.index[x_closest_pos[0]]
            y_max = y_min
        else:
            x_max = df_.index[x_closest_pos[0]+1]
            y_max = df_.loc[df_.index[x_closest_pos[0]+1]].values[0]
    else:
        if x_closest_pos[0] == 0:
            x_min = x_closest
            y_min = df_.loc[x_closest].values[0]
        else:
            x_min = df_.index[x_closest_pos[0]-1]
            y_min = df_.loc[df_.index[x_closest_pos[0]-1]].values[0]
        x_max = x_closest
        y_max = df_.loc[x_closest].values[0]

    return x_min, x_max, y_min, y_max


def find_closest_in_list(list_, value):
    # closest value in dataframe
    closest_value = min(list_, key=lambda x:abs(x-value))

    # find corresponding position/index
    position = [i for i, x in enumerate(list_) if x == closest_value]

    if closest_value - value > 0:
        pos_min = position[0]-1
        pos_max = position[0]
        if pos_max > len(list_):
            value_min = None
        else:
            value_min = list_[pos_max]
        if pos_min > len(list_):
            value_max = None
        else:
            value_max = list_[pos_min]
    else:
        if position[0] == len(list_)-1:
            pos_max = position[0]
            pos_min = position[0]
        else:
            pos_max = position[0]+1
            pos_min = position[0]

        if pos_max > len(list_):
            value_min = None
        else:
            value_min = list_[pos_max]
        if pos_min > len(list_):
            value_max = None
        else:
            value_max = list_[pos_min]

    return value_min, value_max, pos_min, pos_max


def find_closest_in_list_(list_, value):
    # closest value in dataframe
    closest_value = min(list_, key=lambda x:abs(x-value))

    # find corresponding position/index
    position = [i for i, x in enumerate(list_) if x == closest_value]

    if closest_value - value > 0:
        pos_min = position[0]-1
        pos_max = position[0]
        if pos_max > len(list_):
            value_min = None
        else:
            value_min = list_.loc[list_.index[pos_max]]
        if pos_min > len(list_):
            value_max = None
        else:
            value_max = list_.loc[list_.index[pos_min]]
    else:
        if position[0] == len(list_)-1:
            pos_max = position[0]
            pos_min = position[0]
        else:
            pos_max = position[0]+1
            pos_min = position[0]

        if pos_max > len(list_):
            value_min = None
        else:
            value_min = list_.loc[list_.index[pos_max]]
        if pos_min > len(list_):
            value_max = None
        else:
            value_max = list_.loc[list_.index[pos_min]]

    return value_min, value_max, list_.index[pos_min], list_.index[pos_max]


def fitting_to_measpoint(index_, data_df, value_meas, value_cal0, value_cal1):
    meas_closest = find_closest_value_(index=index_, data=data_df.values, value=value_meas)
    cal0_closest = find_closest_value_(index=index_, data=data_df.values, value=value_cal0)
    cal1_closest = find_closest_value_(index=index_, data=data_df.values, value=value_cal1)

    if meas_closest[0] == meas_closest[1]:
        y_meas = meas_closest[2]
    else:
        arg_meas = stats.linregress(x=meas_closest[:2], y=meas_closest[2:])
        y_meas = arg_meas[0] * value_meas + arg_meas[1]
    if cal0_closest[0] == cal0_closest[1]:
        y_cal0 = cal0_closest[2]
    else:
        arg_cal0 = stats.linregress(x=cal0_closest[:2], y=cal0_closest[2:])
        y_cal0 = arg_cal0[0] * value_cal0 + arg_cal0[1]
    if cal1_closest[0] == cal1_closest[1]:
        y_cal1 = cal1_closest[2]
    else:
        arg_cal1 = stats.linregress(x=cal1_closest[:2], y=cal1_closest[2:])
        y_cal1 = arg_cal1[0] * value_cal1 + arg_cal1[1]

    return y_cal0, y_cal1, y_meas


def closest(df, val):
    num = df._get_numeric_data()
    num[num < 0] = np.nan
    diff = np.abs(num - val)

    res_axis1 = pd.DataFrame(diff.idxmin().dropna()).reset_index()
    res_axis2 = pd.DataFrame(diff.idxmin(axis=1).dropna()).reset_index()
    res_axis2.columns = ['x', 'y']
    res_axis1.columns = ['y', 'x']

    r1 = res_axis1[['y', 'x']].values.tolist()
    r2 = res_axis2[['y', 'x']].values.tolist()

    select = []
    select_val = []
    for i in r1:
        if i in r2:
            select.append(i)
            select_val.append(diff.loc[i[1], i[0]])  # y vs. x

    loc = select_val.index(min(select_val))

    return select[loc], select_val[loc]


# ---------------------------------------------------------------------------------------------------------------------
# DLR content
# ---------------------------------------------------------------------------------------------------------------------
def lifetime_to_phiP_rad(f, tau):
    return np.arctan(2*np.pi*f*tau)


def phi_to_lifetime_including_error(phi_f1, phi_f2, err_phaseangle, f1, f2, er=True):
    if er is True:
        # phaseangle including assumed measurement uncertainty
        Phi_f1_rad_er = np.deg2rad([phi_f1 - err_phaseangle, phi_f1, phi_f1 + err_phaseangle])
        Phi_f2_rad_er = np.deg2rad([phi_f2 - err_phaseangle, phi_f2, phi_f2 + err_phaseangle])
    else:
        # phaseangle already includes measurement uncertainty
        Phi_f1_rad_er = np.deg2rad(phi_f1)
        Phi_f2_rad_er = np.deg2rad(phi_f2)

    # Lifetime
    tau_1, tau_2 = two_frequency_lifetime(f1=f1, f2=f2, Phi_f1_rad=Phi_f1_rad_er, Phi_f2_rad=Phi_f2_rad_er)
    tau = tau_selection(tau_1, tau_2)

    return tau, Phi_f1_rad_er, Phi_f2_rad_er


def check_lifetime_coincides(tau0, tau1, method_):
    # individual check
    tau_all = np.concatenate((tau0, tau1), axis=0)

    if method_ == 'std':
        tau = [tau_all.min(), tau_all.mean(), tau_all.max()]
    elif method_ == 'absolute':
        tau = [np.array([tau0[0], tau1[0]]).min(), np.array([tau0[1], tau1[1]]).mean(),
               np.array([tau0[2], tau1[2]]).max()]
    else:
        raise ValueError('Define method how to handle incoherence of phosphorescence lifetimes')
    deviation = np.array(tau).std()

    return tau, deviation


def two_frequency_lifetime(f1, f2, Phi_f1_rad, Phi_f2_rad):
    # preparation
    b = f1**2 - f2**2
    sqrt = np.sqrt((f2**2 - f1**2)**2 - 4*((f1**2) * f2*cot(Phi_f2_rad) - f1*(f2**2)*cot(Phi_f1_rad)) *
                   (f2*cot(Phi_f2_rad) - f1*cot(Phi_f1_rad)))

    # denominator and nominator
    z1 = b + sqrt
    z2 = b - sqrt
    n = 4*np.pi*(f1**2*f2*cot(Phi_f2_rad) - f1*f2**2*cot(Phi_f1_rad))
    return z1/n,  z2/n


def tau_selection(tau1, tau2):
    if isinstance(tau1, np.float):
        if tau1 < 0. and tau2 > 0.:
            tau = tau2
        elif tau1 > 0. and tau2 < 0.:
            tau = tau1
        elif tau1 > 0. and tau2 > 0.:
            if tau1 > tau2:
                tau = tau1
        else:
            print(tau1, tau2)
            raise ValueError('some problems occure while selection the lifetime')
    else:
        if tau1[1] < 0. and tau2[1] > 0.:
            tau = tau2
        elif tau1[1] > 0. and tau2[1] < 0.:
            tau = tau1
        elif tau1[1] > 0. and tau2[1] > 0.:
            if tau1[1] > tau2[1]:
                tau = tau1
            else:
                tau = tau2
        else:
            print(tau1, tau2)
            print('Some problems occur while selecting the lifetime')
    return tau


# lifetime to phase angle
def preparation_lifetime(phi_f1_deg, phi_f2_deg, err_phase, f1, f2, method_, er=True):
    # 2-Frequency-Evaluation --> calculate lifetime at each calibration point
    [tau_CO20_c0, Phi_f1_rad_F0P0,
     Phi_f2_rad_F0P0] = phi_to_lifetime_including_error(phi_f1=phi_f1_deg['fluoro0, phosphor0'], f1=f1, f2=f2, er=er,
                                                        phi_f2=phi_f2_deg['fluoro0, phosphor0'],
                                                        err_phaseangle=err_phase)
    [tau_CO21_c0, Phi_f1_rad_F1P0,
     Phi_f2_rad_F1P0]= phi_to_lifetime_including_error(phi_f1=phi_f1_deg['fluoro1, phosphor0'], f1=f1, f2=f2, er=er,
                                                       phi_f2=phi_f2_deg['fluoro1, phosphor0'],
                                                       err_phaseangle=err_phase)
    [tau_CO20_c1, Phi_f1_rad_F0P1,
     Phi_f2_rad_F0P1] = phi_to_lifetime_including_error(phi_f1=phi_f1_deg['fluoro0, phosphor1'], f1=f1, f2=f2, er=er,
                                                        phi_f2=phi_f2_deg['fluoro0, phosphor1'],
                                                        err_phaseangle=err_phase)
    [tau_CO21_c1, Phi_f1_rad_F1P1,
     Phi_f2_rad_F1P1] = phi_to_lifetime_including_error(phi_f1=phi_f1_deg['fluoro1, phosphor1'], f1=f1, f2=f2, er=er,
                                                        phi_f2=phi_f2_deg['fluoro1, phosphor1'],
                                                        err_phaseangle=err_phase)

    # re-check if phosphorescence lifetime coincides at different pH values (return error!)
    tau_c0, dev_tau_c0 = check_lifetime_coincides(tau0=tau_CO20_c0, tau1=tau_CO21_c0, method_=method_)
    tau_c1, dev_tau_c1 = check_lifetime_coincides(tau0=tau_CO20_c1, tau1=tau_CO21_c1, method_=method_)

    # combining phase angle including measurement uncertainty
    Phi_f1_er = pd.Series({'fluoro0, phosphor0': np.rad2deg(Phi_f1_rad_F0P0),
                           'fluoro0, phosphor1': np.rad2deg(Phi_f1_rad_F0P1),
                           'fluoro1, phosphor0': np.rad2deg(Phi_f1_rad_F1P0),
                           'fluoro1, phosphor1': np.rad2deg(Phi_f1_rad_F1P1)})
    Phi_f2_er = pd.Series({'fluoro0, phosphor0': np.rad2deg(Phi_f2_rad_F0P0),
                           'fluoro0, phosphor1': np.rad2deg(Phi_f2_rad_F0P1),
                           'fluoro1, phosphor0': np.rad2deg(Phi_f2_rad_F1P0),
                           'fluoro1, phosphor1': np.rad2deg(Phi_f2_rad_F1P1)})

    parameter = pd.Series({'tau_c0': dev_tau_c0, 'tau_c1': dev_tau_c1})

    return tau_c0, tau_c1, Phi_f1_er, Phi_f2_er, parameter


def preparation_lifetime_2cp(phi_f1_deg, phi_f2_deg, err_phase, f1, f2, er=True):
    # prepare pd.Series
    calib_keys = phi_f1_deg.keys().tolist()
    Phi_f1_er = pd.Series({key: None for key in calib_keys})
    Phi_f2_er = pd.Series({key: None for key in calib_keys})
    para_tau = pd.Series({key: None for key in calib_keys})

    # 2-Frequency-Evaluation --> calculate lifetime at each calibration point
    for i in phi_f1_deg.keys():
        [tau, Phi_f1_rad,
         Phi_f2_rad] = phi_to_lifetime_including_error(phi_f1=phi_f1_deg[i], f1=f1, f2=f2, er=er, phi_f2=phi_f2_deg[i],
                                                       err_phaseangle=err_phase)
        Phi_f1_er[i] = np.rad2deg(Phi_f1_rad)
        Phi_f2_er[i] = np.rad2deg(Phi_f2_rad)
        para_tau[i] = tau

    # [tau_1, Phi_f1_rad_1,
    #  Phi_f2_rad_1] = phi_to_lifetime_including_error(phi_f1=phi_f1_deg[calib_keys[1]], f1=f1, f2=f2, er=er,
    #                                                  phi_f2=phi_f2_deg[calib_keys[1]], err_phaseangle=err_phase)
    # re-check if phosphorescence lifetime coincides at different pH values (return error!)
    # tau_c1, dev_tau_c1 = check_lifetime_coincides(tau0=tau_CO20_c1, tau1=tau_CO21_c1, method_=method_)

    return Phi_f1_er, Phi_f2_er, para_tau


def lifetime_to_superimposed_phaseangle(tauP, ampl_ratio, f):
    if isinstance(tauP, np.float):
        if tauP > 1:
            tauP = tauP*1E-6
        else:
            tauP = tauP
        phiP_rad = np.arctan(2*np.pi*f*tauP)
    else:
        if tauP[1] > 1:
            tauP = [t*1E-6 for t in tauP]
        else:
            tauP = tauP
        phiP_rad = [(np.arctan(2*np.pi*f*tP_)) for tP_ in tauP]
    cot_dPhi = cot(phiP_rad) + 1/(np.sin(phiP_rad)) * ampl_ratio
    Phi_rad = arccot(cot_dPhi)

    return np.rad2deg(Phi_rad)


def phaseangle_from_lifetime_sim(int_ratio, tauP_c0, tauP_c1, f1, f2):
    # Amplitude ratio for 1st modulation frequency
    ampl_ratio_f1 = pd.Series({'pH0, c0': [int_ratio['pH0, c0'] / demodulation(f=f1, tau=tauP_c0[0]),
                                           int_ratio['pH0, c0'] / demodulation(f=f1, tau=tauP_c0[1]),
                                           int_ratio['pH0, c0'] / demodulation(f=f1, tau=tauP_c0[2])],
                               'pH1, c0': [int_ratio['pH1, c0'] / demodulation(f=f1, tau=tauP_c0[0]),
                                           int_ratio['pH1, c0'] / demodulation(f=f1, tau=tauP_c0[1]),
                                           int_ratio['pH1, c0'] / demodulation(f=f1, tau=tauP_c0[2])],
                               'pH0, c1': [int_ratio['pH0, c1'] / demodulation(f=f1, tau=tauP_c1[0]),
                                           int_ratio['pH0, c1'] / demodulation(f=f1, tau=tauP_c1[1]),
                                           int_ratio['pH0, c1'] / demodulation(f=f1, tau=tauP_c1[2])],
                               'pH1, c1': [int_ratio['pH1, c1'] / demodulation(f=f1, tau=tauP_c1[0]),
                                           int_ratio['pH1, c1'] / demodulation(f=f1, tau=tauP_c1[1]),
                                           int_ratio['pH1, c1'] / demodulation(f=f1, tau=tauP_c1[2])]})
    # Amplitude ratio for 2nd modulation frequency
    ampl_ratio_f2 = pd.Series({'pH0, c0': [int_ratio['pH0, c0'] / demodulation(f=f2, tau=tauP_c0[0]),
                                           int_ratio['pH0, c0'] / demodulation(f=f2, tau=tauP_c0[1]),
                                           int_ratio['pH0, c0'] / demodulation(f=f2, tau=tauP_c0[2])],
                               'pH1, c0': [int_ratio['pH1, c0'] / demodulation(f=f2, tau=tauP_c0[0]),
                                           int_ratio['pH1, c0'] / demodulation(f=f2, tau=tauP_c0[1]),
                                           int_ratio['pH1, c0'] / demodulation(f=f2, tau=tauP_c0[2])],
                               'pH0, c1': [int_ratio['pH0, c1'] / demodulation(f=f2, tau=tauP_c1[0]),
                                           int_ratio['pH0, c1'] / demodulation(f=f2, tau=tauP_c1[1]),
                                           int_ratio['pH0, c1'] / demodulation(f=f2, tau=tauP_c1[2])],
                               'pH1, c1': [int_ratio['pH1, c1'] / demodulation(f=f2, tau=tauP_c1[0]),
                                           int_ratio['pH1, c1'] / demodulation(f=f2, tau=tauP_c1[1]),
                                           int_ratio['pH1, c1'] / demodulation(f=f2, tau=tauP_c1[2])]})

    Phi_f1_pH0_c0 = lifetime_to_superimposed_phaseangle(tauP=tauP_c0, ampl_ratio=ampl_ratio_f1['pH0, c0'], f=f1)
    Phi_f1_pH0_c1 = lifetime_to_superimposed_phaseangle(tauP=tauP_c1, ampl_ratio=ampl_ratio_f1['pH0, c1'], f=f1)
    Phi_f1_pH1_c0 = lifetime_to_superimposed_phaseangle(tauP=tauP_c0, ampl_ratio=ampl_ratio_f1['pH1, c0'], f=f1)
    Phi_f1_pH1_c1 = lifetime_to_superimposed_phaseangle(tauP=tauP_c1, ampl_ratio=ampl_ratio_f1['pH1, c1'], f=f1)

    Phi_f2_pH0_c0 = lifetime_to_superimposed_phaseangle(tauP=tauP_c0, ampl_ratio=ampl_ratio_f2['pH0, c0'], f=f2)
    Phi_f2_pH0_c1 = lifetime_to_superimposed_phaseangle(tauP=tauP_c1, ampl_ratio=ampl_ratio_f2['pH0, c1'], f=f2)
    Phi_f2_pH1_c0 = lifetime_to_superimposed_phaseangle(tauP=tauP_c0, ampl_ratio=ampl_ratio_f2['pH1, c0'], f=f2)
    Phi_f2_pH1_c1 = lifetime_to_superimposed_phaseangle(tauP=tauP_c1, ampl_ratio=ampl_ratio_f2['pH1, c1'], f=f2)

    Phi_f1_deg_er = pd.Series({'pH0, c0': Phi_f1_pH0_c0, 'pH0, c1': Phi_f1_pH0_c1, 'pH1, c0': Phi_f1_pH1_c0,
                               'pH1, c1': Phi_f1_pH1_c1})
    Phi_f2_deg_er = pd.Series({'pH0, c0': Phi_f2_pH0_c0, 'pH0, c1': Phi_f2_pH0_c1, 'pH1, c0': Phi_f2_pH1_c0,
                               'pH1, c1': Phi_f2_pH1_c1})

    return Phi_f1_deg_er, Phi_f2_deg_er


def demodulation(f, tau):
    """
    :param f:       float; modulation frequency in Hz
    :param tau:     float; Lifetime in s
    :return:
    """
    if isinstance(tau, np.float):
        if tau > 1:
            tau = tau*1e-6
        dm = 1/np.sqrt(1+(2*np.pi*f*tau)**2)
    else:
        if tau[0] > 1:
            tau = [t*1e-6 for t in tau]
        dm = [1/np.sqrt(1+(2*np.pi*f*t_)**2) for t_ in tau]
    return dm


def int2ampl(i, dm):
    return i * dm


def ampl_to_int_ratio(f1, f2, tauP, dphi_f1, dphi_f2):
    # check if lifetime is given in seconds or mikro seconds
    if isinstance(tauP, np.float):
        if tauP > 1:
            tauP = tauP*1E-6
        else:
            tauP = tauP
    else:
        if tauP[1] > 1:
            tauP = [t*1E-6 for t in tauP]
        else:
            tauP = tauP

    # intensity ratio with helping hands for the lifetime calculation
    if isinstance(tauP, np.float):
        if isinstance(dphi_f1, np.float):
            int_ratio1 = np.array((2*np.pi*f1*tauP -
                          np.tan(np.deg2rad(dphi_f1))) / ((1 + (2*np.pi*f1*tauP)**2)*np.tan(np.deg2rad(dphi_f1))))
            int_ratio2 = np.array((2*np.pi*f2*tauP -
                          np.tan(np.deg2rad(dphi_f2))) / ((1 + (2*np.pi*f2*tauP)**2)*np.tan(np.deg2rad(dphi_f2))))
        else:
            int_ratio1 = np.array([(2*np.pi*f1*tauP -
                          np.tan(np.deg2rad(i))) / ((1 + (2*np.pi*f1*tauP)**2)*np.tan(np.deg2rad(i))) for i in dphi_f1])
            int_ratio2 = np.array([(2*np.pi*f2*tauP -
                          np.tan(np.deg2rad(i))) / ((1 + (2*np.pi*f2*tauP)**2)*np.tan(np.deg2rad(i))) for i in dphi_f2])
    else:
        if isinstance(dphi_f1, np.float):
            int_ratio1 = np.array([(2*np.pi*f1*t -
                           np.tan(np.deg2rad(dphi_f1))) / ((1 + (2*np.pi*f1*t)**2)*np.tan(np.deg2rad(dphi_f1)))
                          for t in tauP])
            int_ratio2 = np.array([(2*np.pi*f2*t -
                          np.tan(np.deg2rad(dphi_f2))) / ((1 + (2*np.pi*f2*t)**2)*np.tan(np.deg2rad(dphi_f2)))
                          for t in tauP])
        else:
            int_ratio1 = np.array([(2*np.pi*f1*t -
                          np.tan(np.deg2rad(i))) / ((1 + (2*np.pi*f1*t)**2)*np.tan(np.deg2rad(i)))
                          for (t, i) in zip(tauP, dphi_f1)])
            int_ratio2 = np.array([(2*np.pi*f2*t -
                          np.tan(np.deg2rad(i))) / ((1 + (2*np.pi*f2*t)**2)*np.tan(np.deg2rad(i)))
                          for (t, i) in zip(tauP, dphi_f2)])

    # re-check if calculated intensities coincide
    int_ratio = []
    if int_ratio1.size > 1:
        for i, j in enumerate(int_ratio1 == int_ratio2):
            if j == False:
                int_ratio.append((int_ratio1[i] + int_ratio2[i]) / 2)
            else:
                int_ratio.append(int_ratio1[i])
    else:
        if int_ratio1 == int_ratio2:
            int_ratio.append(int_ratio1)
        else:
            int_ratio.append((int_ratio1 + int_ratio2) / 2)

    return int_ratio


def int_ratio_to_ampl(f1, f2, pO2_calib, pO2_calc, int_ratio, intP, tauP):
    # distinct intensity intP
    a = find_closest_value_(index=intP.index, data=intP, value=pO2_calib[0])
    b = find_closest_value_(index=intP.index, data=intP, value=pO2_calib[1])
    c = find_closest_value_(index=intP.index, data=intP, value=pO2_calc[1])

    intP_0 = intP.loc[a[0]].values
    intP_1 = intP.loc[b[0]].values
    intP_meas = intP.loc[c[0]].values
    intP_points = pd.Series({'phosphor0': intP_0, 'phosphor1': intP_1, 'meas': intP_meas})

    # distinct lifetimes tauP
    d = find_closest_value_(index=tauP.index, data=tauP, value=pO2_calib[0])
    e = find_closest_value_(index=tauP.index, data=tauP, value=pO2_calib[1])
    f = find_closest_value_(index=tauP.index, data=tauP, value=pO2_calc[1])

    tauP_0 = tauP.loc[d[0]].values
    tauP_1 = tauP.loc[e[0]].values
    tauP_meas = tauP.loc[f[0]].values

    # demodulation at frequency f1 and f2
    dm_f1_0 = demodulation(f=f1, tau=tauP_0)
    dm_f2_0 = demodulation(f=f2, tau=tauP_0)
    dm_f1_1 = demodulation(f=f1, tau=tauP_1)
    dm_f2_1 = demodulation(f=f2, tau=tauP_1)
    dm_f1_meas = demodulation(f=f1, tau=tauP_meas)
    dm_f2_meas = demodulation(f=f2, tau=tauP_meas)

    dm = pd.Series({'f1, phosphor0': dm_f1_0, 'f1, phosphor1': dm_f1_1, 'f2, phosphor0': dm_f2_0,
                    'f2, phosphor1': dm_f2_1, 'f1, meas': dm_f1_meas, 'f2, meas': dm_f2_meas})

    if isinstance(int_ratio['fluoro0, phosphor0'], np.float):
        a_f1_F0P0 = [np.sqrt((int_ratio['fluoro0, phosphor0']**2 + (1 + 2*int_ratio['fluoro0, phosphor0'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f1, phosphor0'], intP_points['phosphor0'])]
        a_f1_F1P0 = [np.sqrt((int_ratio['fluoro1, phosphor0']**2 + (1 + 2*int_ratio['fluoro1, phosphor0'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f1, phosphor0'], intP_points['phosphor0'])]
        a_f1_F0P1 = [np.sqrt((int_ratio['fluoro0, phosphor1']**2 + (1 + 2*int_ratio['fluoro0, phosphor1'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f1, phosphor1'], intP_points['phosphor1'])]
        a_f1_F1P1 = [np.sqrt((int_ratio['fluoro1, phosphor1']**2 + (1 + 2*int_ratio['fluoro1, phosphor1'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f1, phosphor1'], intP_points['phosphor1'])]
        a_f1_meas = [np.sqrt((int_ratio['meas']**2 + (1 + 2*int_ratio['meas'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f1, meas'], intP_points['meas'])]

        a_f2_F0P0 = [np.sqrt((int_ratio['fluoro0, phosphor0']**2 + (1 + 2*int_ratio['fluoro0, phosphor0'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f2, phosphor0'], intP_points['phosphor0'])]
        a_f2_F1P0 = [np.sqrt((int_ratio['fluoro1, phosphor0']**2 + (1 + 2*int_ratio['fluoro1, phosphor0'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f2, phosphor0'], intP_points['phosphor0'])]
        a_f2_F0P1 = [np.sqrt((int_ratio['fluoro0, phosphor1']**2 + (1 + 2*int_ratio['fluoro0, phosphor1'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f2, phosphor1'], intP_points['phosphor1'])]
        a_f2_F1P1 = [np.sqrt((int_ratio['fluoro1, phosphor1']**2 + (1 + 2*int_ratio['fluoro1, phosphor1'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f2, phosphor1'], intP_points['phosphor1'])]
        a_f2_meas = [np.sqrt((int_ratio['meas']**2 + (1 + 2*int_ratio['meas'])*(d**2))*ip**2)
                     for (d, ip) in zip(dm['f2, meas'], intP_points['meas'])]
    else:
        a_f1_F0P0 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro0, phosphor0'], dm['f1, phosphor0'], intP_points['phosphor0'])]
        a_f1_F1P0 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro1, phosphor0'], dm['f1, phosphor0'], intP_points['phosphor0'])]
        a_f1_F0P1 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro0, phosphor1'], dm['f1, phosphor1'], intP_points['phosphor1'])]
        a_f1_F1P1 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro1, phosphor1'], dm['f1, phosphor1'], intP_points['phosphor1'])]
        a_f1_meas = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['meas'], dm['f1, meas'], intP_points['meas'])]

        a_f2_F0P0 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro0, phosphor0'], dm['f2, phosphor0'], intP_points['phosphor0'])]
        a_f2_F1P0 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro1, phosphor0'], dm['f2, phosphor0'], intP_points['phosphor0'])]
        a_f2_F0P1 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro0, phosphor1'], dm['f2, phosphor1'], intP_points['phosphor1'])]
        a_f2_F1P1 = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['fluoro1, phosphor1'], dm['f2, phosphor1'], intP_points['phosphor1'])]
        a_f2_meas = [np.sqrt((i**2 + (1 + 2*i)*(d**2))*ip**2)
                     for (i, d, ip) in zip(int_ratio['meas'], dm['f2, meas'], intP_points['meas'])]

    ampl_total_f1 = pd.Series({'fluoro0, phosphor0': a_f1_F0P0, 'fluoro0, phosphor1': a_f1_F0P1,
                               'fluoro1, phosphor0': a_f1_F1P0, 'fluoro1, phosphor1': a_f1_F1P1, 'meas': a_f1_meas})
    ampl_total_f2 = pd.Series({'fluoro0, phosphor0': a_f2_F0P0, 'fluoro0, phosphor1': a_f2_F0P1,
                              'fluoro1, phosphor0': a_f2_F1P0, 'fluoro1, phosphor1': a_f2_F1P1, 'meas': a_f2_meas})
    return ampl_total_f1, ampl_total_f2, intP_points


def int_ratio_to_ampl_1cp(f1, f2, pO2_calib, pO2_calc, int_ratio, intP, tauP):
    # distinct intensity intP
    a = find_closest_value_(index=intP.index, data=intP, value=pO2_calib)
    c = find_closest_value_(index=intP.index, data=intP, value=pO2_calc[1])

    intP_1 = intP.loc[a[0]].values
    intP_meas = intP.loc[c[0]].values
    intP_points = pd.Series({'phosphor1': intP_1, 'meas': intP_meas})

    # distinct lifetimes tauP
    d = find_closest_value_(index=tauP.index, data=tauP, value=pO2_calib)
    f = find_closest_value_(index=tauP.index, data=tauP, value=pO2_calc[1])

    tauP_1 = tauP.loc[d[0]].values
    tauP_meas = tauP.loc[f[0]].values

    # demodulation at frequency f1 and f2
    dm_f1_1 = demodulation(f=f1, tau=tauP_1)
    dm_f2_1 = demodulation(f=f2, tau=tauP_1)
    dm_f1_meas = demodulation(f=f1, tau=tauP_meas)
    dm_f2_meas = demodulation(f=f2, tau=tauP_meas)

    dm = pd.Series({'f1, phosphor1': dm_f1_1, 'f2, phosphor1': dm_f2_1, 'f1, meas': dm_f1_meas, 'f2, meas': dm_f2_meas})

    keys = int_ratio.keys().tolist()
    ampl_total_f1 = pd.Series({key: None for key in keys})
    ampl_total_f2 = pd.Series({key: None for key in keys})

    if isinstance(int_ratio[0], np.float):
        for i in keys:
            if len(i.split(' ')) > 1:
                phos = i.split(' ')[1]
            else:
                phos = i.split(' ')[0]
            k1 = 'f1, ' + phos
            k2 = 'f2, ' + phos
            ampl_total_f1[i] =[np.sqrt((int_ratio[i]**2 + (1 + 2*int_ratio[i])*(d**2))*ip**2)
                               for (d, ip) in zip(dm[k1], intP_points[phos])]
            ampl_total_f2[i] =[np.sqrt((int_ratio[i]**2 + (1 + 2*int_ratio[i])*(d**2))*ip**2)
                               for (d, ip) in zip(dm[k2], intP_points[phos])]
    else:
        for i in keys:
            if len(i.split(' ')) > 1:
                phos = i.split(' ')[1]
            else:
                phos = i.split(' ')[0]
            k1 = 'f1, ' + phos
            k2 = 'f2, ' + phos
            if phos == 'phosphor0':
                pass
            else:
                ampl_total_f1[i] = [np.sqrt((l**2 + (1 + 2*l)*(d**2))*ip**2)
                                    for (d, ip, l) in zip(dm[k1], intP_points[phos], int_ratio[i])]
                ampl_total_f2[i] = [np.sqrt((l**2 + (1 + 2*l)*(d**2))*ip**2)
                                    for (d, ip, l) in zip(dm[k2], intP_points[phos], int_ratio[i])]

    return ampl_total_f1, ampl_total_f2, intP_points


def intensity_ratio_calculation(f1, f2, tau, phi_f1_deg, phi_f2_deg):
    # combination to Series
    keys = phi_f1_deg.keys().tolist()
    int_ratio_f1 = pd.Series({key: None for key in keys})
    int_ratio_f2 = pd.Series({key: None for key in keys})

    for i in keys:
        if 'meas' in i:
            phos = i
        elif len(i.split(' ')) > 1:
            phos = i.split(' ')[1]
        else:
            phos = i.split(' ')[0]
        int_ratio_f1[i] = sorted(intensity_ratio(f=f1, tau=tau[phos], phi=phi_f1_deg[i]))
        int_ratio_f2[i] = sorted(intensity_ratio(f=f2, tau=tau[phos], phi=phi_f2_deg[i]))

    # re-checking if intensities are independent of modulation frequency
    int_ratio = intensity_ratio_selection(int_ratio_f1, int_ratio_f2)

    return int_ratio


def intP_to_intF(intP, calibP, calcP, int_ratio):
    # intensity based conversion - i_f = int_ratio * i_p - find closest value
    # 1st calibration point phosphor
    Pfit0_min = find_closest_value_(index=intP.index, data=intP[0], value=calibP[0]) # O2_min,O2_max, intP_min,intP_max
    Pfit0_mean = find_closest_value_(index=intP.index, data=intP[1], value=calibP[0])
    Pfit0_max = find_closest_value_(index=intP.index, data=intP[2], value=calibP[0])
    # 2nd calibration point phosphor
    Pfit1_min = find_closest_value_(index=intP.index, data=intP[0], value=calibP[1])
    Pfit1_mean = find_closest_value_(index=intP.index, data=intP[1], value=calibP[1])
    Pfit1_max = find_closest_value_(index=intP.index, data=intP[2], value=calibP[1])

    if Pfit0_mean[0] == Pfit0_mean[1]:
        iP_0 = list(intP.loc[calibP[0]].values)
    else:
        arg_pO2_c0_min = stats.linregress(x=Pfit0_min[:2], y=Pfit0_min[2:])
        arg_pO2_c0_mean = stats.linregress(x=Pfit0_mean[:2], y=Pfit0_mean[2:])
        arg_pO2_c0_max = stats.linregress(x=Pfit0_max[:2], y=Pfit0_max[2:])
        iP_0 = [arg_pO2_c0_min[0] * calibP[0] + arg_pO2_c0_min[1], arg_pO2_c0_mean[0] * calibP[0] + arg_pO2_c0_mean[1],
                arg_pO2_c0_max[0] * calibP[0] + arg_pO2_c0_max[1]]
    if Pfit1_mean[0] == Pfit1_mean[1]:
        iP_1 = list(intP.loc[calibP[1]].values)
    else:
        arg_pO2_c1_min = stats.linregress(x=Pfit1_min[:2], y=Pfit1_min[2:])
        arg_pO2_c1_mean = stats.linregress(x=Pfit1_mean[:2], y=Pfit1_mean[2:])
        arg_pO2_c1_max = stats.linregress(x=Pfit1_max[:2], y=Pfit1_max[2:])
        iP_1 = [arg_pO2_c1_min[0] * calibP[1] + arg_pO2_c1_min[1], arg_pO2_c1_mean[0] * calibP[1] + arg_pO2_c1_mean[1],
                arg_pO2_c1_max[0] * calibP[1] + arg_pO2_c1_max[1]]

    if isinstance(int_ratio['fluoro0, phosphor0'], np.float):
        if isinstance(iP_0, np.float):
            i_fluoro_F0_P0 = int_ratio['fluoro0, phosphor0'] * iP_0
            i_fluoro_F0_P1 = int_ratio['fluoro0, phosphor1'] * iP_1
            i_fluoro_F1_P0 = int_ratio['fluoro1, phosphor0'] * iP_0
            i_fluoro_F1_P1 = int_ratio['fluoro1, phosphor1'] * iP_1
        else:
            i_fluoro_F0_P0 = [int_ratio['fluoro0, phosphor0'] * i for i in iP_0]
            i_fluoro_F0_P1 = [int_ratio['fluoro0, phosphor1'] * i for i in iP_1]
            i_fluoro_F1_P0 = [int_ratio['fluoro1, phosphor0'] * i for i in iP_0]
            i_fluoro_F1_P1 = [int_ratio['fluoro1, phosphor1'] * i for i in iP_1]
    else:
        if isinstance(iP_0, np.float):
            i_fluoro_F0_P0 = [r * iP_0 for r in int_ratio['fluoro0, phosphor0']]
            i_fluoro_F0_P1 = [r * iP_1 for r in int_ratio['fluoro0, phosphor1']]
            i_fluoro_F1_P0 = [r * iP_0 for r in int_ratio['fluoro1, phosphor0']]
            i_fluoro_F1_P1 = [r * iP_1 for r in int_ratio['fluoro1, phosphor1']]
        else:
            i_fluoro_F0_P0 = [r * i for (r, i) in zip(int_ratio['fluoro0, phosphor0'], iP_0)]
            i_fluoro_F0_P1 = [r * i for (r, i) in zip(int_ratio['fluoro0, phosphor1'], iP_1)]
            i_fluoro_F1_P0 = [r * i for (r, i) in zip(int_ratio['fluoro1, phosphor0'], iP_0)]
            i_fluoro_F1_P1 = [r * i for (r, i) in zip(int_ratio['fluoro1, phosphor1'], iP_1)]

    # --------------------------------------------------------------------------------------------------------------
    # find closest value to phosphor calculated
    e = find_closest_value_(index=intP.index, data=intP[2], value=calcP[0])
    b = find_closest_value_(index=intP.index, data=intP[1], value=calcP[1])
    d = find_closest_value_(index=intP.index, data=intP[0], value=calcP[2])

    # lin regression
    arg_b = stats.linregress(x=b[:2], y=b[2:])
    arg_d = stats.linregress(x=d[:2], y=d[2:])
    arg_e = stats.linregress(x=e[:2], y=e[2:])

    intP_meas = [arg_d[0] * calcP[1] + arg_d[1], arg_b[0] * calcP[1] + arg_b[1], arg_e[0] * calcP[1] + arg_e[1]]
    ls_meas_index = []
    for i in int_ratio.index:
        if 'meas' in i:
            ls_meas_index.append(i)
    print(1016)
    i_fluoro_m = {}
    for m in ls_meas_index:
        if isinstance(int_ratio[m], np.float):
            if isinstance(intP_meas, np.float):
                i_fluoro_m[m] = int_ratio[m] * intP_meas
            else:
                i_fluoro_m[m] = [int_ratio[m] * p for p in intP_meas]
        else:
            if isinstance(intP_meas, np.float):
                i_fluoro_m[m] = [r * intP_meas for r in int_ratio[m]]
            else:
                i_fluoro_m[m] = [r * p for (r, p) in zip(int_ratio[m], intP_meas)]
    print(1029)
    i_fluoro_ = pd.Series({'fluoro0, phosphor0': i_fluoro_F0_P0, 'fluoro0, phosphor1': i_fluoro_F0_P1,
                          'fluoro1, phosphor0': i_fluoro_F1_P0, 'fluoro1, phosphor1': i_fluoro_F1_P1})
    i_fluoro = pd.concat([i_fluoro_, pd.Series(i_fluoro_m)], axis=0)

    return i_fluoro


def intP_to_intF_1cp(intP, calibP, calcP, int_ratio):

    # 1-Point-Calibration
    # pO2_0 = 0 hPa (phosphor0)
    Pfit0_min = find_closest_value_(index=intP.index, data=intP[0], value=0.)
    Pfit0_mean = find_closest_value_(index=intP.index, data=intP[1], value=0.)
    Pfit0_max = find_closest_value_(index=intP.index, data=intP[2], value=0.)

    # pO2_1 = 210 hPa (phosphor1)
    Pfit1_min = find_closest_value_(index=intP.index, data=intP[0], value=calibP)
    Pfit1_mean = find_closest_value_(index=intP.index, data=intP[1], value=calibP)
    Pfit1_max = find_closest_value_(index=intP.index, data=intP[2], value=calibP)

    if Pfit0_mean[0] == Pfit0_mean[1]:
        iP_0 = list(intP.loc[0.0].values)
    else:
        arg_pO2_c0_min = stats.linregress(x=Pfit0_min[:2], y=Pfit0_min[2:])
        arg_pO2_c0_mean = stats.linregress(x=Pfit0_mean[:2], y=Pfit0_mean[2:])
        arg_pO2_c0_max = stats.linregress(x=Pfit0_max[:2], y=Pfit0_max[2:])
        iP_0 = [arg_pO2_c0_min[1], arg_pO2_c0_mean[1], calibP + arg_pO2_c0_max[1]]

    if Pfit1_mean[0] == Pfit1_mean[1]:
        iP_1 = list(intP.loc[calibP].values)
    else:
        arg_pO2_c1_min = stats.linregress(x=Pfit1_min[:2], y=Pfit1_min[2:])
        arg_pO2_c1_mean = stats.linregress(x=Pfit1_mean[:2], y=Pfit1_mean[2:])
        arg_pO2_c1_max = stats.linregress(x=Pfit1_max[:2], y=Pfit1_max[2:])
        iP_1 = [arg_pO2_c1_min[0] * calibP + arg_pO2_c1_min[1], arg_pO2_c1_mean[0] * calibP + arg_pO2_c1_mean[1],
                arg_pO2_c1_max[0] * calibP + arg_pO2_c1_max[1]]

    # -------------------------------------------------------------------
    # find closest value to phosphor calculated
    e = find_closest_value_(index=intP.index, data=intP[2], value=calcP[0])
    b = find_closest_value_(index=intP.index, data=intP[1], value=calcP[1])
    d = find_closest_value_(index=intP.index, data=intP[0], value=calcP[2])

    # lin regression
    arg_b = stats.linregress(x=b[:2], y=b[2:])
    arg_d = stats.linregress(x=d[:2], y=d[2:])
    arg_e = stats.linregress(x=e[:2], y=e[2:])

    intP_meas = [arg_d[0] * calcP[1] + arg_d[1], arg_b[0] * calcP[1] + arg_b[1], arg_e[0] * calcP[1] + arg_e[1]]

    # -------------------------------------------------------------------
    keys = int_ratio.keys().tolist()
    i_fluoro = pd.Series({key: None for key in keys})
    for i in keys:
        if len(i.split(' ')) > 1:
            phos = i.split(' ')[1]
        else:
            phos = i.split(' ')[0]
        if phos == 'phosphor0':
            iP = iP_0
        elif phos == 'phosphor1':
            iP = iP_1
        else:
            iP = intP_meas

        if isinstance(int_ratio[i], np.float):
            if isinstance(iP, np.float):
                i_fluoro[i] = int_ratio[i] * iP
            else:
                i_fluoro[i] = [int_ratio[i] * k for k in iP]
        else:
            if isinstance(iP, np.float):
                i_fluoro[i] = [l * iP for l in int_ratio[i]]
            else:
                i_fluoro[i] = [l * k for (k, l) in zip(iP, int_ratio[i])]

    # --------------------------------------------------------------------------------------------------------------
    # find closest value to phosphor calculated
    e = find_closest_value_(index=intP.index, data=intP[2], value=calcP[0])
    b = find_closest_value_(index=intP.index, data=intP[1], value=calcP[1])
    d = find_closest_value_(index=intP.index, data=intP[0], value=calcP[2])

    # lin regression
    arg_b = stats.linregress(x=b[:2], y=b[2:])
    arg_d = stats.linregress(x=d[:2], y=d[2:])
    arg_e = stats.linregress(x=e[:2], y=e[2:])

    intP_meas = [arg_d[0] * calcP[1] + arg_d[1], arg_b[0] * calcP[1] + arg_b[1], arg_e[0] * calcP[1] + arg_e[1]]
    # store the indices of all measurement points (starts with meas) in a list
    ls_meas_index = []
    for i in int_ratio.index:
        if 'meas' in i:
            ls_meas_index.append(i)

    for m in ls_meas_index:
        if isinstance(int_ratio[m], np.float):
            if isinstance(intP_meas, np.float):
                i_fluoro[m] = int_ratio[m] * intP_meas
            else:
                i_fluoro[m] = [int_ratio[m] * p for p in intP_meas]
        else:
            if isinstance(intP_meas, np.float):
                i_fluoro[m] = [r * intP_meas for r in int_ratio[m]]
            else:
                i_fluoro[m] = [r * p for (r, p) in zip(int_ratio[m], intP_meas)]

    return i_fluoro


def phaseangle_DLR(tau_p, f1, f2, I_ratio):
    """

    :param tau_p:       lifetime of the phosphorescent compound (reference) in s
    :param f1:          modulation frequency-1 in Hz
    :param f2:          modulation frequency-2 in Hz
    :param I_ratio:     intensity ratio of the fluorophor and the phosphor
    :return:
    """
    # Input parameter
    # omega
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2

    # demodulation
    dm1 = 1/np.sqrt(1 + (w1*tau_p)**2)
    dm2 = 1/np.sqrt(1 + (w2*tau_p)**2)

    # ratio of the amplitudes
    a_ratio_f1 = I_ratio * 1/dm1
    a_ratio_f2 = I_ratio * 1/dm2

    # life time of the phosphorescence compound (reference)
    dphi_p_rad_f1 = np.arctan(w1 * tau_p)
    dphi_p_rad_f2 = np.arctan(w2 * tau_p)
    dphi_p_deg_f1 = np.rad2deg(dphi_p_rad_f1)
    dphi_p_deg_f2 = np.rad2deg(dphi_p_rad_f2)

    # life time of the mixed (DLR) system
    cot_phi_mixed_f1 = 1/np.tan(dphi_p_rad_f1) + 1/np.sin(dphi_p_rad_f1)*a_ratio_f1
    cot_phi_mixed_f2 = 1/np.tan(dphi_p_rad_f2) + 1/np.sin(dphi_p_rad_f2)*a_ratio_f2
    dphi_mixed_rad_f1 = np.arctan(1/cot_phi_mixed_f1)
    dphi_mixed_rad_f2 = np.arctan(1/cot_phi_mixed_f2)
    dphi_mixed_deg_f1 = np.rad2deg(dphi_mixed_rad_f1)
    dphi_mixed_deg_f2 = np.rad2deg(dphi_mixed_rad_f2)

    return dphi_p_rad_f1, dphi_p_rad_f2, dphi_p_deg_f1, dphi_p_deg_f2, dphi_mixed_rad_f1, \
           dphi_mixed_rad_f2, dphi_mixed_deg_f1, dphi_mixed_deg_f2, a_ratio_f1, a_ratio_f2


def superimposed_phaseangle_er(tauP_discret, ampl_ratio, f, er_phase, decimal=6):
    phi_deg_F0_P0 = lifetime_to_superimposed_phaseangle(tauP=tauP_discret['phosphor0'],
                                                        ampl_ratio=ampl_ratio['fluoro0, phosphor0'], f=f)
    phi_deg_F0_P1 = lifetime_to_superimposed_phaseangle(tauP=tauP_discret['phosphor1'],
                                                        ampl_ratio=ampl_ratio['fluoro0, phosphor1'], f=f)
    phi_deg_F1_P0 = lifetime_to_superimposed_phaseangle(tauP=tauP_discret['phosphor0'],
                                                        ampl_ratio=ampl_ratio['fluoro1, phosphor0'], f=f)
    phi_deg_F1_P1 = lifetime_to_superimposed_phaseangle(tauP=tauP_discret['phosphor1'],
                                                        ampl_ratio=ampl_ratio['fluoro1, phosphor1'], f=f)

    phi_deg_meas = lifetime_to_superimposed_phaseangle(tauP=tauP_discret['meas'],  ampl_ratio=ampl_ratio['meas'], f=f)

    Phi_deg = pd.Series({'fluoro0, phosphor0': phi_deg_F0_P0, 'fluoro0, phosphor1': phi_deg_F0_P1,
                         'fluoro1, phosphor0': phi_deg_F1_P0, 'fluoro1, phosphor1': phi_deg_F1_P1,
                         'meas': phi_deg_meas})

    # -------------------------------------------------------------------------------------------
    # spread uncertainty in phase angle
    phi_F0P0_er = [phi_deg_F0_P0 - er_phase, phi_deg_F0_P0, phi_deg_F0_P0 + er_phase]
    phi_F0P1_er = [phi_deg_F0_P1 - er_phase, phi_deg_F0_P1, phi_deg_F0_P1 + er_phase]
    phi_F1P0_er = [phi_deg_F1_P0 - er_phase, phi_deg_F1_P0, phi_deg_F1_P0 + er_phase]
    phi_F1P1_er = [phi_deg_F1_P1 - er_phase, phi_deg_F1_P1, phi_deg_F1_P1 + er_phase]
    phi_meas = [phi_deg_meas - er_phase, phi_deg_meas, phi_deg_meas + er_phase]
    Phi_deg_er = pd.Series({'fluoro0, phosphor0': [round(i, decimal) for i in phi_F0P0_er],
                            'fluoro0, phosphor1': [round(i, decimal) for i in phi_F0P1_er],
                            'fluoro1, phosphor0': [round(i, decimal) for i in phi_F1P0_er],
                            'fluoro1, phosphor1': [round(i, decimal) for i in phi_F1P1_er],
                            'meas': [round(i, decimal) for i in phi_meas]})

    return Phi_deg, Phi_deg_er


def lifetime(phi1, phi2, f1, f2):
    """
    Calculation of lifetime of the fluorescence/phosphorescence DLR system. Only the overall
     phase shift (tau1, tau2 each in deg) at two different modulation frequencies (in Hz) is
     given.
    :param phi1:    overall phase shift at modulation frequency f1 in deg
    :param phi2:    overall phase shift at modulation frequency f2 in deg
    :param f1:      modulation frequency in Hz
    :param f2:      modulation frequency in Hz
    :return: tau1     life time of the phosphorescent component in s; usually positive solution
    :return: tau2     life time of the phosphorescent component in s; usually negative solution
    """
    phi1_rad = np.deg2rad(phi1)
    phi2_rad = np.deg2rad(phi2)
    cot1 = 1 / np.tan(phi1_rad)
    cot2 = 1 / np.tan(phi2_rad)

    if isinstance(f1, float) or type(f1) == np.int64 or isinstance(f1, int):
        b = f2**2 - f1**2
        a = 2*np.pi * ((f1**2)*f2*cot2 - f1*(f2**2)*cot1)
        c = 1/(2*np.pi) * (f2*cot2 - f1*cot1)

        # phase shift if argument for square negative!?
        if (b**2 - 4*a*c < 0).all() == True:
            print('square is negative!')
            if (cot1 < 0).all() == True or (cot2 < 0).all() == True:
                cot1 = 1 / np.tan(np.deg2rad(phi1 + 45))
                cot2 = 1 / np.tan(np.deg2rad(phi2 + 45))

            a = (f1**2)*f2*cot2 - f1*(f2**2)*cot1
            c = f2*cot2 - f1*cot1

        tau1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        tau2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    else:
        phi_combinations = []
        tau1 = []
        tau2 = []
        for xs in itertools.product(f1, f2):
            phi_combinations.append(xs)
            if xs[0] != xs[1]:
                b = xs[1]**2 - xs[0]**2
                a = 2*np.pi*((xs[0]**2)*xs[1]*cot2.ix[xs[1], 'Phi_mixed'] - xs[0]*(xs[1]**2)*cot1.ix[xs[0], 'Phi_mixed'])
                c = 1/(2*np.pi)*(xs[1]*cot2.ix[xs[1], 'Phi_mixed'] - xs[0]*cot1.ix[xs[0], 'Phi_mixed'])

                # phase shift if argument for square negative!?
                if (b**2 - 4*a*c < 0).all() == True:
                    if (cot1 < 0).all() == True or (cot2 < 0).all() == True:
                        cot1 = 1 / np.tan(np.deg2rad(phi1 + 45))
                        cot2 = 1 / np.tan(np.deg2rad(phi2 + 45))

                    a = (f1**2)*f2*cot2 - f1*(f2**2)*cot1
                    c = f2*cot2 - f1*cot1

                tau1.append((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
                tau2.append((-b - np.sqrt(b**2 - 4*a*c)) / (2*a))
            else:
                tau1.append(np.nan)
                tau2.append(np.nan)

    return tau1, tau2


def amplitude_ratio(intP_discret, tauP_discret, i_f_discret, f1, f2):
    # amplitude ratio at 2 different modulation frequencies
    amplP_f1_p0 = int2ampl(intP_discret['phosphor0'], dm=demodulation(f=f1, tau=tauP_discret['phosphor0']))
    amplP_f1_p1 = int2ampl(intP_discret['phosphor1'], dm=demodulation(f=f1, tau=tauP_discret['phosphor1']))
    amplP_f1_meas = int2ampl(intP_discret['meas'], dm=demodulation(f=f1, tau=tauP_discret['meas']))

    amplP_f2_p0 = int2ampl(intP_discret['phosphor0'], dm=demodulation(f=f2, tau=tauP_discret['phosphor0']))
    amplP_f2_p1 = int2ampl(intP_discret['phosphor1'], dm=demodulation(f=f2, tau=tauP_discret['phosphor1']))
    amplP_f2_meas = int2ampl(intP_discret['meas'], dm=demodulation(f=f2, tau=tauP_discret['meas']))

    ampl_ratio_f1 = pd.Series({'fluoro0, phosphor0': i_f_discret['fluoro0'] / amplP_f1_p0,
                               'fluoro1, phosphor0': i_f_discret['fluoro1'] / amplP_f1_p0,
                               'fluoro0, phosphor1': i_f_discret['fluoro0'] / amplP_f1_p1,
                               'fluoro1, phosphor1': i_f_discret['fluoro1'] / amplP_f1_p1,
                               'meas': i_f_discret['meas'] / amplP_f1_meas})

    ampl_ratio_f2 = pd.Series({'fluoro0, phosphor0': i_f_discret['fluoro0'] / amplP_f2_p0,
                               'fluoro1, phosphor0': i_f_discret['fluoro1'] / amplP_f2_p0,
                               'fluoro0, phosphor1': i_f_discret['fluoro0'] / amplP_f2_p1,
                               'fluoro1, phosphor1': i_f_discret['fluoro1'] / amplP_f2_p1,
                               'meas': i_f_discret['meas'] / amplP_f2_meas})

    return ampl_ratio_f1, ampl_ratio_f2


def intensity_ratio(f, tau, phi):
    """

    :param f:       float; modulation frequency in Hz
    :param tau:     list; lifetime of the phosphor in µs
    :param phi:     list; measured phase angle including measurement uncertainty in degree
    :return:        pd.Dataframe; intensity ratio I_F / I_P
    """
    # check if phi in deg and tau in seconds
    if isinstance(tau, np.float):
        if tau > 1:
            tau = tau*1e-6
    else:
        if tau[1] > 1:
            tau = [t*1E-6 for t in tau]
    Phi = np.deg2rad(phi)

    # denominator and nominator
    if isinstance(tau, np.float):
        z_f = [2*np.pi*f*tau - np.tan(phi) for phi in Phi]
        n_f = [(1 + (2*np.pi*f*tau)**2)*np.tan(phi) for phi in Phi]
    else:
        z_f = [2*np.pi*f*t - np.tan(phi) for (t, phi) in zip(tau, Phi)]
        n_f = [(1 + (2*np.pi*f*t)**2)*np.tan(phi) for (t, phi) in zip(tau, Phi)]

    # intensity ratio I_F / I_P remain constant for different modulation frequencies
    i_ratio = [z_ / n_ for (z_, n_) in zip(z_f, n_f)]

    return i_ratio


def intensity_ratio_(tau, phi, f):
    """
    Backup compared to intensity_ratio for individual values (floats, ints...) instead of arrays
    Intensity ratio of fluorescence to phosphorescence compound.
    :param tau:     life time of the phosphorescent compound in s
    :param phi:     overall phase shift at modulation frequency f in deg
    :param f:       modulation frequency in Hz
    :return: ratio: intensity ratio of fluorescence to phosphorescence component
    """
    # Conversion into correct unit
    phi_rad = np.deg2rad(phi)

    # helping hands
    w = 2*np.pi*f
    wt = w*tau

    numerator = wt - np.tan(phi_rad)
    denominator = np.tan(phi_rad) * (1 + wt**2)

    # intensity ratio
    ratio = numerator / denominator
    if type(ratio) == np.float:
        ratio = round(ratio, 7)
    else:
        ratio = np.around(ratio, 7)

    return ratio


def intensity_ratio_selection(int_ratio_f1, int_ratio_f2):
    for i in int_ratio_f1.index:
        l = list(np.array(int_ratio_f1[i]) - np.array(int_ratio_f2[i]))
        mean_list = sum(l) / np.float(len(l))
        if mean_list > 0.1:
            raise ValueError('Something went wrong for the calculation of the intensity ratio...')
        else:
            int_ratio = int_ratio_f1

    return int_ratio


# --------------------------------------------------------
def error_phaseangle(Phi_deg, er_phase):
    list_ = []
    for i in Phi_deg.index:
        for j in Phi_deg.columns:
            if i == 'meas' and j != 'meas':
                pass
            elif i != 'meas' and j != 'meas' or i=='meas' and j=='meas':
                if isinstance(Phi_deg.loc[i, j], np.float):
                    list_.append([Phi_deg.loc[i, j] - er_phase, Phi_deg.loc[i, j], Phi_deg.loc[i, j] + er_phase])
                else:
                    list_.append(Phi_deg.loc[i, j])
    return list_


def error(Res1, Res2, ref_tau, ref_tau_neg, Intensity1, Intensity2, ref_y, ref_y_neg):
    error_tau1_rel = (Res1 - ref_tau) / ref_tau * 100
    error_tau2_rel = (Res2 - ref_tau_neg) / ref_tau_neg * 100

    error_intensity1_rel = (Intensity1 - ref_y) / ref_y * 100
    error_intensity2_rel = (Intensity2 - ref_y_neg) / ref_y_neg * 100
    error_intensity1_abs = (Intensity1 - ref_y)
    error_intensity2_abs = (Intensity2 - ref_y_neg)

    return error_tau1_rel, error_tau2_rel, error_intensity1_abs, error_intensity2_abs, error_intensity1_rel, \
           error_intensity2_rel


def dlr_ouput(tau_p, f1, f2, I_ratio, a_ratio_f1, a_ratio_f2, p1, p2, er, Res1_mis, error_tau1_rel, Intensity1,
              error_intensity1_abs, error_intensity1_rel, save=False):
    if type(Res1_mis) == float:
        print('No error propagation for this frequency combination:', f1, f2, 'Hz')
        parameter = None
        output = None
    else:
        output = pd.concat([Res1_mis, error_tau1_rel, Intensity1, error_intensity1_abs, error_intensity1_rel])

        parameter = pd.DataFrame({'value': {'tau': round(tau_p*1E6, 4), 'f1': f1/1000,  'f2': f2/1000,
                                            'I(F:P)': I_ratio, 'A_ratio(f1)': math.ceil(a_ratio_f1*1000)/1000,
                                            'A_ratio(f2)': math.ceil(a_ratio_f2*1000)/1000,  'phi(f1)': p1,
                                            'phi(f2)': p2, 'dphi±': er},
                                  'unit': {'tau': 'µs', 'f1': 'Hz', 'f2': 'Hz', 'I(F:P)': '', 'A_ratio(f1)': '',
                                           'A_ratio(f2)': '', 'phi(f1)': '°', 'phi(f2)': '°', 'dphi±': '°'}})

        print(colored('DLR calculations \n', 'blue'))
        print('Input')
        print('Modulation frequencies:')
        print('\t \t \t f1 = ', f1/1000, 'kHz')
        print('\t \t \t f2 = ', f2/1000, 'kHz')
        print('intensity ratio (Fl:P) = ', I_ratio)
        print('amplitude ratio:', )
        print('\t \t \t A_ratio(f1) = ', math.ceil(a_ratio_f1*1000)/1000)
        print('\t \t \t A_ratio(f2) = ', math.ceil(a_ratio_f2*1000)/1000)
        print('superimposed phase angles:')
        print('\t \t \t Φ(f1) = ', round(p1, 4), '°')
        print('\t \t \t Φ(f2) = ', round(p2, 4), '°')
        print('assumed uncertainty: ±', er, '° \n \n')

        print('|#| life time')
        print(colored('absolute values [µs]', 'green'))
        print(tabulate(Res1_mis,
                       headers=['Φ(f2) \ Φ(f1)', round(error_tau1_rel.columns.values[0], 2),
                                round(error_tau1_rel.columns.values[1], 2), round(error_tau1_rel.columns.values[2], 2)],
                       floatfmt=".2f", tablefmt="fancy_grid", showindex=True))

        print(colored('relative error [%]', 'green'))
        print(tabulate(error_tau1_rel,
                       headers=['Φ(f2) \ Φ(f1)', round(error_tau1_rel.columns.values[0], 2),
                                round(error_tau1_rel.columns.values[1], 2), round(error_tau1_rel.columns.values[2], 2)],
                       floatfmt=".2f", tablefmt="fancy_grid", showindex=True))

        print('\n')
        print('|#| Intensity')
        print(colored('absolute values', 'green'))
        print(tabulate(Intensity1,
                       headers=['I(F) / I(P)', round(error_tau1_rel.columns.values[0], 2),
                                round(error_tau1_rel.columns.values[1], 2), round(error_tau1_rel.columns.values[2], 2)],
                       floatfmt=".4f", tablefmt="fancy_grid", showindex=True))

        print(colored('absolute error [%]', 'green'))
        print(tabulate(error_intensity1_abs,
                       headers=['I(F) / I(P)', round(error_tau1_rel.columns.values[0], 2),
                                round(error_tau1_rel.columns.values[1], 2), round(error_tau1_rel.columns.values[2], 2)],
                       floatfmt=".4f", tablefmt="fancy_grid", showindex=True))

        print(colored('relative error [%]', 'green'))
        print(tabulate(error_intensity1_rel,
                       headers=['I(F) / I(P)', round(error_tau1_rel.columns.values[0], 2),
                                round(error_tau1_rel.columns.values[1], 2), round(error_tau1_rel.columns.values[2], 2)],
                       floatfmt=".2f", tablefmt="fancy_grid", showindex=True))
        if save is True:
            # Input parameter -> DataFrame with no special index (0-8)
            parameter2 = pd.DataFrame(np.zeros(shape=(len(parameter.index), len(parameter.columns)+1)))
            parameter2.ix[:, 0] = parameter.index
            parameter2.ix[:, 1] = parameter.ix[:, 'value'].values
            parameter2.ix[:, 2] = parameter.ix[:, 'unit'].values

            # error calculated -> DataFrame with no special index
            # table for lifetime
            err_lifetime = pd.DataFrame(np.zeros(shape=(len(Res1_mis.index)+1, len(Res1_mis.columns)*2+1)))
            for u, v in enumerate(Res1_mis.columns):
                err_lifetime.ix[1:, u+1] = Res1_mis.ix[:, v].values
            for d, g in enumerate(error_tau1_rel):
                err_lifetime.ix[1:, d+4] = error_tau1_rel.ix[:, g].values
            t_index = []
            t_index.extend(['Phi1 \ Phi2'])
            t_index.extend(Res1_mis.index)
            err_lifetime.ix[:, 0] = t_index
            t_col = []
            t_col.extend(['Phi1 \ Phi2'])
            t_col.extend(Res1_mis.columns)
            t_col.extend(Res1_mis.columns)
            err_lifetime.ix[0, :] = t_col

            # table for intensity ratio
            err_itensity = pd.DataFrame(np.zeros(shape=(len(Intensity1.index)+1, 3*len(Intensity1.columns)+1)))

            for u, v in enumerate(Intensity1.columns):
                err_itensity.ix[1:, u+1] = Intensity1.ix[:, v].values
            for d, g in enumerate(error_intensity1_abs):
                err_itensity.ix[1:, d+4] = error_intensity1_abs.ix[:, g].values
            for h, i in enumerate(error_intensity1_rel):
                err_itensity.ix[1:, h+7] = error_intensity1_rel.ix[:, i].values
            t_index2 = []
            t_index2.extend(['Phi1 \ Phi2 - rel'])
            t_index2.extend(Intensity1.index)
            err_itensity.ix[:, 0] = t_index2
            t_col2 = []
            t_col2.extend(['I_ratrio - abs -  rel'])
            t_col2.extend(Intensity1.columns)
            t_col2.extend(Intensity1.columns)
            t_col2.extend(Intensity1.columns)
            err_itensity.ix[0, :] = t_col2

            output_save = pd.concat([parameter2, err_lifetime, err_itensity], ignore_index=True)

            directory = 'D:/01_data_processing/dualsensor/errorpropagation_frequency-pair/{}Hz/'.format(f1)
            if not os.path.exists(directory):
                os.makedirs(directory)
            saving_name = directory + '{}_errorcalculation_{}-{}Hz.txt'.format(today.isoformat(), round(f1, 2),
                                                                               round(f2, 2))

            pd.DataFrame(output_save).to_csv(saving_name, sep='\t', index=False, header=False)

    return parameter, output


def spread_imperfection(phi1, phi2, er, f1, f2, select):
    # error range for lifetime and intensity ratio
    # phase shift array with assumed absolute error [°]
    phi_1 = [round(phi1-er, 4), phi1, round(phi1+er, 4)]
    phi_2 = [round(phi2-er, 4), phi2, round(phi2+er, 4)]

    # total phase shift in deg
    p1 = phi1
    p2 = phi2

    # Life time for the whole range
    Phi1, Phi2 = np.meshgrid(phi_1, phi_2)
    [Tau1, Tau2] = lifetime(Phi1, Phi2, f1, f2)

    if select == 1:
        Tau = Tau1
        Tau_neg = Tau2
    elif select == 2:
        Tau = Tau2
        Tau_neg = Tau1
    elif select == 3:
        # if tau1 = tau2 = np.nan
        Tau = Tau1
    else:
        raise ValueError('There might be a problem with the life times...')

    # DataFrame
    if np.isnan(Tau[0][0]) == True:
        Res1 = np.nan
        Res2 = np.nan
        Res1_mikros = np.nan
        Res2_mikros = np.nan
        Intensity1 = np.nan
        Intensity2 = np.nan
    else:
        Res1 = pd.DataFrame(Tau, index=phi_2, columns=phi_1)
        Res2 = pd.DataFrame(Tau_neg, index=phi_2, columns=phi_1)
        Res1_mikros = Res1*1E6
        Res2_mikros = Res2*1E6

        # Intensity ratio
        Ratio1 = intensity_ratio_(tau=Tau, phi=Phi1, f=f1)
        Ratio3 = intensity_ratio_(tau=Tau_neg, phi=Phi1, f=f1)

        # DataFrame
        Intensity1 = pd.DataFrame(Ratio1, index=phi_2, columns=phi_1)
        Intensity2 = pd.DataFrame(Ratio3, index=phi_2, columns=phi_1)

    return Res1, Res2, Intensity1, Intensity2, p1, p2, Res1_mikros, Res2_mikros


def phaseangle_dynamics(f_start, f_end, tau_phosphor):
    tau = tau_phosphor*1E-6
    f_test = np.arange(f_start, f_end)
    ddphi = pd.DataFrame(np.zeros(shape=(len(f_test), 3)), index=f_test,
                         columns=['I-ratio 0', 'I-ratio 1', 'ddphi'])

    for i in f_test:
        w = 2*np.pi*i
        nenner = w * tau
        denominator_0 = 0*(1 + (w*tau)**2) + 1
        denominator_1 = 1*(1 + (w*tau)**2) + 1

        tan_phi_0 = nenner/denominator_0
        tan_phi_1 = nenner/denominator_1
        ddphi.ix[i, 'I-ratio 0'] = math.ceil(np.rad2deg(float(atan(tan_phi_0)))*1000)/1000
        ddphi.ix[i, 'I-ratio 1'] = math.ceil(np.rad2deg(float(atan(tan_phi_1)))*1000)/1000

        delta = math.ceil((np.rad2deg(float(atan(tan_phi_0))) - np.rad2deg(float(atan(tan_phi_1))))*100)/100
        ddphi.ix[i, 'ddphi'] = delta

    return ddphi


def error_propagation_DLR(tau_p, f1, f2, I_ratio, er, save_output):

    # superimposed phase angle
    [dphi_p_rad_f1, dphi_p_rad_f2, dphi_p_deg_f1, dphi_p_deg_f2, dphi_mixed_rad_f1, dphi_mixed_rad_f2,
     dphi_mixed_deg_f1, dphi_mixed_deg_f2, a_ratio_f1, a_ratio_f2] = phaseangle_DLR(tau_p, f1, f2, I_ratio)

    # superimposed phase angle in degree
    phi1 = dphi_mixed_deg_f1
    phi2 = dphi_mixed_deg_f2

    # modulation frequencies in Hz
    f1 = f1
    f2 = f2

    # -----------------------------------------------------------------------------------------------------
    # Reference point
    # Lifetimes
    [x1, x2] = lifetime(phi1=phi1, phi2=phi2, f1=f1, f2=f2)

    # Intensity ratio
    y1 = intensity_ratio_(tau=x1, phi=phi1, f=f1)
    y2 = intensity_ratio_(tau=x1, phi=phi2, f=f2)
    y3 = intensity_ratio_(tau=x2, phi=phi1, f=f1)
    y4 = intensity_ratio_(tau=x2, phi=phi2, f=f2)

    if x1 < 0 and x2 > 0:
        select = 2
        ref_tau = x2
        ref_tau_neg = x1
        ref_y = y3
        ref_y_neg = y1
    elif x2 < 0 and x1 > 0:
        select = 1
        ref_tau = x1
        ref_tau_neg = x2
        ref_y = y1
        ref_y_neg = y3
    elif x1 > 0 and x2 > 0:
        if abs(x1 - tau_p) < abs(x2 - tau_p):
            select = 1
            ref_tau = x1
            ref_tau_neg = x2
            ref_y = y1
            ref_y_neg = y3
        else:
            select = 2
            ref_tau = x2
            ref_tau_neg = x1
            ref_y = y3
            ref_y_neg = y1
    else:
        select = 3
        ref_tau = np.nan
        ref_tau_neg = np.nan
        ref_y = np.nan
        ref_y_neg = np.nan

    # -------------------------------------------------------------------------------------------------------
    # error range for lifetime and intensity ratio
    [Res1, Res2, Intensity1, Intensity2, p1, p2, Res1_mikros, Res2_mikros] = spread_imperfection(phi1, phi2, er, f1, f2,
                                                                                                 select)

    # ----------------------------------------------------------------------------------------------
    # error calculation
    if select == 3:
        parameter = []
        output = []
    else:
        [e_tau1_rel, e_tau2_rel, e_intensity1_abs, e_intensity2_abs, e_intensity1_rel,
         e_intensity2_rel] = error(Res1, Res2, ref_tau, ref_tau_neg, Intensity1, Intensity2, ref_y, ref_y_neg)

        # Output - error propagation in DLR systems
        [parameter, output] = dlr_ouput(tau_p, f1, f2, I_ratio, a_ratio_f1, a_ratio_f2, p1, p2, er, Res1_mikros,
                                        e_tau1_rel, Intensity1, e_intensity1_abs, e_intensity1_rel, save=save_output)

    return parameter, output, Res1, Res2, Intensity1, Intensity2


# ---------------------------------------------------------------------------------------
# Functions optimized for GUI
def phaseangle_DLR_GUI(tau_p, f1, f2, I_ratio):
    """
    :param tau_p:       lifetime of the phosphorescent compound (reference) in s
    :param f1:          modulation frequency-1 in Hz
    :param f2:          modulation frequency-2 in Hz
    :param I_ratio:     intensity ratio of the fluorophor and the phosphor
    :return:
    """
    # Input parameter
    # omega
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2

    # demodulation
    dm1 = np.sqrt(1 + (w1*tau_p)**2)
    dm2 = np.sqrt(1 + (w2*tau_p)**2)

    # ratio of the amplitudes
    a_ratio_f1 = I_ratio * dm1
    a_ratio_f2 = I_ratio * dm2

    # life time of the phosphorescence compound (reference)
    dphi_p_rad_f1 = np.arctan(w1 * tau_p)
    dphi_p_rad_f2 = np.arctan(w2 * tau_p)

    # life time of the mixed (DLR) system
    cot_phi_mixed_f1 = 1/np.tan(dphi_p_rad_f1) + 1/np.sin(dphi_p_rad_f1)*a_ratio_f1
    cot_phi_mixed_f2 = 1/np.tan(dphi_p_rad_f2) + 1/np.sin(dphi_p_rad_f2)*a_ratio_f2

    dphi_mixed_deg_f1 = np.rad2deg(np.arctan(1/cot_phi_mixed_f1))
    dphi_mixed_deg_f2 = np.rad2deg(np.arctan(1/cot_phi_mixed_f2))

    return dphi_mixed_deg_f1, dphi_mixed_deg_f2


def spread_imperfection_GUI(phi1, phi2, er, f1, f2, select):

    # error range for lifetime and intensity ratio
    # phase shift array with assumed absolute error [°]
    phi_1 = [np.float64(round(phi1-er, 4)), np.float64(phi1), np.float64(round(phi1+er, 4))]
    phi_2 = [np.float64(round(phi2-er, 4)), np.float64(phi2), np.float64(round(phi2+er, 4))]

    # total phase shift in deg
    p1 = phi1
    p2 = phi2

    # Life time for the whole range
    Phi1, Phi2 = np.meshgrid(phi_1, phi_2)
    [Tau1, Tau2] = lifetime(Phi1, Phi2, f1, f2)

    if select == 1:
        Tau = Tau1
        Tau_neg = Tau2
    elif select == 2:
        Tau = Tau2
        Tau_neg = Tau1
    elif select == 3:
        # if tau1 = tau2 = np.nan
        Tau = Tau1
    else:
        raise ValueError('There might be a problem with the life times...')

    # DataFrame
    if np.isnan(Tau[1][1]) == True:
        Res1 = np.nan
        Res1_mikros = np.nan
        Intensity1 = np.nan
    else:
        Res1 = pd.DataFrame(Tau, index=phi_2, columns=phi_1)
        Res1_mikros = Res1*1E6

        # Intensity ratio
        Ratio1 = intensity_ratio_(tau=Tau, phi=Phi1, f=f1)

        # DataFrame
        Intensity1 = pd.DataFrame(Ratio1, index=phi_2, columns=phi_1)

    return Res1, Intensity1, p1, p2, Res1_mikros


def error_GUI(Res1, ref_tau, Intensity1, ref_y):
    error_tau1_rel = (Res1 - ref_tau) / ref_tau * 100

    error_intensity1_rel = (Intensity1 - ref_y) / ref_y * 100
    error_intensity1_abs = (Intensity1 - ref_y)

    return error_tau1_rel, error_intensity1_abs, error_intensity1_rel


def dlr_ouput_GUI(tau_p, f1, f2, p1, p2, er, I_ratio, Res1_mis, error_tau1_rel, Intensity1, error_intensity1_abs,
                  error_intensity1_rel, save=False):
    if type(Res1_mis) == float:
        parameter = None
    else:
        parameter = pd.DataFrame({'value': {'tau': round(tau_p*1E6, 2), 'f1': f1/1000,  'f2': f2/1000,
                                            'I(F:P)': I_ratio, 'phi(f1)': p1, 'phi(f2)': p2,
                                            'dphi±': er},
                                  'unit': {'tau': 'µs', 'f1': 'Hz', 'f2': 'Hz', 'I(F:P)': '', 'phi(f1)': '°',
                                           'phi(f2)': '°', 'dphi±': '°'}})

        if save is True:
            # Input parameter -> DataFrame with no special index (0-8)
            parameter2 = pd.DataFrame(np.zeros(shape=(len(parameter.index), len(parameter.columns)+1)))
            parameter2.ix[:, 0] = parameter.index
            parameter2.ix[:, 1] = parameter.ix[:, 'value'].values
            parameter2.ix[:, 2] = parameter.ix[:, 'unit'].values

            # error calculated -> DataFrame with no special index
            # table for lifetime
            err_lifetime = pd.DataFrame(np.zeros(shape=(len(Res1_mis.index)+1, len(Res1_mis.columns)*2+1)))
            for u, v in enumerate(Res1_mis.columns):
                err_lifetime.ix[1:, u+1] = Res1_mis.ix[:, v].values
            for d, g in enumerate(error_tau1_rel):
                err_lifetime.ix[1:, d+4] = error_tau1_rel.ix[:, g].values
            t_index = []
            t_index.extend(['Phi1 \ Phi2'])
            t_index.extend(Res1_mis.index)
            err_lifetime.ix[:, 0] = t_index
            t_col = []
            t_col.extend(['Phi1 \ Phi2'])
            t_col.extend(Res1_mis.columns)
            t_col.extend(Res1_mis.columns)
            err_lifetime.ix[0, :] = t_col

            # table for intensity ratio
            err_itensity = pd.DataFrame(np.zeros(shape=(len(Intensity1.index)+1, 3*len(Intensity1.columns)+1)))

            for u, v in enumerate(Intensity1.columns):
                err_itensity.ix[1:, u+1] = Intensity1.ix[:, v].values
            for d, g in enumerate(error_intensity1_abs):
                err_itensity.ix[1:, d+4] = error_intensity1_abs.ix[:, g].values
            for h, i in enumerate(error_intensity1_rel):
                err_itensity.ix[1:, h+7] = error_intensity1_rel.ix[:, i].values
            t_index2 = []
            t_index2.extend(['Phi1 \ Phi2 - rel'])
            t_index2.extend(Intensity1.index)
            err_itensity.ix[:, 0] = t_index2
            t_col2 = []
            t_col2.extend(['I_ratrio - abs -  rel'])
            t_col2.extend(Intensity1.columns)
            t_col2.extend(Intensity1.columns)
            t_col2.extend(Intensity1.columns)
            err_itensity.ix[0, :] = t_col2

            output_save = pd.concat([parameter2, err_lifetime, err_itensity], ignore_index=True)

    return parameter


# ---------------------------------------------------------------------------------------
# Extraction o information

def frequency(file, f1):
    parameters = pd.read_csv(file, sep='\t', index_col=0, header=None, skipfooter=8, usecols=[0,1],
                             encoding='latin-1', engine='python')
    f2 = parameters.loc['f2', 1] # Hz
    frequencies = (f1, f2)
    return frequencies


def phi_absolute_values(file):
    parameters = pd.read_csv(file, sep='\t', index_col=0, header=None, skipfooter=8, usecols=[0,1],
                         encoding='latin-1', engine='python')

    phi_f1 = parameters.loc['phi(f1)'].values[0]
    phi_f2 = parameters.loc['phi(f2)'].values[0]
    phi_absolute = (phi_f1, phi_f2)
    return phi_absolute


def tau_absolute_values(file):
    tau_absolute = pd.read_csv(file, sep='\t', index_col=0, skiprows=9, skipfooter=4,
                               usecols=[0, 1, 2, 3], encoding='latin-1', engine='python')
    tau_mean = tau_absolute.loc[tau_absolute.index[1], tau_absolute.columns[1]]
    tau_min = tau_absolute.min().min()
    tau_max = tau_absolute.max().max()
    tau_values = (tau_mean, tau_min, tau_max)
    return tau_values


def tau_error_extraction(file):
    tau_error_rel = pd.read_csv(file, sep='\t', index_col=0, skiprows=9, skipfooter=4,
                                usecols=[0,4,5,6], encoding='latin-1', engine='python')
    tau_er_max = tau_error_rel.max().max()
    tau_er_min = tau_error_rel.min().min()
    tau_error = (tau_er_min, tau_er_max)
    return tau_error


def f1_folders(file, tau=None, lifetime=False, intensity=False, zoom=2, zoom_er=3):
    f1 = int(file.split('\\')[-1].split('H')[0])
    if lifetime is True:
        df = visualisation_frequencies(file, f1, zoom=zoom, zoom_er=zoom_er, plot=False, save=True, fontsize_=13)
    else:
        pass
    if intensity is True:
        if tau is None:
            raise ValueError('Enter the lifetime of the mixed system!')
        else:
            df = visualisation_intensity(file, f1, tau=tau, plot=False, zoom_rel=zoom, zoom_abs=zoom_er, save=True,
                                         fontsize_=13)
    else:
        pass
    return f1


def intensity_values_(file):
    intensity_values = pd.read_csv(file, sep='\t', index_col=0, skiprows=13, usecols=[0, 1, 2, 3],
                                   encoding='latin-1')
    intensity_error_rel = pd.read_csv(file, sep='\t', index_col=0, skiprows=13, usecols=[0, 4, 5, 6],
                                      encoding='latin-1')
    intensity_error_abs = pd.read_csv(file, sep='\t', index_col=0, skiprows=13, usecols=[0, 7, 8, 9],
                                      encoding='latin-1')

    i_min = intensity_values.min().min()
    i_max = intensity_values.max().max()
    i_mean = intensity_values.loc[intensity_values.index[1], intensity_values.columns[1]]

    i_err_abs_max = intensity_error_abs.max().max()
    i_err_abs_min = intensity_error_abs.min().min()

    i_err_rel_max = intensity_error_rel.max().max()
    i_err_rel_min = intensity_error_rel.min().min()
    intensity = (i_mean, i_min, i_max, i_err_abs_max, i_err_abs_min, i_err_rel_max, i_err_rel_min)

    return intensity


# --------------------------------------------------------
# GUI visualizations
def GUI_visualisation(tau_p, f1_, f2_, i_ratio, er_, f_combination):
    # superimposed phase angle
    # I-ratio1 + tau_p1
    [dphi_mixed_deg_f1, dphi_mixed_deg_f2] = phaseangle_DLR_GUI(tau_p=tau_p, f1=f1_, f2=f2_, I_ratio=i_ratio)

    # superimposed phase angle in degree
    phi1 = dphi_mixed_deg_f1
    phi2 = dphi_mixed_deg_f2

    # ---------------------------------------------------------------------------------------------------
    # Reference point
    # Lifetimes
    [x1, x2] = lifetime(phi1=phi1, phi2=phi2, f1=f1_, f2=f2_)

    # Intensity ratio
    y1 = intensity_ratio_(tau=x1, phi=phi1, f=f1_)
    y3 = intensity_ratio_(tau=x2, phi=phi1, f=f1_)

    if x1 < 0 and x2 > 0:
        select = 2
        ref_tau = x2
        ref_y = y3
    elif x2 < 0 and x1 > 0:
        select = 1
        ref_tau = x1
        ref_y = y1
    elif x1 > 0 and x2 > 0:
        if abs(x1 - tau_p) < abs(x2 - tau_p):
            select = 1
            ref_tau = x1
            ref_y = y1
        else:
            select = 2
            ref_tau = x2
            ref_y = y3
    else:
        select = 3
        ref_tau = np.nan
        ref_y = np.nan

    # ------------------------------------------------------------------------------------------------
    # error range for lifetime and intensity ratio
    [Res1, Intensity1, p1, p2, Res1_mikros] = spread_imperfection_GUI(phi1, phi2, er=er_, f1=f1_, f2=f2_, select=select)

    # ----------------------------------------------------------------------------------------------
    # error calculation
    if select == 3:
        pass
    else:
        [e_tau1_rel, e_intensity1_abs, e_intensity1_rel] = error_GUI(Res1, ref_tau, Intensity1, ref_y)

        if type(Res1_mikros) == float:
            pass
        else:
            # life time and deviation in µs
            tau_abs_mean = Res1_mikros.loc[Res1_mikros.index[1], Res1_mikros.columns[1]]
            tau_abs_min = Res1_mikros.min().min()
            tau_abs_max = Res1_mikros.max().max()
            # relative error of lifetime compared to initial lifetime in %
            tau_max_rel = e_tau1_rel.max().max()
            tau_min_rel = e_tau1_rel.min().min()
            # intensity and its deviation and (absolute and relative) errors
            i_max = Intensity1.max().max()
            i_min = Intensity1.min().min()
            i_abs_max = e_intensity1_abs.max().max()
            i_abs_min = e_intensity1_abs.min().min()
            i_rel_max = e_intensity1_rel.max().max()
            i_rel_min = e_intensity1_rel.min().min()

            f_combination.append((f1_, f2_, p1, p2, tau_abs_mean, tau_abs_min, tau_abs_max, tau_min_rel,
                                  tau_max_rel, i_ratio, i_min, i_max, i_abs_min, i_abs_max, i_rel_min,
                                  i_rel_max))

    return f_combination


# --------------------------------------------------------
# Visualization
def visualisation_frequencies(file, f1, plot=False, zoom=1, zoom_er=5, save=True, fontsize_=13):

    # extracting frequencies, phase angle, life time and its relative error
    freq = [frequency(i, f1) for i in glob(file + '/*_*.txt')]
    frequencies = pd.DataFrame(freq, columns=['f1 [kHz]', 'f2 [kHz]'])

    p = [phi_absolute_values(i) for i in glob(file + '/*_*.txt')]
    phi = pd.DataFrame(p, columns=['phi(f1) [deg]', 'phi(f2) [deg]'])

    p_tau_er = [tau_error_extraction(i) for i in glob(file + '/*_*.txt')]
    tau_er_ = pd.DataFrame(p_tau_er, columns=['tau_error min', 'tau_error max'])

    p_tau = [tau_absolute_values(i) for i in glob(file + '/*_*.txt')]
    tau_abs_ = pd.DataFrame(p_tau, columns=['tau', 'tau min', 'tau max'])

    phi_error_relative_ = pd.concat([frequencies, phi, tau_abs_, tau_er_], axis=1).sort_values(by='f2 [kHz]')
    phi_error_relative = phi_error_relative_.reset_index(drop=True)

    # plotting preparation

    fig1 = plt.figure(figsize=(12, 8))
    gs = gspec.GridSpec(3, 2)
    gs.update(hspace=0.3, wspace=0.3)

    ax = fig1.add_subplot(gs[0, :])
    ax_tau = fig1.add_subplot(gs[1, :1])
    ax_err = plt.subplot(gs[1, 1])

    ax_err_zoom = fig1.add_subplot(gs[2, 1])
    ax_tau_zoom= fig1.add_subplot(gs[2, :-1])

    # absolute phase angles [deg]
    ax.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'phi(f2) [deg]'], color='#20948b',
            label='Φ(f2) [°]')
    ax.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'phi(f1) [deg]'], color='#de7a22',
            label='Φ(f1) [°]')
    ax.legend(loc=0, fontsize=fontsize_)

    # life time tau and its absolute deviation
    ax_tau.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau'], color='#f9ba32',
                label='{} µs'.format(round(phi_error_relative.loc[0, 'tau'], 2)))
    ax_tau.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau min'], color='#2f3131',
                label='τ$_{min}$ [µs]')
    ax_tau.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau max'], color='#426e86',
                label='τ$_{max}$ [µs]')
    ax_tau.legend(loc=0, fontsize=fontsize_)

    # relative error for life time
    ax_err.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau_error min'], color='#2f3131',
                label='ξ$_{min}$ [%]')
    ax_err.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau_error max'], color='#426e86',
                label='ξ$_{max}$ [%]')
    ax_err.legend(loc=0, fontsize=fontsize_)

    # Zoom in plots
    ax_tau_zoom.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau'], color='#f9ba32')
    ax_tau_zoom.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau min'], color='#2f3131')
    ax_tau_zoom.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau max'], color='#426e86')

    ax_tau_zoom.set_ylim(phi_error_relative.loc[0, 'tau']-zoom, phi_error_relative.loc[0, 'tau'] + zoom)
    ax_tau_zoom.set_title('Zoom in plot', fontsize=fontsize_*0.7)
    ax_tau_zoom.tick_params(labelsize=fontsize_*0.7)

    ax_err_zoom.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau_error min'], color='#2f3131')
    ax_err_zoom.plot(phi_error_relative.loc[:, 'f2 [kHz]'], phi_error_relative.loc[:, 'tau_error max'], color='#426e86')
    ax_err_zoom.set_ylim(-zoom_er, zoom_er)
    ax_err_zoom.set_title('Zoom in plot', fontsize=fontsize_*0.7)
    ax_err_zoom.tick_params(labelsize=fontsize_*0.7)

    ax.set_ylabel('Phase angle [deg]', fontsize=fontsize_)

    ax_tau.set_ylabel('Deviation tau [µs]', fontsize=fontsize_)
    ax_tau_zoom.set_ylabel('Deviation tau [µs]', fontsize=fontsize_)
    ax_tau_zoom.set_xlabel('Modulation frequency f$_2$ [kHz]', fontsize=fontsize_)

    ax_err_zoom.set_xlabel('Modulation frequency f$_2$ [kHz]', fontsize=fontsize_)
    ax_err.set_ylabel('Relative error tau [%]', fontsize=fontsize_)
    ax_err_zoom.set_ylabel('Relative error tau [%]', fontsize=fontsize_)

    if type(f1) is float:
        f1 = int(f1)*1000
    elif type(f1) is str:
        f1 = f1
    else:
        f1 = str(f1 / 1000) + 'k'
    ax.set_title('Error propagation for τ depending on the pair of modulation frequencies '
                 '(f$_1$ = {} Hz; f$_2$)'.format(f1), fontsize=fontsize_+1)
    plt.tight_layout(h_pad=10)
    if plot is False:
        plt.close(fig1)
    else:
        pass

    if save is True:
        directory = 'D:/01_data_processing/dualsensor/errorpropagation_frequency-pair/figures_tau/'.format(f1)
        if not os.path.exists(directory):
                os.makedirs(directory)
        savename = directory + '{}_error-propagation_lifetime_f1-{}Hz.png'.format(today, f1)
        fig1.savefig(savename, dpi=300)
    else:
        pass

    return phi_error_relative


def visualisation_intensity(file, f1, tau, plot=False, zoom_rel=1, zoom_abs=0.02, save=True, fontsize_=13):

    # extracting frequencies, phase angle, life time and its relative error
    freq = [frequency(i, f1) for i in glob(file + '/*_*.txt')]
    frequencies = pd.DataFrame(freq, columns=['f1 [kHz]', 'f2 [kHz]'])

    p = [phi_absolute_values(i) for i in glob(file + '/*_*.txt')]
    phi = pd.DataFrame(p, columns=['phi(f1) [deg]', 'phi(f2) [deg]'])

    p_i_er = [intensity_values_(i) for i in glob(file + '/*_*.txt')]
    iratio = pd.DataFrame(p_i_er, columns=['I ratio', 'I ratio min', 'I ratio max', 'I rel max', 'I rel min',
                                           'I err abs max', 'I err abs min'])

    i_error_relative_ = pd.concat([frequencies, phi, iratio], axis=1).sort_values(by='f2 [kHz]')
    i_error_relative = i_error_relative_.reset_index(drop=True)

    # plotting preparation
    fig = plt.figure(figsize=(12, 8))
    gs = gspec.GridSpec(3, 2)
    gs.update(hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(gs[0, :-1])

    ax_intensity_error = fig.add_subplot(gs[1, :-1])
    ax_intensity_err_zoom = fig.add_subplot(gs[2, :-1])

    ax_intensity = fig.add_subplot(gs[0, 1])
    ax_intensity_dev = fig.add_subplot(gs[1, 1])
    ax_intensity_abs_zoom = fig.add_subplot(gs[2, 1])

    # absolute phase angles [deg]
    ax.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'phi(f2) [deg]'], color='#20948b',
            label='Φ(f2) [°]')
    ax.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'phi(f1) [deg]'], color='#de7a22',
            label='Φ(f1) [°]')
    ax.legend(loc=0, fontsize=fontsize_)

    # intensity ratio
    ax_intensity.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I ratio'], color='#f9ba32',
                      label='mean')
    ax_intensity.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I ratio min'],
                      color='#2f3131', label='min')
    ax_intensity.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I ratio max'],
                      color='#426e86', label='max')
    ax_intensity.legend(loc=0, fontsize=fontsize_)

    # absolute deviation of intensity ratio
    ax_intensity_dev.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I err abs min'],
                          color='#2f3131', label='min')
    ax_intensity_dev.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I err abs max'],
                          color='#426e86', label='max')
    ax_intensity_dev.legend(loc=0, fontsize=fontsize_)

    # relative error for intensity ratio
    ax_intensity_error.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I rel min'],
                            color='#2f3131')
    ax_intensity_error.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I rel max'],
                            color='#426e86')
    ax_intensity_error.legend(loc=0, fontsize=fontsize_)

    # Zoom into absolute deviation of intensity ratio
    ax_intensity_abs_zoom.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I err abs min'],
                               color='#2f3131')
    ax_intensity_abs_zoom.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I err abs max'],
                               color='#426e86', label='max')
    ax_intensity_abs_zoom.set_ylim(-zoom_abs, zoom_abs)

    # Zoom into relative error rate
    ax_intensity_err_zoom.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I rel min'],
                               color='#2f3131', label='min')
    ax_intensity_err_zoom.plot(i_error_relative.loc[:, 'f2 [kHz]'], i_error_relative.loc[:, 'I rel max'],
                               color='#426e86', label='max')
    ax_intensity_err_zoom.set_ylim(-zoom_rel, zoom_rel)

    # labelling axes
    ax.set_ylabel('Phase angle [deg]', fontsize=fontsize_)
    ax_intensity.set_ylabel('Intensity ratio', fontsize=fontsize_)
    ax_intensity_dev.set_ylabel('Abs. deviation I-ratio', fontsize=fontsize_)
    ax_intensity_abs_zoom.set_ylabel('Abs. deviation I-ratio', fontsize=fontsize_)
    ax_intensity_error.set_ylabel('Rel. error I-ratio [%]', fontsize=fontsize_)
    ax_intensity_err_zoom.set_ylabel('Rel. error I-ratio [%]', fontsize=fontsize_)

    ax_intensity_err_zoom.set_xlabel('Modulation frequency f$_2$ [kHz]', fontsize=fontsize_)
    ax_intensity_abs_zoom.set_xlabel('Modulation frequency f$_2$ [kHz]', fontsize=fontsize_)

    ax_intensity_err_zoom.set_title('Zoom in plot', fontsize=fontsize_*0.7)
    ax_intensity_abs_zoom.set_title('Zoom in plot', fontsize=fontsize_*0.7)

    if type(f1) is float:
        f1 = int(f1)*1000
    elif type(f1) is str:
        f1 = f1
    else:
        f1 = str(f1 / 1000) + 'k'
    ax.set_title('Error propagation for I-ratio depending on the pair of modulation frequencies '
                 '(f$_1$ = {} Hz; f$_2$) at {}µs'.format(f1, tau), fontsize=fontsize_+1, y=1.08, x=1.15)
    plt.tight_layout(h_pad=10)

    if plot is False:
        plt.close(fig)
    else:
        pass

    if save is True:
        directory = 'D:/01_data_processing/dualsensor/errorpropagation_frequency-pair/figures_intensity/'.format(f1)
        if not os.path.exists(directory):
                os.makedirs(directory)
        savename = directory + '{}_error-propagation_intensity-ratio_f1-{}Hz_{}µs.png'.format(today, f1, tau)
        fig.savefig(savename, dpi=300)
    else:
        pass

    return i_error_relative


# -----------------------------------------------------------------------------------------------------------------
def report(t, I, signals, lifetime_phosphor, intensity_ratio, error_assumed):
    f1 = signals.loc[:, 'er(tau, min) [%]'].idxmax()
    f2 = signals.loc[:, 'er(tau, max) [%]'].idxmin()
    f1_ = signals.loc[:, 'er_abs(i, min) [%]'].idxmax()
    f2_ = signals.loc[:, 'er_abs(i, max) [%]'].idxmin()

    if signals.loc[f1, 'f2 [Hz]'] <= signals.loc[f2, 'f2 [Hz]']:
        if (signals.loc[f1, 'er_abs(i, min) [%]']) <= (signals.loc[f1_, 'er_abs(i, min) [%]']):
            f = f1
        else:
            f = f1_
    else:
        if (signals.loc[f2, 'er_abs(i, min) [%]']) <= (signals.loc[f2_, 'er_abs(i, min) [%]']):
            f = f2
        else:
            f = f2_

    rep = pd.DataFrame(np.zeros(shape=(20, 3)))
    rep.columns = ['', 'scalar', 'unit']

    rep.iloc[0, :] = ['INPUT parameter', '', '']
    rep.iloc[1, :] = ['f1: ', signals.loc[f, 'f1 [Hz]'], 'Hz']
    rep.iloc[2, :] = ['tau{}: '.format(t), lifetime_phosphor, 'µs']
    rep.iloc[3, :] = ['I-ratio {}: '.format(I), intensity_ratio, '']
    rep.iloc[4, :] = ['assumed uncertainty: ', error_assumed, '°']
    rep.iloc[5, :] = [' ------------------', ' ---', ' ---']
    rep.iloc[6, :] = ['OUTPUT parameter', '', '']
    rep.iloc[7, :] = ['f2: ', signals.loc[f, 'f2 [Hz]'].round(0), 'Hz']
    rep.iloc[8, :] = ['phi(f1): ', signals.loc[f, 'phi(f1) [°]'].round(2), '°']
    rep.iloc[9, :] = ['phi(f2): ', signals.loc[f, 'phi(f2) [°]'].round(2), '°']
    rep.iloc[10, :] = ['tau{}(min): '.format(t), signals.loc[f, 'tau(min) [µs]'].round(3), 'µs']
    rep.iloc[11, :] = ['tau{}(max): '.format(t), signals.loc[f, 'tau(max) [µs]'].round(3), 'µs']
    rep.iloc[12, :] = ['er(tau{}, min): '.format(t), signals.loc[f, 'er(tau, min) [%]'].round(3), '%']
    rep.iloc[13, :] = ['er(tau{}, max): '.format(t), signals.loc[f, 'er(tau, max) [%]'].round(3), '%']
    rep.iloc[14, :] = ['I-ratio(min): ', signals.loc[f, 'i_ratio(min)'].round(2), '']
    rep.iloc[15, :] = ['I-ratio(max): ', signals.loc[f, 'i_ratio(max)'].round(2), '']
    rep.iloc[16, :] = ['dev(I{}, min): '.format(I), '{:.2e}'.format(signals.loc[f, 'er_abs(i, min) [%]']), '']
    rep.iloc[17, :] = ['dev(I{}, max): '.format(I), '{:.2e}'.format(signals.loc[f, 'er_abs(i, max) [%]']), '']
    rep.iloc[18, :] = ['er(I{}, min): '.format(I), '{:.2e}'.format(signals.loc[f, 'er_rel(i, min) [%]']), '%']
    rep.iloc[19, :] = ['er(I{}, max): '.format(I), '{:.2e}'.format(signals.loc[f, 'er_rel(i, max) [%]']), '%']

    return rep
