__author__ = 'szieger'
__project__ = 'dualsensor ph/O2 sensing'

import matplotlib
import additional_functions as af
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import stats
from scipy.optimize import curve_fit

conv_temp = 273.15


# --------------------------------------------------------------------------------------------------------------------
# Temperature compensation pH
# --------------------------------------------------------------------------------------------------------------------
def datatable_to_channel(df):
    pH_calib = df['pH'].values
    temp_calib = df['Temperature'].values
    temp_calib_ = temp_calib[~pd.isnull(temp_calib)]
    print('T = {:.2f} ± {:.2f}°C'.format(temp_calib_.mean(), temp_calib_.std()))

    for i in range(len(df.columns)-2):
        if df.columns[i].split('_')[1][0] == '1':
            ch1 = i
        elif df.columns[i].split('_')[1][0] == '2':
            ch2 = i
        elif df.columns[i].split('_')[1][0] == '3':
            ch3 = i
        elif df.columns[i].split('_')[1][0] == '4':
            ch4 = i
    print('total amount of columns:', len(df.columns))
    channel1 = df.loc[:, df.columns[0]:df.columns[ch1]]
    channel2 = df.loc[:, df.columns[ch1+1]:df.columns[ch2]]
    channel3 = df.loc[:, df.columns[ch2+1]:df.columns[ch3]]
    channel4 = df.loc[:, df.columns[ch3+1]:df.columns[ch4]]

    for i in range(len(channel1.columns)):
        if channel1.columns[i].split('_')[1][1:] == 'dPhi':
            dphi = i
    ch1_dphi = channel1.loc[:, channel1.columns[0]: channel1.columns[dphi]]
    ch1_intensity = channel1.loc[:, channel1.columns[dphi+1]: channel1.columns[-1]]

    ch2_dphi = channel2.loc[:, channel2.columns[0]: channel2.columns[dphi]]
    ch2_intensity = channel2.loc[:, channel2.columns[dphi+1]: channel2.columns[-1]]

    ch3_dphi = channel3.loc[:, channel3.columns[0]: channel3.columns[dphi]]
    ch3_intensity = channel3.loc[:, channel3.columns[dphi+1]: channel3.columns[-1]]

    ch4_dphi = channel4.loc[:, channel4.columns[0]: channel4.columns[dphi]]
    ch4_intensity = channel4.loc[:, channel4.columns[dphi+1]: channel4.columns[-1]]

    data = pd.Series({'ch1_dPhi': ch1_dphi, 'ch1_Intensity': ch1_intensity,
                      'ch2_dPhi': ch2_dphi, 'ch2_Intensity': ch2_intensity,
                      'ch3_dPhi': ch3_dphi, 'ch3_Intensity': ch3_intensity,
                      'ch4_dPhi': ch4_dphi, 'ch4_Intensity': ch4_intensity,
                      'Temperature': temp_calib, 'pH': pH_calib})
    return data


def func_boltzmann_sigmoid(pH, top, bottom, v50, slope):
    cot = bottom + (top - bottom) / (1 + np.exp((pH - v50)/slope))
    return cot


def plotting_fit_4channels(xdata, ph_scan, df, fit_df, fontsize_=13):
    fig, ax_fit = plt.subplots(ncols=2, nrows=2, sharex=True)
    ax_fit_twin00 = ax_fit[0][0].twinx()
    ax_fit_twin01 = ax_fit[0][1].twinx()
    ax_fit_twin10 = ax_fit[1][0].twinx()
    ax_fit_twin11 = ax_fit[1][1].twinx()
    ax_row = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ax_row2 = [ax_fit_twin00, ax_fit_twin01, ax_fit_twin10, ax_fit_twin11]

    # --------------------------------------------------------------------------------------------
    # data points for fitting
    freq = []
    for i in range(len(df.columns)):
        freq_ = df.columns[i].split('_')[-1]
        freq.append(freq_)
        n = ax_row[i][0]
        r = ax_row[i][1]
        d2 = ax_row2[i].plot(xdata, df[df.columns[i]],marker='d', color='#bcbabe', lw=0, label='data')
        ax_fit[n][r].set_title(freq_)

    # --------------------------------------------------------------------------------------------
    # fit
    for i in range(len(df.columns)):
        n = ax_row[i][0]
        r = ax_row[i][1]
        d3 = ax_fit[n][r].plot(ph_scan, fit_df['fit_data cotPhi'][fit_df['fit_data cotPhi'].columns[i]],
                               color='navy', lw=0.75, label='fit_cot(dPhi)')
        d4 = ax_row2[i].plot(ph_scan, fit_df['fit_data dPhi'][fit_df['fit_data cotPhi'].columns[i]],
                             color='forestgreen', lw=0.75, label='fit_dPhi')

    # --------------------------------------------------------------------------------------------
    lns = d2 + d3 + d4
    labs = [l.get_label() for l in lns]
    ax_fit[0][0].legend(lns, labs, loc='upper center', bbox_to_anchor=(1.3, 1.5), ncol=4)

    ax_fit[1][0].set_xlabel('pH', fontsize=fontsize_)
    ax_fit[1][1].set_xlabel('pH', fontsize=fontsize_)
    ax_fit[0][0].set_ylabel('cot(dPhi)', color='navy', fontsize=fontsize_)
    ax_fit[1][0].set_ylabel('cot(dPhi)', color='navy', fontsize=fontsize_)
    ax_fit_twin01.set_ylabel('dPhi [deg]', color='forestgreen', fontsize=fontsize_)
    ax_fit_twin11.set_ylabel('dPhi [deg]', color='forestgreen', fontsize=fontsize_)

    plt.tight_layout()


def boltzmann_multiple_frequencies(ph_scan, xdata, data_raw):
    # preparation of dataframe
    freq = []
    cotPhi_temp = pd.DataFrame(np.zeros(shape=(len(data_raw.index), len(data_raw.columns))))
    dPhi_temp = pd.DataFrame(np.zeros(shape=(len(data_raw.index), len(data_raw.columns))))

    for i in range(len(data_raw.columns)):
        freq_ = data_raw.columns[i].split('_')[-1]
        freq.append(freq_)
        cotPhi_temp.iloc[:, i] = af.cot(np.deg2rad(data_raw[data_raw.columns[i]])).values
        dPhi_temp.iloc[:, i] = data_raw[data_raw.columns[i]].values

    cotPhi_temp.columns = freq
    dPhi_temp.columns = freq

    # -------------------------------------------------------------------
    # Boltzmann sigmoid fit with scipy.optimize - curve_fit
    fit_para_cotPhi = pd.DataFrame(np.zeros(shape=(len(freq), 4)), index=freq, columns=['Top', 'Bottom', 'V50', 'slope'])
    fit_para_dPhi = pd.DataFrame(np.zeros(shape=(len(freq), 4)), index=freq, columns=['Top', 'Bottom', 'V50', 'slope'])
    data_fit_cotPhi = pd.DataFrame(np.zeros(shape=(len(ph_scan), len(freq))), index=ph_scan, columns=freq)
    data_fit_dPhi = pd.DataFrame(np.zeros(shape=(len(ph_scan), len(freq))), index=ph_scan, columns=freq)

    # -----------------------------
    # fit for all frequencies
    for i in freq:
        popt_cotPhi, pcov_cotPhi = curve_fit(func_boltzmann_sigmoid, xdata=xdata, ydata=cotPhi_temp[i])
        popt_dPhi, pcov_dPhi = curve_fit(func_boltzmann_sigmoid, xdata=xdata, ydata=dPhi_temp[i])

        fit_para_cotPhi.loc[i] = popt_cotPhi
        fit_para_dPhi.loc[i] = popt_dPhi

        data_fit_cotPhi[i] = func_boltzmann_sigmoid(ph_scan, *popt_cotPhi)
        data_fit_dPhi[i] = func_boltzmann_sigmoid(ph_scan, *popt_dPhi)

    # -------------------------------------------------------------------
    # preparation for output
    fitting = pd.Series({'fit_parameter cotPhi': fit_para_cotPhi, 'fit_parameter dPhi': fit_para_dPhi,
                         'fit_data cotPhi': data_fit_cotPhi, 'fit_data dPhi': data_fit_dPhi})

    return fitting


def temp_compensation_pH_regression(boltzmann_temp, temp_discret, temp_scan):
    # linea regression with stats.linregress
    # Top
    [slope_top, intercept_top, r_value_top,
     p_value_top, std_top] = stats.linregress(x=temp_discret, y=boltzmann_temp.loc['Top'].values)
    y_top = temp_scan*slope_top + intercept_top

    # Bottom
    [slope_bottom, intercept_bottom, r_value_bottom,
     p_value_bottom, std_bottom] = stats.linregress(x=temp_discret, y=boltzmann_temp.loc['Bottom'].values)
    y_bottom = temp_scan*slope_bottom + intercept_bottom

    # V50
    [slope_V50, intercept_V50, r_value_V50,
     p_value_V50, std_V50] = stats.linregress(x=temp_discret, y=boltzmann_temp.loc['V50'].values)
    y_V50 = temp_scan*slope_V50 + intercept_V50

    # slope
    [slope_slope, intercept_slope, r_value_slope,
     p_value_slope, std_slope] = stats.linregress(x=temp_discret, y=boltzmann_temp.loc['slope'].values)
    y_slope = temp_scan*slope_slope + intercept_slope

    r_value = pd.Series({'Top': r_value_top, 'Bottom': r_value_bottom, 'V50': r_value_V50, 'slope': r_value_slope})
    slope_ = pd.Series({'Top': slope_top, 'Bottom': slope_bottom, 'V50': slope_V50, 'slope': slope_slope})
    intercept_ = pd.Series({'Top': intercept_top, 'Bottom': intercept_bottom, 'V50': intercept_V50,
                            'slope': intercept_slope})

    fit_parameter = pd.Series({'Temperature scan': temp_scan, 'Top': y_top, 'Bottom': y_bottom,
                               'V50': y_V50, 'slope': y_slope, 'R values': r_value, 'slopes': slope_,
                               'intercepts': intercept_})

    return fit_parameter


def temp_compensation_pH(temp_discret, temp_scan, df_fit_temp1, df_fit_temp2, df_fit_temp3, fit='dPhi',
                         plotting=True, temp_meas=None, lw_=0.75, fontsize_=13, plot_freq='1000Hz'):
    if fit == 'dPhi':
        label = 'fit_parameter dPhi'
    elif fit == 'cotPhi':
        label = 'fit_parameter cotPhi'
    else:
        raise ValueError('Select way of fitting - either choose dPhi or cotPhi...')

    fit_para_cotPhi = []
    boltzmann_discret = []
    col_name = []
    for i in temp_discret:
        col_name.append(str(i) + ' °C')

    for i in df_fit_temp1[label].index:
        boltzmann_temp = pd.concat([df_fit_temp1[label].loc[i], df_fit_temp2[label].loc[i],
                                    df_fit_temp3[label].loc[i]], axis=1)
        boltzmann_temp.columns = col_name

        fit_cotPhi = temp_compensation_pH_regression(boltzmann_temp=boltzmann_temp,
                                                     temp_discret=temp_discret, temp_scan=temp_scan)
        fit_para_cotPhi.append(fit_cotPhi)
        boltzmann_discret.append(boltzmann_temp)
    fit_cotPhi_pH_temp_freq = pd.Series(fit_para_cotPhi, index=df_fit_temp1[label].index)
    data_cotPhi_pH_temp_freq = pd.Series(boltzmann_discret, index=df_fit_temp1[label].index)

    # ----------------------------------------------------------------------------------------------
    if temp_meas is None:
        fit_temp_meas = None
    else:
        fit_temp_meas = pd.DataFrame(np.zeros(shape=(len(df_fit_temp1[label].index),
                                                     len(fit_cotPhi_pH_temp_freq['1000Hz']['slopes'].index))),
                                     index=df_fit_temp1[label].index,
                                     columns=fit_cotPhi_pH_temp_freq['1000Hz']['slopes'].index)
        for i in fit_temp_meas.index:
            fit_temp_meas.loc[i, :] = fit_cotPhi_pH_temp_freq[i]['slopes']*temp_meas + \
                                      fit_cotPhi_pH_temp_freq[i]['intercepts']
    # ----------------------------------------------------------------------------------------------
    if plotting is True:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)

        df = fit_cotPhi_pH_temp_freq[plot_freq]
        df_discret = data_cotPhi_pH_temp_freq[plot_freq]
        ax[0][0].plot(df['Temperature scan'], df['slope'], color='k', lw=lw_)
        ax[0][0].plot(temp_discret, df_discret.loc['slope'], color='#68829e', lw=0, marker='o')

        ax[0][1].plot(df['Temperature scan'], df['Bottom'], color='k', lw=lw_)
        ax[0][1].plot(temp_discret, df_discret.loc['Bottom'], color='#07575b', lw=0, marker='o')

        ax[1][0].plot(df['Temperature scan'], df['V50'], color='k', lw=lw_)
        ax[1][0].plot(temp_discret, df_discret.loc['V50'], color='#004445', lw=0, marker='o')

        ax[1][1].plot(df['Temperature scan'], df['Top'], color='k', lw=lw_)
        ax[1][1].plot(temp_discret, df_discret.loc['Top'], color='#375e97', lw=0, marker='o')

        ax[0][0].legend(['R = {:.2f}'.format(fit_cotPhi_pH_temp_freq[plot_freq]['R values']['slope'])])
        ax[0][1].legend(['R = {:.2f}'.format(fit_cotPhi_pH_temp_freq[plot_freq]['R values']['Bottom'])])
        ax[1][0].legend(['R = {:.2f}'.format(fit_cotPhi_pH_temp_freq[plot_freq]['R values']['V50'])])
        ax[1][1].legend(['R = {:.2f}'.format(fit_cotPhi_pH_temp_freq[plot_freq]['R values']['Top'])])

        # if Temperature is measured
        if temp_meas is None:
            pass
        else:
            ax[0][0].axhline(fit_temp_meas['slope'].loc[plot_freq], color='crimson', lw=0.75, ls='--')
            ax[0][0].axvline(temp_meas, color='crimson', lw=0.75, ls='--')

            ax[0][1].axhline(fit_temp_meas['Bottom'].loc[plot_freq], color='crimson', lw=0.75, ls='--')
            ax[0][1].axvline(temp_meas, color='crimson', lw=0.75, ls='--')

            ax[1][0].axhline(fit_temp_meas['V50'].loc[plot_freq], color='crimson', lw=0.75, ls='--')
            ax[1][0].axvline(temp_meas, color='crimson', lw=0.75, ls='--')

            ax[1][1].axhline(fit_temp_meas['Top'].loc[plot_freq], color='crimson', lw=0.75, ls='--')
            ax[1][1].axvline(temp_meas, color='crimson', lw=0.75, ls='--')

        ax[1][0].set_xlabel('pH', fontsize=fontsize_)
        ax[1][1].set_xlabel('pH', fontsize=fontsize_)

        ax[0][0].set_ylabel('slope', color='#68829e', fontsize=fontsize_)
        ax[0][1].set_ylabel('bottom', color='#07575b', fontsize=fontsize_)
        ax[1][0].set_ylabel('V50', color='#004445', fontsize=fontsize_)
        ax[1][1].set_ylabel('top', color='#375e97', fontsize=fontsize_)
        plt.xlim(0,50)

        plt.tight_layout()

    return fit_cotPhi_pH_temp_freq, data_cotPhi_pH_temp_freq, fit_temp_meas


# --------------------------------------------------------------------------------------------------------------------
# INPUT parameter / Simulation
# --------------------------------------------------------------------------------------------------------------------
# slope/intercept for T sensing

# pka, top and bottom calibration pH sensing (T compensation)
def temperature_compensation(file, T_meas, temp_range, f1, f2, fit_k=True, fit_bottom=True, fit_top=True, fit_pka=True,
                             plotting=True):
    """

    :param file:
    :param T_meas:
    :param temp_range:      in °C
    :param f1:              Modulation frequency
    :param f2:              Modulation frequency
    :param fit_k:
    :param fit_bottom:
    :param fit_top:
    :param fit_pka:
    :param plotting:
    :return:
    """

    # sensor feature
    ddf = pd.read_csv(file, encoding='latin-1', sep='\t', usecols=[1, 2, 3, 4, 5])

    # fitting along temperature range
    x = ddf['T [°C]'].values
    slope_k, intercept_k, r_value_k, p_value_k, std_err_k = stats.linregress(x=x, y=ddf['slope'])
    slope_pka, intercept_pka, r_value_pka, p_value_pka, std_err_pka = stats.linregress(x=x, y=ddf['V50'])
    slope_top, intercept_top, r_value_top, p_value_top, std_err_top = stats.linregress(x=x, y=ddf['Top'])
    slope_bottom, intercept_bottom, r_value_bottom, p_value_bottom, std_err_bottom = stats.linregress(x=x,
                                                                                                      y=ddf['Bottom'])

    parameter_fit = pd.Series({'slope, slope': slope_k, 'top, slope': slope_top, 'bottom, slope': slope_bottom,
                               'pka, slope': slope_pka, 'slope, intercept': intercept_k,
                               'top, intercept': intercept_top, 'bottom, intercept': intercept_bottom,
                               'pka, intercept': intercept_pka})

    # determination of sensor characteristics at measurement temperature
    if np.isnan(slope_k) is True or fit_k is False:
        slope_k = ddf['slope'].mean()
        fit_k = False
        k = slope_k
    else:
        k = slope_k*T_meas + intercept_k
    if np.isnan(slope_pka) is True or fit_pka is False:
        slope_pka = ddf['V50'].mean()
        fit_pka = False
        pka = slope_pka
    else:
        pka = slope_pka*T_meas + intercept_pka
    if np.isnan(slope_top) is True or fit_top is False:
        slope_top = ddf['Top'].mean()
        fit_top = False
        top = slope_top
    else:
        top = slope_top*T_meas + intercept_top
    if np.isnan(slope_bottom) is True or fit_bottom is False:
        slope_bottom = ddf['Bottom'].mean()
        fit_bottom = False
        bottom = slope_bottom
    else:
        bottom = slope_bottom*T_meas + intercept_bottom

    para_meas = pd.Series({'slope': k, 'pka': pka, 'top': top, 'bottom': bottom})

    y_k = slope_k*temp_range + intercept_k
    y_pka = slope_pka*temp_range + intercept_pka
    y_top = slope_top*temp_range + intercept_top
    y_bottom = slope_bottom*temp_range + intercept_bottom

    # ----------------------------------------------------
    # Fitting along frequency

    # -------------------------------------------------------------------------------------------------------------
    if plotting is True:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)

        ax[0][0].plot(ddf['T [°C]'], ddf['slope'], marker='o', markersize=5, lw=0,
                      label='{:.2f} ± {:.2f}'.format(ddf['slope'].mean(), ddf['slope'].std()))
        if fit_k is True:
            ax[0][0].plot(temp_range, y_k, lw=0.75, ls='-', color='k')
            ax[0][0].axhline(k, color='crimson', lw=0.5, ls='--')
            ax[0][0].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[1][0].plot(ddf['T [°C]'], ddf['V50'], marker='o', markersize=5, lw=0,
                      label='{:.2f} ± {:.2f}'.format(ddf['V50'].mean(), ddf['V50'].std()))
        if fit_pka is True:
            ax[1][0].plot(temp_range, y_pka, lw=0.75, ls='-', color='k')
            ax[1][0].axhline(pka, color='crimson', lw=0.5, ls='--')
            ax[1][0].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[0][1].plot(ddf['T [°C]'], ddf['Bottom'], marker='o', markersize=5, lw=0,
                      label='{:.2f} ± {:.2f}'.format(ddf['Bottom'].mean(), ddf['Bottom'].std()))
        if fit_bottom is True:
            ax[0][1].plot(temp_range, y_bottom, lw=0.75, ls='-', color='k')
            ax[0][1].axhline(bottom, color='crimson', lw=0.5, ls='--')
            ax[0][1].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[1][1].plot(ddf['T [°C]'], ddf['Top'], marker='o', markersize=5, lw=0,
                      label='{:.2f} ± {:.2f}'.format(ddf['Top'].mean(), ddf['Top'].std()))
        if fit_top is True:
            ax[1][1].plot(temp_range, y_top, lw=0.75, ls='-', color='k')
            ax[1][1].axhline(top, color='crimson', lw=0.5, ls='--')
            ax[1][1].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[0][0].set_ylabel('slope', fontsize=13)
        ax[1][0].set_ylabel('pka', fontsize=13)
        ax[0][1].set_ylabel('Bottom', fontsize=13)
        ax[1][1].set_ylabel('Top', fontsize=13)

        ax[1][0].set_xlabel('temperature T [°C]', fontsize=13)
        ax[1][1].set_xlabel('temperature T [°C]', fontsize=13)

        ax[0][0].legend()
        ax[1][0].legend()
        ax[0][1].legend()
        ax[1][1].legend()

        plt.tight_layout()

    return ddf, para_meas, parameter_fit


def boltzmann_sigmoid(top, bottom, slope, pka, pH):
    """returns the cot(dPhi) of the mixed system at certain T and frequency """
    return bottom + (top - bottom) / (1 + 10**((pH - pka)/slope))


def phi_from_cot(para_f, T0, T1, ph0, ph1, ph_meas, f, ph_range, plot_T_comp=True):
    """

    :param para_f:
    :param T0:
    :param T1:
    :param ph0:
    :param ph1:
    :param ph_meas:
    :param f:
    :param ph_range:
    :param plot_T_comp:
    :return:
    """

    # cotPhi of the mixed system at frequency f1
    cotPhi_ph0_T0 = boltzmann_sigmoid(top=para_f['phosphor0']['top'], bottom=para_f['phosphor0']['bottom'],
                                      slope=para_f['phosphor0']['slope'], pka=para_f['phosphor0']['pka'], pH=ph0)
    cotPhi_ph1_T0 = boltzmann_sigmoid(top=para_f['phosphor0']['top'], bottom=para_f['phosphor0']['bottom'],
                                      slope=para_f['phosphor0']['slope'], pka=para_f['phosphor0']['pka'], pH=ph1)
    cotPhi_ph0_T1 = boltzmann_sigmoid(top=para_f['phosphor1']['top'], bottom=para_f['phosphor1']['bottom'],
                                      slope=para_f['phosphor1']['slope'], pka=para_f['phosphor1']['pka'], pH=ph0)
    cotPhi_ph1_T1 = boltzmann_sigmoid(top=para_f['phosphor1']['top'], bottom=para_f['phosphor1']['bottom'],
                                      slope=para_f['phosphor1']['slope'], pka=para_f['phosphor1']['pka'], pH=ph1)
    cotPhi_meas = boltzmann_sigmoid(top=para_f['meas']['top'], bottom=para_f['meas']['bottom'],
                                    slope=para_f['meas']['slope'], pka=para_f['meas']['pka'], pH=ph_meas)

    cotPhi_T0 = boltzmann_sigmoid(top=para_f['phosphor0']['top'], bottom=para_f['phosphor0']['bottom'],
                                  slope=para_f['phosphor0']['slope'], pka=para_f['phosphor0']['pka'], pH=ph_range)
    cotPhi_T1 = boltzmann_sigmoid(top=para_f['phosphor1']['top'], bottom=para_f['phosphor1']['bottom'],
                                  slope=para_f['phosphor1']['slope'], pka=para_f['phosphor1']['pka'], pH=ph_range)
    cotPhi_meas_ = boltzmann_sigmoid(top=para_f['meas']['top'], bottom=para_f['meas']['bottom'],
                                     slope=para_f['meas']['slope'], pka=para_f['meas']['pka'], pH=ph_range)

    cotPhi = pd.Series({'phosphor0': cotPhi_T0, 'phosphor1': cotPhi_T1, 'meas scan': cotPhi_meas_, 'meas': cotPhi_meas,
                        'fluoro0, phosphor0': cotPhi_ph0_T0, 'fluoro0, phosphor1': cotPhi_ph0_T1,
                        'fluoro1, phosphor0': cotPhi_ph1_T0, 'fluoro1, phosphor1': cotPhi_ph1_T1})

    # Phi of the mixed system at frequency f1
    Phi_ph0_T0 = np.rad2deg(af.arccot(cotPhi_ph0_T0))
    Phi_ph1_T0 = np.rad2deg(af.arccot(cotPhi_ph1_T0))
    Phi_ph0_T1 = np.rad2deg(af.arccot(cotPhi_ph0_T1))
    Phi_ph1_T1 = np.rad2deg(af.arccot(cotPhi_ph1_T1))
    Phi_deg_meas = np.rad2deg(af.arccot(cotPhi_meas))

    Phi_deg = pd.DataFrame(np.zeros(shape=(2, 2)), index=[ph0, ph1], columns=[T0, T1])
    Phi_deg.loc[ph0, T0] = Phi_ph0_T0
    Phi_deg.loc[ph1, T0] = Phi_ph1_T0
    Phi_deg.loc[ph0, T1] = Phi_ph0_T1
    Phi_deg.loc[ph1, T1] = Phi_ph1_T1
    print('Superimposed phaseangle at {:.2f}kHz: {:.2f}°C'.format(f/1000, Phi_deg_meas))

    if plot_T_comp is True:
        fig, ax_2 = plt.subplots()

        ax_2.plot(ph_range, cotPhi_T0, color='navy', lw=1.25)
        ax_2.plot(ph_range, cotPhi_T1, color='navy', lw=1.25)
        ax_2.plot(ph_range, cotPhi_meas_, color='forestgreen', lw=0.75)

        ax_2.plot(ph0, cotPhi_ph0_T0, marker='D', color='orange')
        ax_2.plot(ph1, cotPhi_ph1_T0, marker='D', color='orange')
        ax_2.plot(ph0, cotPhi_ph0_T1, marker='o', alpha=0.5, color='orange')
        ax_2.plot(ph1, cotPhi_ph1_T1, marker='o', alpha=0.5, color='orange')
        ax_2.plot(ph_meas, cotPhi_meas, marker='s', alpha=0.5, color='forestgreen')

        ax_2.axhline(para_f['phosphor0']['bottom'], color='k', lw=0.75, ls='--')
        ax_2.axhline(para_f['phosphor1']['bottom'], color='k', lw=0.75, ls='--')
        ax_2.axhline(para_f['meas']['bottom'], color='k', lw=0.45, ls='--')

        ax_2.set_ylabel('cot(dPhi)', color='navy', fontsize=13)
        ax_2.set_xlabel('pH', fontsize=13)
        ax_2.set_title('pH calibration simulation incl. T compensation', fontsize=13)

    return Phi_deg, Phi_deg_meas, cotPhi


def superimposed_phaseangle_temp_compensated(file, T0, T1, T_meas, ph0, ph1, ph_meas, ph_range, f1, f2, temp_range,
                                             fit_k=True, fit_bottom=True, fit_top=True, fit_pka=True, plotting_=True,
                                             plot_T_comp=True):
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2

    # Temperature compensation at f1
    [ddf, para_T0_f1, para_fit] = temperature_compensation(file=file, T_meas=T0, fit_k=fit_k, fit_bottom=fit_bottom,
                                                           f1=f1, f2=f2, temp_range=temp_range, fit_top=fit_top,
                                                           fit_pka=fit_pka, plotting=False)

    [ddf, para_T1_f1, para_fit] = temperature_compensation(file=file, T_meas=T1, fit_k=fit_k, fit_bottom=fit_bottom,
                                                           f1=f1, f2=f2, temp_range=temp_range, fit_top=fit_top,
                                                           fit_pka=fit_pka, plotting=False)

    [ddf, para_meas_f1, para_fit_meas] = temperature_compensation(file=file, T_meas=T_meas, fit_k=fit_k, f1=f1, f2=f2,
                                                                  fit_bottom=fit_bottom, temp_range=temp_range,
                                                                  fit_top=fit_top, fit_pka=fit_pka, plotting=plotting_)

    para_f1 = pd.Series({'phosphor0': para_T0_f1, 'phosphor1': para_T1_f1, 'meas': para_meas_f1})

    # Superimposed phase angle from Temperature compensation
    tauP_T0 = np.tan(af.arccot(para_f1['phosphor0']['bottom'])) / w1
    tauP_T1 = np.tan(af.arccot(para_f1['phosphor1']['bottom'])) / w1
    tauP_meas = np.tan(af.arccot(para_f1['meas']['bottom'])) / w1
    tauP = pd.Series({'phosphor0': tauP_T0, 'phosphor1': tauP_T1, 'meas': tauP_meas})

    phiP_f1 = pd.Series({'phosphor0': np.arctan(tauP_T0*w1), 'phosphor1': np.arctan(tauP_T1*w1),
                         'meas': np.arctan(tauP_meas*w1)})
    phiP_deg_f1 = np.rad2deg(phiP_f1)

    phiP_f2 = pd.Series({'phosphor0': np.arctan(tauP_T0*w2), 'phosphor1': np.arctan(tauP_T1*w2),
                         'meas': np.arctan(tauP_meas*w2)})
    phiP_deg_f2 = np.rad2deg(phiP_f2)

    # returning the phase angle in degree
    [Phi_deg_f1, Phi_deg_meas_f1,
     cotPhi_f1] = phi_from_cot(para_f=para_f1, T0=T0, T1=T1, ph0=ph0, ph1=ph1, ph_meas=ph_meas, f=f1, ph_range=ph_range,
                               plot_T_comp=plot_T_comp)

    return ddf, tauP, Phi_deg_f1, Phi_deg_meas_f1, phiP_f1, phiP_f2, phiP_deg_f1, phiP_deg_f2, para_f1, cotPhi_f1, \
           para_fit


def conversion_between_frequencies(cotPhi_f1, para_f1, phiP_f1, phiP_f2, tauP, w1, w2, ph_range, ph0, ph1):
    # amplitude ratio at f1 and different temperatures
    amp_T0_f1 = (cotPhi_f1['phosphor0'] - af.cot(phiP_f1['phosphor0'])) * np.sin(phiP_f1['phosphor0'])
    amp_T1_f1 = (cotPhi_f1['phosphor1'] - af.cot(phiP_f1['phosphor1'])) * np.sin(phiP_f1['phosphor1'])
    amp_meas_f1 = (cotPhi_f1['meas scan'] - af.cot(phiP_f1['meas'])) * np.sin(phiP_f1['meas'])

    # demodulation
    dm_T0_f1 = 1 / np.sqrt(1 + (w1*tauP['phosphor0'])**2)
    dm_T1_f1 = 1 / np.sqrt(1 + (w1*tauP['phosphor1'])**2)
    dm_meas_f1 = 1 / np.sqrt(1 + (w1*tauP['meas'])**2)

    dm_T0_f2 = 1 / np.sqrt(1 + (w2*tauP['phosphor0'])**2)
    dm_T1_f2 = 1 / np.sqrt(1 + (w2*tauP['phosphor1'])**2)
    dm_meas_f2 = 1 / np.sqrt(1 + (w2*tauP['meas'])**2)

    # amplitude ratio at f2
    amp_T0_f2 = amp_T0_f1 * dm_T0_f1 / dm_T0_f2
    amp_T1_f2 = amp_T1_f1 * dm_T1_f1 / dm_T1_f2
    amp_meas_f2 = amp_meas_f1 * dm_meas_f1 / dm_meas_f2

    cotPhi_f2_T0 = af.cot(phiP_f2['phosphor0']) + 1/np.sin(phiP_f2['phosphor0'])*amp_T0_f2
    cotPhi_f2_T1 = af.cot(phiP_f2['phosphor1']) + 1/np.sin(phiP_f2['phosphor1'])*amp_T1_f2
    cotPhi_f2_meas = af.cot(phiP_f2['meas']) + 1/np.sin(phiP_f2['meas'])*amp_meas_f2
    df_cotPhi_T0_f2 = pd.DataFrame(cotPhi_f2_T0, index=ph_range)
    df_cotPhi_T1_f2 = pd.DataFrame(cotPhi_f2_T1, index=ph_range)
    df_cotPhi_meas_f2 = pd.DataFrame(cotPhi_f2_meas, index=ph_range)

    # find closest value to calibration points
    a_ph0 = af.find_closest_value_(index=df_cotPhi_T0_f2.index, data=df_cotPhi_T0_f2, value=ph0)
    a_ph1 = af.find_closest_value_(index=df_cotPhi_T0_f2.index, data=df_cotPhi_T0_f2, value=ph1)
    b_ph0 = af.find_closest_value_(index=df_cotPhi_T1_f2.index, data=df_cotPhi_T1_f2, value=ph0)
    b_ph1 = af.find_closest_value_(index=df_cotPhi_T1_f2.index, data=df_cotPhi_T1_f2, value=ph1)
    c_ph0 = af.find_closest_value_(index=df_cotPhi_meas_f2.index, data=df_cotPhi_meas_f2, value=ph0)
    c_ph1 = af.find_closest_value_(index=df_cotPhi_meas_f2.index, data=df_cotPhi_meas_f2, value=ph1)

    # linear regression
    arg_T0_ph0 = stats.linregress(x=a_ph0[:2], y=a_ph0[2:])
    df_cotPhi_T0_f2_ph0 = arg_T0_ph0[0] * ph0 + arg_T0_ph0[1]

    arg_T0_ph1 = stats.linregress(x=a_ph1[:2], y=a_ph1[2:])
    df_cotPhi_T0_f2_ph1 = arg_T0_ph1[0] * ph1 + arg_T0_ph1[1]

    arg_T1_ph0 = stats.linregress(x=b_ph0[:2], y=b_ph0[2:])
    df_cotPhi_T1_f2_ph0 = arg_T1_ph0[0] * ph0 + arg_T1_ph0[1]

    arg_T1_ph1 = stats.linregress(x=b_ph1[:2], y=b_ph1[2:])
    df_cotPhi_T1_f2_ph1 = arg_T1_ph1[0] * ph1 + arg_T1_ph1[1]

    arg_meas_ph0 = stats.linregress(x=c_ph0[:2], y=c_ph0[2:])
    df_cotPhi_meas_f2_ph0 = arg_meas_ph0[0] * ph0 + arg_meas_ph0[1]

    arg_meas_ph1 = stats.linregress(x=c_ph1[:2], y=c_ph1[2:])
    df_cotPhi_meas_f2_ph1 = arg_meas_ph1[0] * ph1 + arg_meas_ph1[1]

    [y_T0_f2, para_T0_f2] = boltzmann_fit_(ph_range=ph_range, ph0=ph0, y0=df_cotPhi_T0_f2_ph0, ph1=ph1,
                                           y1=df_cotPhi_T0_f2_ph1, pk_a=para_f1['phosphor0']['pka'],
                                           slope=para_f1['phosphor0']['slope'])
    [y_T1_f2, para_T1_f2] = boltzmann_fit_(ph_range=ph_range, ph0=ph0, y0=df_cotPhi_T1_f2_ph0, ph1=ph1,
                                           y1=df_cotPhi_T1_f2_ph1, pk_a=para_f1['phosphor1']['pka'],
                                           slope=para_f1['phosphor1']['slope'])
    [y_meas_f2, para_meas_f2] = boltzmann_fit_(ph_range=ph_range, ph0=ph0, y0=df_cotPhi_meas_f2_ph0, ph1=ph1,
                                               y1=df_cotPhi_meas_f2_ph1, pk_a=para_f1['meas']['pka'],
                                               slope=para_f1['meas']['slope'])
    para_f2 = pd.Series({'phosphor0': para_T0_f2, 'phosphor1': para_T1_f2, 'meas': para_meas_f2})

    return para_f2


def linregression_temp_compensation_frequency(x, para_f, f):
    y_bottom = para_f[f]['phosphor0']['bottom'], para_f[f]['phosphor1']['bottom']
    y_top = para_f[f]['phosphor0']['top'], para_f[f]['phosphor1']['top']
    y_slope = para_f[f]['phosphor0']['slope'], para_f[f]['phosphor1']['slope']
    y_pka = para_f[f]['phosphor0']['pka'], para_f[f]['phosphor1']['pka']

    para_fit_f = pd.Series(0., index=['bottom, slope', 'bottom, intercept', 'top, slope', 'top, intercept',
                                      'slope, slope', 'slope, intercept', 'pka, slope', 'pka, intercept'])

    para_fit_f['bottom, slope'] = (y_bottom[1] - y_bottom[0]) / (x[1] - x[0])
    para_fit_f['top, slope'] = (y_top[1] - y_top[0]) / (x[1] - x[0])
    para_fit_f['slope, slope'] = (y_slope[1] - y_slope[0]) / (x[1] - x[0])
    para_fit_f['pka, slope'] = (y_pka[1] - y_pka[0]) / (x[1] - x[0])

    para_fit_f['bottom, intercept'] = y_bottom[1] - para_fit_f['bottom, slope']*x[1]
    para_fit_f['top, intercept'] = y_top[1] - para_fit_f['top, slope']*x[1]
    para_fit_f['slope, intercept'] = y_slope[1] - para_fit_f['slope, slope']*x[1]
    para_fit_f['pka, intercept'] = y_pka[1] - para_fit_f['pka, slope']*x[1]

    return para_fit_f


def input_simulation_pH_T(file, T0, T1, T_meas, temp_range, f1, f2, ph_range, ph0, ph1, ph_meas, er, plotting_=True,
                          fit_slope=True, fit_bottom=True, fit_top=True, fit_pka=True, plot_T_comp_f1=True,
                          plot_T_comp_f2=True):
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2

    # Superimposed phase angle at frequency f1 including temperature compensation
    [ddf, tauP, Phi_deg_f1, Phi_deg_meas_f1, phiP_f1, phiP_f2, phiP_deg_f1, phiP_deg_f2, para_f1, cotPhi_f1,
     para_fit] = superimposed_phaseangle_temp_compensated(file=file, T0=T0, T1=T1, T_meas=T_meas, ph0=ph0, ph1=ph1,
                                                          temp_range=temp_range, ph_meas=ph_meas, ph_range=ph_range,
                                                          f1=f1, f2=f2, fit_k=fit_slope, fit_bottom=fit_bottom,
                                                          fit_top=fit_top, fit_pka=fit_pka, plotting_=plotting_,
                                                          plot_T_comp=plot_T_comp_f1)

    # boltzmann fit for second modulation frequency f2
    para_f2 = conversion_between_frequencies(cotPhi_f1=cotPhi_f1, para_f1=para_f1, phiP_f1=phiP_f1, ph1=ph1,
                                             phiP_f2=phiP_f2, tauP=tauP, w1=w1, w2=w2, ph_range=ph_range, ph0=ph0)
    [Phi_deg_f2, Phi_deg_meas_f2, cotPhi_f2] = phi_from_cot(para_f=para_f2, T0=T0, T1=T1, ph0=ph0, ph1=ph1,
                                                            ph_meas=ph_meas, f=f2, ph_range=ph_range,
                                                            plot_T_comp=plot_T_comp_f2)

    # combining sensor parameters
    parameters = pd.Series({'tauP': tauP, 'phiP_deg_f1': phiP_deg_f1, 'phiP_deg_f2': phiP_deg_f2,
                            'T compensation': ddf})
    Phi_f1 = pd.Series({'calib, deg': Phi_deg_f1, 'meas, deg': Phi_deg_meas_f1})
    Phi_f2 = pd.Series({'calib, deg': Phi_deg_f2, 'meas, deg': Phi_deg_meas_f2})
    para_f = pd.Series({'f1': para_f1, 'f2': para_f2})
    cotPhi = pd.Series({'f1': cotPhi_f1, 'f2': cotPhi_f2})

    # spread imperfection
    [Phi_f1_deg_er, Phi_f2_deg_er, Phi_f1_er,
     Phi_f2_er] = spread_imperfection_Phi(ph0=ph0, ph1=ph1, T0=T0, T1=T1, Phi_f1=Phi_f1['calib, deg'],
                                          Phi_f2=Phi_f2['calib, deg'], er=er, Phi_meas_f1=Phi_f1['meas, deg'],
                                          Phi_meas_f2=Phi_f2['meas, deg'])

    return Phi_f1_deg_er, Phi_f1, Phi_f2_deg_er, Phi_f2, cotPhi, parameters, para_fit, para_f


def spread_imperfection_Phi(ph0, ph1, T0, T1, Phi_f1, Phi_f2, Phi_meas_f1, Phi_meas_f2, er):
    """

    :param ph0:
    :param ph1:
    :param T0:
    :param T1:
    :param Phi_f1:      superimposed phase angle in degree at modulation frequency f1
    :param Phi_f2:      superimposed phase angle in degree at modulation frequency f2
    :param Phi_meas_f1:
    :param Phi_meas_f2:
    :param er:
    :return:
    """
    # superimposed phaseangle in deg including measurement uncertainty
    Phi_ph0_T0_f1_deg_er = [Phi_f1.loc[ph0, T0] + er, Phi_f1.loc[ph0, T0], Phi_f1.loc[ph0, T0] - er]
    Phi_ph1_T0_f1_deg_er = [Phi_f1.loc[ph1, T0] + er, Phi_f1.loc[ph1, T0], Phi_f1.loc[ph1, T0] - er]
    Phi_ph0_T1_f1_deg_er = [Phi_f1.loc[ph0, T1] + er, Phi_f1.loc[ph0, T1], Phi_f1.loc[ph0, T1] - er]
    Phi_ph1_T1_f1_deg_er = [Phi_f1.loc[ph1, T1] + er, Phi_f1.loc[ph1, T1], Phi_f1.loc[ph1, T1] - er]
    Phi_meas_f1_deg_er = [Phi_meas_f1 + er, Phi_meas_f1, Phi_meas_f1 - er]

    Phi_ph0_T0_f2_deg_er = [Phi_f2.loc[ph0, T0] + er, Phi_f2.loc[ph0, T0], Phi_f2.loc[ph0, T0] - er]
    Phi_ph1_T0_f2_deg_er = [Phi_f2.loc[ph1, T0] + er, Phi_f2.loc[ph1, T0], Phi_f2.loc[ph1, T0] - er]
    Phi_ph0_T1_f2_deg_er = [Phi_f2.loc[ph0, T1] + er, Phi_f2.loc[ph0, T1], Phi_f2.loc[ph0, T1] - er]
    Phi_ph1_T1_f2_deg_er = [Phi_f2.loc[ph1, T1] + er, Phi_f2.loc[ph1, T1], Phi_f2.loc[ph1, T1] - er]
    Phi_meas_f2_deg_er = [Phi_meas_f2 + er, Phi_meas_f2, Phi_meas_f2 - er]

    Phi_f1_deg_er = pd.Series({'fluoro0, phosphor0': Phi_ph0_T0_f1_deg_er, 'fluoro1, phosphor0': Phi_ph1_T0_f1_deg_er,
                               'fluoro0, phosphor1': Phi_ph0_T1_f1_deg_er, 'fluoro1, phosphor1': Phi_ph1_T1_f1_deg_er,
                               'meas': Phi_meas_f1_deg_er})
    Phi_f2_deg_er = pd.Series({'fluoro0, phosphor0': Phi_ph0_T0_f2_deg_er, 'fluoro1, phosphor0': Phi_ph1_T0_f2_deg_er,
                               'fluoro0, phosphor1': Phi_ph0_T1_f2_deg_er, 'fluoro1, phosphor1': Phi_ph1_T1_f2_deg_er,
                               'meas': Phi_meas_f2_deg_er})

    # superimposed phaseangle in rad including measurement uncertainty
    Phi_ph0_T0_f1_er = np.deg2rad(Phi_f1_deg_er['fluoro0, phosphor0'])
    Phi_ph1_T0_f1_er = np.deg2rad(Phi_f1_deg_er['fluoro1, phosphor0'])
    Phi_ph0_T1_f1_er = np.deg2rad(Phi_f1_deg_er['fluoro0, phosphor1'])
    Phi_ph1_T1_f1_er = np.deg2rad(Phi_f1_deg_er['fluoro1, phosphor1'])
    Phi_meas_f1_er = np.deg2rad(Phi_f1_deg_er['meas'])

    Phi_ph0_T0_f2_er = np.deg2rad(Phi_f2_deg_er['fluoro0, phosphor0'])
    Phi_ph1_T0_f2_er = np.deg2rad(Phi_f2_deg_er['fluoro1, phosphor0'])
    Phi_ph0_T1_f2_er = np.deg2rad(Phi_f2_deg_er['fluoro0, phosphor1'])
    Phi_ph1_T1_f2_er = np.deg2rad(Phi_f2_deg_er['fluoro1, phosphor1'])
    Phi_meas_f2_er = np.deg2rad(Phi_f2_deg_er['meas'])

    Phi_f1_er = pd.Series({'fluoro0, phosphor0': Phi_ph0_T0_f1_er, 'fluoro1, phosphor0': Phi_ph1_T0_f1_er,
                           'fluoro0, phosphor1': Phi_ph0_T1_f1_er, 'fluoro1, phosphor1': Phi_ph1_T1_f1_er,
                           'meas': Phi_meas_f1_er})
    Phi_f2_er = pd.Series({'fluoro0, phosphor0': Phi_ph0_T0_f2_er, 'fluoro1, phosphor0': Phi_ph1_T0_f2_er,
                           'fluoro0, phosphor1': Phi_ph0_T1_f2_er, 'fluoro1, phosphor1': Phi_ph1_T1_f2_er,
                           'meas': Phi_meas_f2_er})

    return Phi_f1_deg_er, Phi_f2_deg_er, Phi_f1_er, Phi_f2_er


# --------------------------------------------------------------------------------------------------------------------
# Individual sensing
# --------------------------------------------------------------------------------------------------------------------
# Temperature sensing
def parameter_check(tauP_calib, ph, temp, conv_temp, para_temp_pH0, para_temp_pH1):
    temp0_test0 = ((tauP_calib.loc[ph[0], temp[0]]*1E6 -
                     para_temp_pH0['intercept [µs]']) / para_temp_pH0['slope [µs/K]']) - conv_temp
    temp1_test0 = ((tauP_calib.loc[ph[0], temp[1]]*1E6 -
                    para_temp_pH0['intercept [µs]']) / para_temp_pH0['slope [µs/K]']) - conv_temp

    temp0_test1 = ((tauP_calib.loc[ph[1], temp[0]]*1E6 -
                    para_temp_pH1['intercept [µs]']) / para_temp_pH1['slope [µs/K]']) - conv_temp
    temp1_test1 = ((tauP_calib.loc[ph[1], temp[1]]*1E6 -
                    para_temp_pH1['intercept [µs]']) / para_temp_pH1['slope [µs/K]']) - conv_temp

    # Parameter check - para_temp should be the same for pH0 and pH1
    if temp0_test0.round(2) != temp0_test1.round(2):
        print('ERROR - parameter don´t return the same calibration temperature')
        print('{:.2f}°C vs. {:.2f}°C'.format(temp0_test0, temp0_test1))
    else:
        if temp1_test0.round(2) != temp1_test1.round(2):
            print('ERROR - parameter don´t return the same calibration temperature')
            print('{:.2f}°C vs. {:.2f}°C'.format(temp1_test0, temp1_test1))
        else:
            print('parameter check successfully performed')


def temperature_calibration(Phi_f1_deg_er, Phi_f2_deg_er, f1, f2):
    # calibration points
    [tauP1_T0_ph0, tauP2_T0_ph0] = af.lifetime(phi1=Phi_f1_deg_er['fluoro0, phosphor0'],
                                               phi2=Phi_f2_deg_er['fluoro0, phosphor0'], f1=f1, f2=f2)
    [tauP1_T1_ph0, tauP2_T1_ph0] = af.lifetime(phi1=Phi_f1_deg_er['fluoro0, phosphor1'],
                                               phi2=Phi_f2_deg_er['fluoro0, phosphor1'], f1=f1, f2=f2)
    [tauP1_T0_ph1, tauP2_T0_ph1] = af.lifetime(phi1=Phi_f1_deg_er['fluoro1, phosphor0'],
                                               phi2=Phi_f2_deg_er['fluoro1, phosphor0'], f1=f1, f2=f2)
    [tauP1_T1_ph1, tauP2_T1_ph1] = af.lifetime(phi1=Phi_f1_deg_er['fluoro1, phosphor1'],
                                               phi2=Phi_f2_deg_er['fluoro1, phosphor1'], f1=f1, f2=f2)

    tau_T0_ph0_er = af.tau_selection(tauP1_T0_ph0, tauP2_T0_ph0)
    tau_T1_ph0_er = af.tau_selection(tauP1_T1_ph0, tauP2_T1_ph0)
    tau_T0_ph1_er = af.tau_selection(tauP1_T0_ph1, tauP2_T0_ph1)
    tau_T1_ph1_er = af.tau_selection(tauP1_T1_ph1, tauP2_T1_ph1)

    # -------------------------------------------------------------
    # measurement points
    list_phi_meas = []
    ls_tau_meas = {}
    for ind in Phi_f1_deg_er.index:
        if 'meas' in ind:
            list_phi_meas.append(ind)

    for i, num in enumerate(list_phi_meas):
        [tauP1_meas, tauP2_meas] = af.lifetime(phi1=Phi_f1_deg_er[num], phi2=Phi_f2_deg_er[num], f1=f1, f2=f2)
        tau_meas_er = af.tau_selection(tauP1_meas, tauP2_meas)
        ls_tau_meas[num] = tau_meas_er

    tauP_calib = pd.Series({'fluoro0, phosphor0': tau_T0_ph0_er, 'fluoro0, phosphor1': tau_T1_ph0_er,
                            'fluoro1, phosphor0': tau_T0_ph1_er, 'fluoro1, phosphor1': tau_T1_ph1_er})

    return tauP_calib, pd.Series(ls_tau_meas)


def temperature_sensing(Phi_f1_deg_er, Phi_f2_deg_er, T_calib, f1, f2, option='moderate', plotting=True, fontsize_=13):
    # calibration
    [tauP_calib, tau_meas] = temperature_calibration(Phi_f1_deg_er, Phi_f2_deg_er, f1, f2)
    for i in range(len(tau_meas)):
        print('Lifetime phosphor: {:.2f}µs'.format(tau_meas[i][1]*1E6))

    # linear regression
    # unit 1) slope := µs/K or µs/°C 2) intercept := µs
    slope_ph0 = (tauP_calib['fluoro0, phosphor1']*1E6 -
                 tauP_calib['fluoro0, phosphor0']*1E6) / (T_calib[1] - T_calib[0])
    slope_ph1 = (tauP_calib['fluoro1, phosphor1']*1E6 -
                 tauP_calib['fluoro1, phosphor0']*1E6) / (T_calib[1] - T_calib[0])
    intercept_ph0 = tauP_calib['fluoro0, phosphor1']*1E6 - slope_ph0*(T_calib[1] + conv_temp)
    intercept_ph1 = tauP_calib['fluoro1, phosphor1']*1E6 - slope_ph1*(T_calib[1] + conv_temp)
    para_temp = pd.Series({'fluoro0, slope [µs/K]': slope_ph0, 'fluoro0, intercept [µs]': intercept_ph0,
                           'fluoro1, slope [µs/K]': slope_ph1, 'fluoro1, intercept [µs]': intercept_ph1})

    T = T_calib[0]
    tau_fit_T0_all = [para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2]]

    T = T_calib[1]
    tau_fit_T1_all = [para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][2]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][1]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][0],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][1],
                      para_temp['fluoro0, slope [µs/K]'][0]*(T+conv_temp) + para_temp['fluoro0, intercept [µs]'][2]]

    y_min = [min(tau_fit_T0_all), min(tau_fit_T1_all)]
    y_max = [max(tau_fit_T0_all), max(tau_fit_T1_all)]
    y_mean = [(y_max[0] + y_min[0])/2, (y_max[1] + y_min[1])/2]

    # temperature determination at measurement point
    ls_a = {}
    ls_b = {}
    for i in range(len(tau_meas)):
        if option == 'pessimistic':
            a = (tau_meas[i][2]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][2]) / para_temp['fluoro0, slope [µs/K]'][0] - conv_temp
            b = (tau_meas[i][0]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][0]) / para_temp['fluoro0, slope [µs/K]'][2] - conv_temp
            # ylim_min = tau_meas[2]*1E6
            # ylim_max = tau_meas[0]*1E6
        elif option == 'moderate':
            a = (tau_meas[i][1]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][0]) / para_temp['fluoro0, slope [µs/K]'][2] - conv_temp
            b = (tau_meas[i][1]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][2]) / para_temp['fluoro0, slope [µs/K]'][0] - conv_temp
            # ylim_min = tau_meas[2]*1E6
            # ylim_max = tau_meas[0]*1E6
        elif option == 'optimistic':
            a = (tau_meas[i][2]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][0]) / para_temp['fluoro0, slope [µs/K]'][2] - conv_temp
            b = (tau_meas[i][0]*1E6 -
                 para_temp['fluoro0, intercept [µs]'][2]) / para_temp['fluoro0, slope [µs/K]'][0] - conv_temp
            # ylim_min = tau_meas[2]*1E6
            # ylim_max = tau_meas[0]*1E6
        else:
            raise ValueError('Select error propagation - pessimistic, moderate or optimisitc')
        ls_a[i] = a
        ls_b[i] = b

    ls_temp_std = {}
    ls_temp_mean = {}
    ls_temp_deg = {}
    for i in range(len(tau_meas)):
        temp_std = np.abs((ls_a[i]-ls_b[i])/2)
        temp_mean = (ls_a[i]+ls_b[i])/2
        temp_deg = [temp_mean - temp_std, temp_mean, temp_mean + temp_std]
        ls_temp_std[i] = temp_std
        ls_temp_mean[i] = temp_mean
        ls_temp_deg[i] = temp_deg
        print('Measurement point', i+1)
        print('Temperature calculated: {:.2f} ± {:.2f}°C'.format(temp_mean, temp_std))
    tau_phos = [y_min, y_mean, y_max]
    res_T = pd.Series({'T': T_calib, 'tauP': tau_phos})

    # plotting
    if plotting is True:
        f, ax = plt.subplots()

        ax.fill_between(T_calib, y_min, y_max, color='gray', alpha=0.1, lw=0.25)
        ax.plot(T_calib, y_mean, lw=1, color='navy')
        ax.axvline(a, color='k', alpha=0.5, ls='--', lw=0.75)
        ax.axvline(b, color='k', alpha=0.5, ls='--', lw=0.75)

        ax.set_xlabel('Temperature [°C]', fontsize=fontsize_)
        ax.set_ylabel('tau$_{Phosphor}$ [µs]', fontsize=fontsize_)

    return temp_deg, para_temp, res_T


# -------------------------------------------------------------------
# pH sensing
def boltzmann_sig_scan(boltz_para, i, ph_scan):
    a = boltzmann_sigmoid(top=boltz_para.loc[i, 'top'], bottom=boltz_para.loc[i, 'bottom'],
                          slope=boltz_para.loc[i, 'slope'], pka=boltz_para.loc[i, 'pka'], pH=ph_scan)
    return a


def boltzmann_fit_(ph_range, ph0, y0, ph1, y1, pk_a, slope):
    alpha = 1 + 10**((ph0 - pk_a)/slope)
    beta = 1 + 10**((ph1 - pk_a)/slope)

    if isinstance(y0, float) and isinstance(y1, float):
        top = y0 + (1 - 1/alpha)*(y0 - y1) / (1/alpha - 1/beta)
        bottom = top - (y0 - y1) / (1/alpha - 1/beta)
        cot_phi = bottom + (top - bottom) / (1 + 10**((ph_range - pk_a)/slope))
    else:
        top_ = [(1 - 1/alpha)*i / (1/alpha - 1/beta) for i in list(np.array(y0) - np.array(y1))]
        top = list(np.array(y0) + np.array(top_))
        bottom_ = [j / (1/alpha - 1/beta) for j in list(np.array(y0) - np.array(y1))]
        bottom = (list(np.array(top) - np.array(bottom_)))

        p = [k / (1 + 10**((ph_range - pk_a)/slope)) for k in list(np.array(top) - np.array(bottom))]

        cot_phi = [bottom[0] + p[0], bottom[1] + p[1], bottom[2] + p[2]]

    para = pd.Series({'top': top, 'bottom': bottom, 'pka': pk_a, 'slope': slope})

    return cot_phi, para


def pH_calculation(reg_top_f, reg_bottom_f, reg_pka_f, reg_slope_f, Phi_f1_deg_er, ph_range, T_calc_deg):
    # Boltzmann sigmoid parameter at measured temperature (in °C)
    bottom_f = [t*reg_bottom_f['slope'] for t in T_calc_deg] + reg_bottom_f['intercept']
    pka_f = [t*reg_pka_f['slope'] for t in T_calc_deg] + reg_pka_f['intercept']
    top_f = [t*reg_top_f['slope'] for t in T_calc_deg] + reg_top_f['intercept']
    slope_f = [t*reg_slope_f['slope'] for t in T_calc_deg] + reg_slope_f['intercept']
    para_meas = pd.Series({'bottom': bottom_f, 'top': top_f, 'pka': pka_f, 'slope': slope_f})

    # pH determination according to Boltzmann parameter
    ph_screen_f = []
    for i in ph_range:
        ph_screen_f.append(bottom_f + (top_f - bottom_f)/(1 + 10**((i - pka_f) / slope_f)))
    df_ph_screen_f = pd.DataFrame(ph_screen_f, index=ph_range)

    pH_calc = pka_f + slope_f*np.log10((top_f - bottom_f) / (af.cot(np.deg2rad(Phi_f1_deg_er['meas'])) - bottom_f) - 1)

    return pH_calc, df_ph_screen_f, para_meas


def pH_calibration(para_fit):
    reg_top = pd.Series({'slope': para_fit['top, slope'], 'intercept': para_fit['top, intercept']})
    reg_bottom = pd.Series({'slope': para_fit['bottom, slope'], 'intercept': para_fit['bottom, intercept']})
    reg_pka = pd.Series({'slope': para_fit['pka, slope'], 'intercept': para_fit['pka, intercept']})
    reg_slope = pd.Series({'slope': para_fit['slope, slope'], 'intercept': para_fit['slope, intercept']})

    return reg_top, reg_bottom, reg_pka, reg_slope


def pH_recheck(temp_deg, temp, temp_range_deg, para_fit_f, reg_slope, reg_bottom, reg_pka, reg_top):
    fig, ax_fit = plt.subplots(ncols=2, nrows=2, sharex=True)
    # slope
    ax_fit[0][0].axvline(temp_deg[1], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[0][0].axvline(temp[0], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[0][0].axvline(temp[1], lw=1, ls='--', color='k', alpha=0.4)

    ax_fit[0][0].axhline(temp_deg[1]*reg_slope['slope'] + reg_slope['intercept'], lw=1, ls='--', color='navy',
                         label='fit T_meas')
    ax_fit[0][0].axhline(temp[0]*reg_slope['slope'] + reg_slope['intercept'], lw=1, ls='--', color='navy',
                         label='fit T0')
    ax_fit[0][0].axhline(temp[1]*reg_slope['slope'] + reg_slope['intercept'], lw=1, ls='--', color='navy',
                         label='fit T1')
    ax_fit[0][0].axhline(para_fit_f['slope, intercept'], ls='--', color='orange', label='sim')
    ax_fit[0][0].legend(loc=1, ncol=2, fontsize=8)

    # -------------------------------------------------------------
    # bottom
    ax_fit[0][1].axvline(temp_deg[1], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[0][1].axvline(temp[0], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[0][1].axvline(temp[1], lw=1, ls='--', color='k', alpha=0.4)

    ax_fit[0][1].axhline(temp_deg[1]*reg_bottom['slope'] + reg_bottom['intercept'], lw=1, ls='--', color='navy',
                         label='fit T_meas')
    ax_fit[0][1].axhline(temp[0]*reg_bottom['slope'] + reg_bottom['intercept'], lw=1, ls='--', color='navy',
                         label='fit T0')
    ax_fit[0][1].axhline(temp[1]*reg_bottom['slope'] + reg_bottom['intercept'], lw=1, ls='--', color='navy',
                         label='fit T1')
    ax_fit[0][1].plot(temp_range_deg, para_fit_f['bottom, slope']*temp_range_deg+para_fit_f['bottom, intercept'], lw=1,
                      ls='--', color='darkorange', label='sim')
    ax_fit[0][1].legend(loc=1, ncol=2, fontsize=8)

    # -------------------------------------------------------------
    # pka
    ax_fit[1][0].axvline(temp_deg[1], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[1][0].axvline(temp[0], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[1][0].axvline(temp[1], lw=1, ls='--', color='k', alpha=0.4)

    ax_fit[1][0].axhline(temp_deg[1]*reg_pka['slope'] + reg_pka['intercept'], lw=1, ls='--', color='navy',
                         label='fit T_meas')
    ax_fit[1][0].axhline(temp[0]*reg_pka['slope'] + reg_pka['intercept'], lw=1,  ls='--', color='navy', label='fit T0')
    ax_fit[1][0].axhline(temp[1]*reg_pka['slope'] + reg_pka['intercept'], lw=1, ls='--', color='navy', label='fit T1')
    ax_fit[1][0].plot(temp_range_deg, para_fit_f['pka, slope']*temp_range_deg+para_fit_f['pka, intercept'], lw=1,
                      ls='--', color='darkorange', label='sim')
    ax_fit[1][0].legend(loc=1, ncol=2, fontsize=8)

    # -------------------------------------------------------------
    # top
    ax_fit[1][1].axvline(temp_deg[1], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[1][1].axvline(temp[0], lw=1, ls='--', color='k', alpha=0.4)
    ax_fit[1][1].axvline(temp[1], lw=1, ls='--', color='k', alpha=0.4)

    ax_fit[1][1].axhline(temp_deg[1]*reg_top['slope'] + reg_top['intercept'], lw=1, ls='--', color='navy',
                         label='fit Tmeas')
    ax_fit[1][1].axhline(temp[0]*reg_top['slope'] + reg_top['intercept'], lw=1, ls='--', color='navy', label='fit T0')
    ax_fit[1][1].axhline(temp[1]*reg_top['slope'] + reg_top['intercept'], lw=1, ls='--', color='navy', label='fit T1')
    ax_fit[1][1].axhline(temp[0]*para_fit_f['top, slope'] + para_fit_f['top, intercept'], ls='--', color='orange',
                         lw=1., label='sim')
    ax_fit[1][1].legend(loc=0, ncol=2, fontsize=8)

    # -------------------------------------------------------------
    # layout
    ax_fit[0][0].set_ylabel('slope', fontsize=13)
    ax_fit[1][0].set_ylabel('pka', fontsize=13)
    ax_fit[0][1].set_ylabel('Bottom', fontsize=13)
    ax_fit[1][1].set_ylabel('Top', fontsize=13)
    ax_fit[1][0].set_xlabel('temperature [°C]', fontsize=13)
    ax_fit[1][1].set_xlabel('temperature [°C]', fontsize=13)

    plt.tight_layout()


def pH_sensing(para_fit, Phi_f1_deg_er, ph_range, temp, T_range_K, T_calc_deg, f, plotting=True, fontsize_=13,
               re_check=True):

    # pH Calibration
    # linear regression for slope, top, bottom and pka
    [reg_top_f, reg_bottom_f, reg_pka_f, reg_slope_f] = pH_calibration(para_fit=para_fit)

    if re_check is True:
        temp_range_deg = T_range_K - conv_temp
        pH_recheck(temp_deg=T_calc_deg, temp=temp, temp_range_deg=temp_range_deg, para_fit_f=para_fit,
                   reg_slope=reg_slope_f, reg_bottom=reg_bottom_f, reg_pka=reg_pka_f, reg_top=reg_top_f)

    # pH calculation at certain measurement temperature T_calc_deg
    [pH_calc, df_ph_screen_f, para_meas] = pH_calculation(reg_top_f=reg_top_f, reg_bottom_f=reg_bottom_f,
                                                          reg_pka_f=reg_pka_f, reg_slope_f=reg_slope_f,
                                                          Phi_f1_deg_er=Phi_f1_deg_er, ph_range=ph_range,
                                                          T_calc_deg=T_calc_deg)
    if True in np.isnan(pH_calc):
        if np.isnan(pH_calc[0]) and np.isnan(pH_calc[2]):
            std = '--'
        elif np.isnan(pH_calc[0]) and pH_calc[2] != np.nan:
            std = np.abs(pH_calc[2] - pH_calc[1])
            std = "%.2f" % std
        elif np.isnan(pH_calc[2]) and pH_calc[0] != np.nan:
            std = np.abs(pH_calc[0] - pH_calc[1])
            std = "%.2f" % std
        elif pH_calc[0] != np.nan and pH_calc[2] != np.nan:
            std = np.abs(pH_calc[2] - pH_calc[0]) / 2
            std = "%.2f" % std
        else:
            std = '--'
        if np.isnan(pH_calc[1]):
            if pH_calc[0] != np.nan:
                pH = pH_calc[0]
            elif pH_calc[2] != np.nan:
                pH = pH_calc[2]
            else:
                pH = pH_calc[1]
        else:
            pH = pH_calc[1]
    else:
        std = np.abs(pH_calc[2] - pH_calc[0])/2
        std = "%.2f" % std
        pH = pH_calc[1]

    # Visualization
    if plotting is True:
        fig, ax_ph = plt.subplots()
        ax1 = ax_ph.twinx()

        ax1.plot(ph_range, np.rad2deg(af.arccot(df_ph_screen_f[1])), color='forestgreen')
        ax1.fill_between(ph_range, np.rad2deg(af.arccot(df_ph_screen_f[0])), np.rad2deg(af.arccot(df_ph_screen_f[2])),
                         color='gray', lw=.25, alpha=0.1)
        ax_ph.plot(ph_range, df_ph_screen_f[1], color='steelblue')
        ax_ph.fill_between(ph_range, df_ph_screen_f[0], df_ph_screen_f[2], color='gray', lw=.25, alpha=0.1)

        # pH calculated as marker
        xerr = af.cot(np.deg2rad(Phi_f1_deg_er['meas'][2])) - af.cot(np.deg2rad(Phi_f1_deg_er['meas'][0]))
        phi_xerr = Phi_f1_deg_er['meas'][2] - Phi_f1_deg_er['meas'][0]
        ax_ph.errorbar(pH_calc[1], af.cot(np.deg2rad(Phi_f1_deg_er['meas'][1])), xerr=xerr, fmt='o', color='orange')
        ax1.errorbar(pH_calc[1], (Phi_f1_deg_er['meas'][1]), xerr=phi_xerr, fmt='s', color='orange')

        ax_ph.axhline(af.cot(np.deg2rad(Phi_f1_deg_er['meas'][0])), lw=0.75, color='k', ls='--')
        ax_ph.axvline(pH_calc[0], lw=0.75, color='k', ls='--')
        ax_ph.axvline(pH_calc[1], lw=0.75, color='k', ls='--')
        ax_ph.axvline(pH_calc[2], lw=0.75, color='k', ls='--')

        ax_ph.set_xlabel('pH', fontsize=fontsize_)
        ax_ph.set_ylabel('cot(dPhi) at {:.1f}Hz'.format(f/1000), color='k', fontsize=fontsize_)
        ax1.set_ylabel('dPhi [°] at {:.1f}Hz'.format(f/1000), color='forestgreen', fontsize=fontsize_)

        plt.tight_layout()

    return pH_calc, std, df_ph_screen_f


# --------------------------------------------------------------------------------------------------------------------
# Dual sensing
# --------------------------------------------------------------------------------------------------------------------
# Plotting dualsensor
def plot_dualsensor_ph_T(temp_scan_K, ph_scan, ph_range, res_T, temp_deg, pH_meas, Phi_f1_deg_er, Phi_f2_deg_er,
                         df_ph_screen_f1, df_ph_screen_f2, dual_scan, f1, f2, which='both', fontsize_=13):
    Xf, Xp = np.meshgrid(temp_scan_K - conv_temp, ph_scan)

    # 3D plot
    fig_3d = plt.figure(figsize=(8, 7.5))
    ax_f1 = fig_3d.add_subplot(221, projection='3d')
    ax_f2 = fig_3d.add_subplot(222, projection='3d')
    ax_temp = fig_3d.add_subplot(223)
    ax_pH = fig_3d.add_subplot(224)
    ax_pH1 = ax_pH.twinx()
    ax_f1.view_init(35, -60)
    ax_f2.view_init(35, -60)

    # --------------------------------------------------------------------------------------
    # Temperature sensing
    ax_temp.plot(res_T['T'], res_T['tauP'][1], color='navy', lw=1.)
    ax_temp.fill_between(res_T['T'], res_T['tauP'][0], res_T['tauP'][2], color='grey', alpha=0.1, lw=0.25)

    ax_temp.axvline(temp_deg[0], color='grey', ls='--', lw=.75)
    ax_temp.axvline(temp_deg[2], color='grey', ls='--', lw=.75)
    ax_temp.set_xlabel('Temp [°C]', fontsize=fontsize_)
    ax_temp.set_ylabel('Lifetime [µs]', fontsize=fontsize_)

    # --------------------------------------------------------------------------------------
    # pH sensing

    if which == 'f1':
        # cot(dPhi) and dPhi
        ax_pH.plot(ph_range, df_ph_screen_f1[1], color='steelblue', lw=1.)
        ax_pH.fill_between(ph_range, df_ph_screen_f1[0], df_ph_screen_f1[2], color='grey', alpha=0.1, lw=0.25)
        ax_pH1.plot(ph_range, np.rad2deg(af.arccot(df_ph_screen_f1[1])), color='forestgreen', lw=1.)
        ax_pH1.fill_between(ph_range, np.rad2deg(af.arccot(df_ph_screen_f1[0])),
                            np.rad2deg(af.arccot(df_ph_screen_f1[2])), color='grey', alpha=0.1, lw=0.25)

        # pH calculated as marker
        ax_pH1.plot(pH_meas['f1'][1], Phi_f1_deg_er['meas'][1], marker='d', color='orange')
        ax_pH.plot(pH_meas['f1'][1], af.cot(np.deg2rad(Phi_f1_deg_er['meas'][1])), marker='d', color='orange')

        ax_pH1.axhline(Phi_f1_deg_er['meas'][1], color='k', ls='--', lw=.5)
        ax_pH1.axvline(pH_meas['f1'][1], color='k', ls='--', lw=.5)

    elif which == 'f2':
        # cot(dPhi) and dPhi
        ax_pH.plot(ph_range, df_ph_screen_f2[1], color='steelblue', lw=1.)
        ax_pH.fill_between(ph_range, df_ph_screen_f2[0], df_ph_screen_f2[2], color='grey', alpha=0.1, lw=0.25)
        ax_pH1.plot(ph_range, np.rad2deg(af.arccot(df_ph_screen_f2[1])), color='forestgreen', lw=1.)
        ax_pH1.fill_between(ph_range, np.rad2deg(af.arccot(df_ph_screen_f2[0])),
                            np.rad2deg(af.arccot(df_ph_screen_f2[2])), color='grey', alpha=0.1, lw=0.25)

        # pH calculated as marker
        ax_pH1.plot(pH_meas['f2'][1], Phi_f2_deg_er['meas'][1], marker='d', color='orange')
        ax_pH.plot(pH_meas['f2'][1], af.cot(np.deg2rad(Phi_f2_deg_er['meas'][1])), marker='d', color='orange')

        ax_pH1.axhline(Phi_f2_deg_er['meas'][1], color='k', ls='--', lw=.5)
        ax_pH1.axvline(pH_meas['f2'][1], color='k', ls='--', lw=.5)

    elif which == 'both':
        # cot(dPhi) and dPhi
        ax_pH.plot(ph_range, df_ph_screen_f1[1], color='steelblue', lw=1.)
        ax_pH.plot(ph_range, df_ph_screen_f2[1], color='steelblue', lw=1.)

        ax_pH1.plot(ph_range, np.rad2deg(af.arccot(df_ph_screen_f1[1])), color='forestgreen', lw=1.)
        ax_pH1.plot(ph_range, np.rad2deg(af.arccot(df_ph_screen_f2[1])), color='forestgreen', lw=1.)

        # pH calculated as marker
        ax_pH1.plot(pH_meas['f1'][1], Phi_f1_deg_er['meas'][1], marker='d', color='orange')
        ax_pH.plot(pH_meas['f1'][1], af.cot(np.deg2rad(Phi_f1_deg_er['meas'][1])), marker='d', color='orange')
        ax_pH1.plot(pH_meas['f2'][1], Phi_f2_deg_er['meas'][1], marker='d', color='orange')
        ax_pH.plot(pH_meas['f2'][1], af.cot(np.deg2rad(Phi_f2_deg_er['meas'][1])), marker='d', color='orange')
        ax_pH1.axvline(pH_meas['f1'][1], color='k', ls='--', lw=.5)

    ax_pH.set_xlabel('pH', fontsize=fontsize_)
    ax_pH.set_ylabel('cot(dPhi)', fontsize=fontsize_)
    ax_pH1.set_ylabel('dPhi [deg]', fontsize=fontsize_)

    # ------------------------------------------------------
    # dPhi(f1)
    ax_f1.plot_surface(Xf, Xp, np.rad2deg(af.arccot(dual_scan['f1'])).T, rstride=1, cstride=1, cmap='ocean',
                       edgecolor=None)
    ax_f1.set_xlabel('Temp [°C]', fontsize=fontsize_)
    ax_f1.set_ylabel('pH', fontsize=fontsize_)
    ax_f1.set_zlabel('dPhi(f1) [deg]', fontsize=fontsize_*0.8)
    ax_f1.set_title('Phase angle {:.1f}kHz'.format(f1/1000), fontsize=fontsize_, loc='left')

    # ------------------------------------------------------
    # dPhi(f2)
    ax_f2.plot_surface(Xf, Xp, np.rad2deg(af.arccot(dual_scan['f2'])).T, rstride=1, cstride=1, cmap='ocean',
                       edgecolor=None)
    ax_f2.set_xlabel('Temp [°C]', fontsize=fontsize_)
    ax_f2.set_ylabel('pH', fontsize=fontsize_)
    ax_f2.set_zlabel('dPhi(f2) [deg]', fontsize=fontsize_*0.8)
    ax_f2.set_title('Phase angle {:.1f}kHz'.format(f2/1000), fontsize=fontsize_, loc='left')

    plt.subplots_adjust(left=0.1, right=0.89, wspace=.25, hspace=.4)

    return


def boltzmann_regression(y, temp_scan_K):
    bottom_T = y['bottom, slope']*(temp_scan_K-conv_temp) + y['bottom, intercept']
    top_T = y['top, slope']*(temp_scan_K-conv_temp) + y['top, intercept']
    slope_T = y['slope, slope']*(temp_scan_K-conv_temp) + y['slope, intercept']
    pka_T = y['pka, slope']*(temp_scan_K-conv_temp) + y['pka, intercept']

    boltzmann_para = pd.concat([pd.DataFrame(bottom_T, index=temp_scan_K.round(2)),
                                   pd.DataFrame(top_T, index=temp_scan_K.round(2)),
                                   pd.DataFrame(slope_T, index=temp_scan_K.round(2)),
                                   pd.DataFrame(pka_T, index=temp_scan_K.round(2))], axis=1)
    boltzmann_para.columns = ['bottom', 'top', 'slope', 'pka']

    return boltzmann_para


def scanning_ph_T(temp_range_K, para_temp, para_fit_f1, para_fit_f2, temp_calib, steps):
    # temperature simulation
    temp_scan_K = np.linspace(start=temp_range_K[0], stop=temp_range_K[-1],
                              num=int((temp_range_K[-1]-temp_range_K[0])/steps + 1.))
    ph_scan = np.linspace(start=0, stop=14, num=len(temp_scan_K))

    a = [s* temp_scan_K for s in para_temp['fluoro0, slope [µs/K]']]
    tauP_scan = a + para_temp['fluoro0, intercept [µs]'][0]

    # ---------------------------------------------------------------
    # top, bottom, pka and slope at certain T and different modulation frequencies
    x = temp_calib # °C
    boltzmann_para_f1 = boltzmann_regression(y=para_fit_f1, temp_scan_K=temp_scan_K)
    boltzmann_para_f2 = boltzmann_regression(y=para_fit_f2, temp_scan_K=temp_scan_K)
    boltzmann_para = pd.Series({'f1': boltzmann_para_f1, 'f2': boltzmann_para_f2})

    # --------------------------------------------------------------------------------------------------------------
    # single signal
    cot_dPhi_f1 = boltzmann_sigmoid(top=boltzmann_para_f1['top'], bottom=boltzmann_para_f1['bottom'],
                                    slope=boltzmann_para_f1['slope'], pka=boltzmann_para_f1['pka'], pH=ph_scan)
    cot_dPhi_f2 = boltzmann_sigmoid(top=boltzmann_para_f2['top'], bottom=boltzmann_para_f2['bottom'],
                                    slope=boltzmann_para_f2['slope'], pka=boltzmann_para_f2['pka'], pH=ph_scan)
    cot_dPhi = pd.Series({'f1': cot_dPhi_f1, 'f2': cot_dPhi_f2})

    # ---------------------------------------------------------------
    # pH/T scan - dual signal
    dual_scan_f1 = pd.DataFrame(np.zeros(shape=(len(temp_scan_K), len(ph_scan))), index=temp_scan_K.round(2),
                                columns=ph_scan)
    dual_scan_f2 = pd.DataFrame(np.zeros(shape=(len(temp_scan_K), len(ph_scan))), index=temp_scan_K.round(2),
                                columns=ph_scan)

    for i in temp_scan_K.round(2):
        dual_scan_f1.loc[i] = boltzmann_sig_scan(boltz_para=boltzmann_para_f1, i=i, ph_scan=ph_scan.round(2))
        dual_scan_f2.loc[i] = boltzmann_sig_scan(boltz_para=boltzmann_para_f2, i=i, ph_scan=ph_scan.round(2))
    dual_scan = pd.Series({'f1': dual_scan_f1, 'f2': dual_scan_f2})

    return dual_scan, boltzmann_para, tauP_scan, temp_scan_K, ph_scan, cot_dPhi


# -------------------------------------------------------------------
def pH_T_dualsensing_sim(Phi_f1_deg_er, Phi_f2_deg_er, temp_calib, f1, f2, ph_range, temp_range_K, para_fit_f1,
                         para_fit_f2, option='moderate', fit_bottom=True, plot=True, which='both', steps_T=5,
                         fontsize_=13, re_check=True):

    # Temperature sensing
    [temp_deg, para_temp,
     res_T] = temperature_sensing(Phi_f1_deg_er=Phi_f1_deg_er, Phi_f2_deg_er=Phi_f2_deg_er, T_calib=temp_calib,
                                  f1=f1, f2=f2, option=option, plotting=False, fontsize_=fontsize_)

    # ----------------------------------------------------------------------------------------------
    # pH sensing
    pH_f1, std_f1, df_ph_screen_f1 = pH_sensing(para_fit=para_fit_f1, Phi_f1_deg_er=Phi_f1_deg_er, ph_range=ph_range,
                                                temp=temp_calib, T_range_K=temp_range_K, T_calc_deg=temp_deg, f=f1,
                                                plotting=False, re_check=re_check)
    pH_f2, std_f2, df_ph_screen_f2 = pH_sensing(para_fit=para_fit_f2, Phi_f1_deg_er=Phi_f2_deg_er, ph_range=ph_range,
                                                temp=temp_calib, T_range_K=temp_range_K, T_calc_deg=temp_deg, f=f2,
                                                plotting=False, re_check=re_check)
    pH_meas = pd.Series({'f1': pH_f1, 'f2': pH_f2, 'std': std_f1})
    pH = (pH_f1[1] + pH_f2[1])/2
    print('Calculated pH: {:.2f} ± {} at T = {:.2f}°C'.format(pH, std_f1, temp_deg[1]))

    # ----------------------------------------------------------------------------------------------
    [dual_scan, boltzmann_para, df_tauP_scan, temp_scan_K, ph_scan,
     cot_dPhi] = scanning_ph_T(temp_range_K=temp_range_K, para_temp=para_temp, para_fit_f1=para_fit_f1,
                               para_fit_f2=para_fit_f2, steps=steps_T, temp_calib=temp_calib)
    # Plotting
    if plot is True:
        # plotting dual sensor
        plot_dualsensor_ph_T(temp_scan_K=temp_scan_K, ph_scan=ph_scan, ph_range=ph_range, res_T=res_T, f1=f1, f2=f2,
                             temp_deg=temp_deg, pH_meas=pH_meas, fontsize_=fontsize_, Phi_f1_deg_er=Phi_f1_deg_er,
                             Phi_f2_deg_er=Phi_f2_deg_er, df_ph_screen_f1=df_ph_screen_f1, dual_scan=dual_scan,
                             df_ph_screen_f2=df_ph_screen_f2, which=which)
    else:
        dual_scan = None
        boltzmann_para = None
        df_tauP_scan = None

    return temp_deg, pH_meas, para_temp, dual_scan, boltzmann_para, df_tauP_scan, temp_scan_K, res_T, cot_dPhi, ph_scan


def pH_T_dualsensing_meas(file, Phi_f1_deg_er, Phi_f2_deg_er, f1, f2, T_calib, temp_range_deg, plotting=True,
                          fontsize_=13):
    # T sensing
    # lifetime from phase angle
    tauP_calib, tauP_meas_er = temperature_calibration(Phi_f1_deg_er=Phi_f1_deg_er, Phi_f2_deg_er=Phi_f2_deg_er, f1=f1,
                                                       f2=f2)

    # --------------------------------------------------------------------------
    # temperature calibration
    x = T_calib
    y_T0_min = [tauP_calib['fluoro0, phosphor0'][0]*1E6, tauP_calib['fluoro1, phosphor0'][0]*1E6]
    y_T0_max = [tauP_calib['fluoro0, phosphor0'][2]*1E6, tauP_calib['fluoro1, phosphor0'][2]*1E6]

    y_T1_min = [tauP_calib['fluoro0, phosphor1'][0]*1E6, tauP_calib['fluoro1, phosphor1'][0]*1E6]
    y_T1_max = [tauP_calib['fluoro0, phosphor1'][2]*1E6, tauP_calib['fluoro1, phosphor1'][2]*1E6]

    y_meas = tauP_meas_er*1E6

    # linear regression for calibration
    [slope_ph1_min, intercept_ph1_min, r_value_ph1_min,
     p_value_ph1_min, std_err_ph1_min] = stats.linregress(x=x, y=[y_T0_min[1], y_T1_min[1]])
    [slope_ph1_max, intercept_ph1_max, r_value_ph1_max,
     p_value_ph1_max, std_err_ph1_max] = stats.linregress(x=x, y=[y_T0_max[1], y_T1_max[1]])

    y_reg_ph1 = pd.DataFrame(np.zeros(shape=(len(temp_range_deg), 2)), index=temp_range_deg, columns=['min', 'max'])
    y_reg_ph1['min'] = slope_ph1_min*temp_range_deg + intercept_ph1_min
    y_reg_ph1['max'] = slope_ph1_max*temp_range_deg + intercept_ph1_max

    # --------------------------------------------------------------------------
    # calculate temperature at pH = 10 (no fluorescence present as background)
    ls_tauP = {}
    ls_T = {}
    for i in range(len(y_meas)):
        [tau_min1, tau_max1, temp_min1, temp_max1] = af.find_closest_value_(index=y_reg_ph1['min'].values,
                                                                            data=y_reg_ph1['min'].index,
                                                                            value=y_meas[i][1])
        [tau_min2, tau_max2, temp_min2, temp_max2] = af.find_closest_value_(index=y_reg_ph1['max'].values,
                                                                            data=y_reg_ph1['max'].index,
                                                                            value=y_meas[i][1])

        tau_phos_ = [tau_min1, tau_max2]
        temp_calc_ = [temp_min1, temp_max2]
        temp_calc = [temp_calc_[0], (temp_calc_[1] + temp_calc_[0]) / 2, temp_calc_[1]]
        tau_phos = [tau_phos_[0], (tau_phos_[1] + tau_phos_[0]) / 2, tau_phos_[1]]

        tau_phos = sorted(tau_phos, reverse=True)
        temp_calc = sorted(temp_calc)
        print('Calculated T = {:.2f} +/- {:.2f}°C'.format(temp_calc[1], (temp_calc[2]-temp_calc[0])/2))

        ls_tauP[i] = tau_phos
        ls_T[i] = temp_calc

# ---------------------------------------------------------------------------------------------------------------------
    # pH sensing
    # Temperature compensation at f1 and at T_calc
    ls_para_calc_f1 = {}
    ls_para_f1_min = {}
    ls_para_f1_max = {}
    for i in range(len(y_meas)):
        [ddf, para_calc_f1_min,
         para_fit_min] = temperature_compensation(file=file, T_meas=ls_T[i][0], fit_k=False, fit_bottom=True, f1=f1,
                                                  f2=f2, temp_range=temp_range_deg, fit_top=True, fit_pka=True,
                                                  plotting=False)

        [ddf, para_calc_f1_max,
         para_fit_max] = temperature_compensation(file=file, T_meas=ls_T[i][2], fit_k=False, fit_bottom=True, f1=f1,
                                                  f2=f2, temp_range=temp_range_deg, fit_top=True, fit_pka=True,
                                                  plotting=False)

        para_calc_f1 = (para_calc_f1_min + para_calc_f1_max) / 2
        ls_para_calc_f1[i] = para_calc_f1
        ls_para_f1_min[i] = para_calc_f1_min
        ls_para_f1_max[i] = para_calc_f1_max

    # --------------------------------------------------------------------------
    # boltzmann sigmoid at T_calc for f1
    ph_range = np.linspace(start=0, stop=14, num=int(14 / 0.1 + 1))

    ls_cot_dPhi_min = {}
    ls_cot_dPhi_max = {}
    ls_dPhi_min = {}
    ls_dPhi_max = {}
    ls_pH = {}
    for i in range(len(y_meas)):
        cot_dPhi_min = boltzmann_sigmoid(top=ls_para_f1_min[i]['top'], bottom=ls_para_f1_min[i]['bottom'],
                                     slope=ls_para_f1_min[i]['slope'], pka=ls_para_f1_min[i]['pka'], pH=ph_range)
        cot_dPhi_max = boltzmann_sigmoid(top=ls_para_f1_max[i]['top'], bottom=ls_para_f1_max[i]['bottom'],
                                     slope=ls_para_f1_max[i]['slope'], pka=ls_para_f1_max[i]['pka'], pH=ph_range)
        dPhi_min = np.rad2deg(af.arccot(cot_dPhi_min))
        dPhi_max = np.rad2deg(af.arccot(cot_dPhi_max))

        # find closest value for cot(dPhi) or dPhi (superimposed phase angle)
        [dphi1_min, dphi1_max, idx_dphi1_min,
         idx_dphi1_max] = af.find_closest_in_list(list_=dPhi_min, value=Phi_f1_deg_er['meas {}'.format(i)][1])
        [dphi2_min, dphi2_max, idx_dphi2_min,
         idx_dphi2_max] = af.find_closest_in_list(list_=dPhi_max, value=Phi_f1_deg_er['meas {}'.format(i)][1])

        pH_calc_all = [ph_range[idx_dphi1_min], ph_range[idx_dphi1_max], ph_range[idx_dphi2_min],
                       ph_range[idx_dphi2_max]]
        pH_calc = [min(pH_calc_all), sum(pH_calc_all) / np.float(len(pH_calc_all)), max(pH_calc_all)]
        print('Calculated pH: {:.2f} ± {:.2f} at T = {:.2f}°C'.format(pH_calc[1], (pH_calc[2] - pH_calc[0]) / 2, ls_T[i][1]))

        # store in dictionary
        ls_cot_dPhi_min[i] = cot_dPhi_min
        ls_cot_dPhi_max[i] = cot_dPhi_max
        ls_dPhi_min[i] = dPhi_min
        ls_dPhi_max[i] = dPhi_max
        ls_pH[i] = pH_calc

    # --------------------------------------------------------------------------
    # combining to Series for common storage and export
    parameter = pd.Series({'linear regression T': y_reg_ph1, 'tauP_calc': ls_tauP,
                           'cotPhi_min': ls_cot_dPhi_min, 'cotPhi_max': ls_cot_dPhi_max,
                           'dPhi_min': ls_dPhi_min, 'dPhi_max': ls_dPhi_max, 'parameter_pH_f1': ls_para_calc_f1})

# ---------------------------------------------------------------------------------------------------------------------
    if plotting is True:
        f, (ax_temp, ax_pH) = plt.subplots(ncols=2, figsize=(10,3))

        # T evaluation
        ax_temp.plot(temp_range_deg, y_reg_ph1['min'], color='navy', lw=1.)
        ax_temp.plot(temp_range_deg, y_reg_ph1['max'], color='navy', lw=1.)
        ax_temp.fill_between(temp_range_deg, y_reg_ph1['min'], y_reg_ph1['max'], color='grey', alpha=0.1, lw=0.25)
        ax_temp.legend(['T = {:.2f} ± {:.2f} °C'.format(temp_calc[1], (temp_calc[2]-temp_calc[0])/2)], loc=0,
                       frameon=True, framealpha=0.5, fancybox=True, fontsize=fontsize_*0.8)

        # reticle for evaluated temperature
        ax_temp.axhline(tau_phos[0], color='black', lw=0.5, ls='--')
        ax_temp.axhline(tau_phos[2], color='black', lw=0.5, ls='--')
        ax_temp.fill_between(temp_range_deg, tau_phos[0], tau_phos[2], color='grey', alpha=0.1, lw=0.25)

        ax_temp.axvline(temp_calc[0], color='black', lw=0.5, ls='--')
        ax_temp.axvline(temp_calc[2], color='black', lw=0.5, ls='--')
        ax_temp.axvspan(temp_calc[0], temp_calc[2], color='grey', alpha=0.2)

        # layout
        ax_temp.set_xlabel('Temperature [°C]', fontsize=fontsize_)
        ax_temp.set_ylabel('tauP [µs]', fontsize=fontsize_)

        # -----------------------------------------------------------
        # pH evaluation
        ax1 = ax_pH.twinx()
        ax_pH.plot(ph_range, cot_dPhi_min, color='navy', lw=1.)
        ax_pH.plot(ph_range, cot_dPhi_max, color='navy', lw=1.)
        ax_pH.legend(['pH = {:.2f} ± {:.2f}'.format(pH_calc[1], (pH_calc[2]-pH_calc[0])/2)], loc='upper center',
                     bbox_to_anchor=(0.79, 0.88), frameon=True, framealpha=0.5, fancybox=True, fontsize=fontsize_*0.8)

        ax1.plot(ph_range, dPhi_min, color='forestgreen', lw=1.)
        ax1.plot(ph_range, dPhi_max, color='forestgreen', lw=1.)
        ax_pH.fill_between(ph_range, cot_dPhi_min, cot_dPhi_max, color='grey', alpha=0.1, lw=0.25)

        # reticle for evaluated pH
        ax1.axhline(Phi_f1_deg_er['meas'][0], color='black', lw=0.5, ls='--')
        ax1.axhline(Phi_f1_deg_er['meas'][2], color='black', lw=0.5, ls='--')
        ax1.fill_between(ph_range, Phi_f1_deg_er['meas'][0], Phi_f1_deg_er['meas'][2], color='grey', alpha=0.1, lw=0.25)

        ax1.axvline(pH_calc[0], color='black', lw=0.5, ls='--')
        ax1.axvline(pH_calc[2], color='black', lw=0.5, ls='--')
        ax1.axvspan(pH_calc[0], pH_calc[2], color='grey', alpha=0.2)

        # layout
        ax_pH.set_xlabel('pH', fontsize=13)
        ax1.set_ylabel('dPhi [°]', color='forestgreen', fontsize=fontsize_)
        ax_pH.set_ylabel('cot(dPhi)', color='navy', fontsize=fontsize_)

    return pd.Series(ls_T), pd.Series(ls_pH), parameter
