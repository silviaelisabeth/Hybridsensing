__author__ = 'szieger'
__project__ = 'dualsensor T/O2 sensing'

import matplotlib
import additional_functions as af
import Dualsensing_T_O2 as multi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
from datetime import date
import seaborn as sns

sns.set_context('paper', 1.5)
sns.set_style('ticks')
# sns.set(style="whitegrid")
sns.set_palette('Set1')

conv_temp = 273.15


# --------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------
def read_data(p, start_data='hh:mm:ss', start_header='Messdaten', end_header='Uhrzeit', usecols=None, index_='Zeit'):
    lines = []
    header = []

    # load data
    with open(p, 'r') as f:
        for line in f:
            if start_data in line:
                for line in f: # now you are at the lines you want
                    lines.append(line.replace(',', '.')[:-1].split('\t'))
    df = pd.DataFrame(lines)

    # load header
    header = []
    with open(p, 'rt') as f:
        for line in f:
            if start_header in line:
                for line in f:
                    if line.startswith(end_header):
                        header.append(line.replace(',', '.')[:-1].split('\t'))
    header = header[0]

    # combine data and header
    df.columns = header

    if len(usecols) == 0:
        ddf = df
    else:
        l = []
        h = []
        for i in usecols:
            l.append(df[df.columns[i]].values)
            h.append(df.columns[i])

        ddf = pd.DataFrame(l, index=h).T
    ddf = ddf.convert_objects(convert_numeric=True)
    ddf = ddf.set_index(index_)
    if index_ == 'Uhrzeit':
        ddf.index = pd.DatetimeIndex(ddf.index)
    else:
        ddf.index = [np.float(d) for d in ddf.index]

    return ddf


def convert_austrian_timestamp_LockIn(p, df):
    l = []
    with open(p, 'rt') as f:
        for line in f:
            if 'Anmerkung' in line:
                for line in f:
                    if line.startswith('Datum'):
                        l.append(line.replace(',', '.')[:-1].split('\t'))
    l = l[0][1]
    if 'Jänner' in l:
        l = l.replace('Jänner', '01').replace('.', ' ')
    elif 'Februar' in l:
        l = l.replace('Februar', '02').replace('.', ' ')
    elif 'März' in l:
        l = l.replace('März', '03').replace('.', ' ')
    elif 'April' in l:
        l = l.replace('April', '04').replace('.', ' ')
    elif 'Mai' in l:
        l = l.replace('Mai', '05').replace('.', ' ')
    elif 'Juni' in l:
        l = l.replace('Juni', '06').replace('.', ' ')
    elif 'Juli' in l:
        l = l.replace('Juli', '07').replace('.', ' ')
    elif 'August' in l:
        l = l.replace('August', '08').replace('.', ' ')
    elif 'September' in l:
        l = l.replace('September', '09').replace('.', ' ')
    elif 'Oktober' in l:
        l = l.replace('Oktober', '10').replace('.', ' ')
    elif 'November' in l:
        l = l.replace('November', '11').replace('.', ' ')
    elif 'Dezember' in l:
        l = l.replace('Dezember', '12').replace('.', ' ')
    date_lockin = datetime.datetime.strptime(l, '%d %m %Y')

    df.index = df.index.map(lambda t: t.replace(year=date_lockin.date().year, month=date_lockin.date().month,
                                                day=date_lockin.date().day))

    return df


def write_txt_to_df(p):
    content = []
    with open(p, 'r') as f:
        for curline in f:
            if 'Date' in curline:
                for line in f:
                    content.append(line.split('\t')[:-1])

    head = pd.DataFrame(content[:2], dtype=np.float64)
    data = pd.DataFrame(content[2:], dtype=np.float64)

    header = []
    for i in range(len(head.columns)-1):
        header.append(' '.join(list(head[i].values)))
    data.columns = header

    return data


def prepare_rawdata_lockIn(f1_, f2_, ddphi1_, ddphi2_, p1, p2, fontsize_=11, saving=False, index_='Zeit', header=None,
                           plotting=True, usecols=None):
    if header is None:
        header = ['Amplitude [V]', 'Phi [deg]', 'N2 [mL/min]', 'O2 [mL/min]']
    if usecols is None:
        usecols = [1, 2, 4, 6, 8]

    f1 = np.float64(f1_.replace(',', '.'))
    f2 = np.float64(f2_.replace(',', '.'))

    ddphi_f1 = np.float64(ddphi1_.replace(',', '.'))
    ddphi_f2 = np.float64(ddphi2_.replace(',', '.'))

    # pre-check if the files were measured at the same temperature
    if p1.split('/')[-1].split('_')[-2] == p2.split('/')[-1].split('_')[-2]:
        pass
    else:
        raise ValueError('The samples ought to be measured at the same temperature!')

    df1 = read_data(p=p1, usecols=usecols, index_=index_)
    df2 = read_data(p=p2, usecols=usecols, index_=index_)

    df1['Theta'] = np.abs(df1['Theta'] + ddphi_f1)
    df2['Theta'] = np.abs(df2['Theta'] + ddphi_f2)

    df1.columns = header # ['Amplitude [V]', 'Phi [deg]', 'N2 [mL/min]', 'O2 [mL/min]']
    df2.columns = header # ['Amplitude [V]', 'Phi [deg]', 'N2 [mL/min]', 'O2 [mL/min]']

    time_min2 = df2.index / 60
    time_min1 = df1.index / 60

    # PLOTTING
    plt.ioff()
    # f1
    fig1, ax = plt.subplots(nrows=2, figsize=(5, 4), sharex=True)

    majorLocatorx = MultipleLocator(20)
    minorLocatorx = MultipleLocator(10)

    # phase angle and amplitude
    ax1 = ax[0].twinx()
    ax[0].plot(time_min1, df1['Amplitude [V]']*1000, color='navy', lw=1.)
    ax1.plot(time_min1, df1['Phi [deg]'], color='darkorange', lw=1.)

    ax1.tick_params(which='xaxis', direction='out', top='on', right='on', labelsize=8)
    ax1.xaxis.set_major_locator(majorLocatorx)
    ax1.xaxis.set_minor_locator(minorLocatorx)

    ax[1].set_xlabel('Time [min]', fontsize=fontsize_)
    ax[0].set_ylabel('Amplitude [mV]', color='navy', fontsize=fontsize_)
    ax1.set_ylabel('phase angle [deg]', color='darkorange',fontsize=fontsize_)

    # -------------------------------------------
    # flow rate nitrogen and oxygen
    ax2 = ax[1].twinx()
    ax[1].plot(time_min1, df1['N2 [mL/min]'], color='k', ls='--', lw=1.)
    ax2.plot(time_min1, df1['O2 [mL/min]'], color='grey', lw=1.)

    ax[1].set_ylabel('flow rate N$_2$ [mL min$^{-1}$]', fontsize=fontsize_)
    ax2.set_ylabel('flow rate air [mL min$^{-1}$]', color='grey', fontsize=fontsize_)
    ax[0].set_title('Modulation frequency f = {:.2f}Hz'.format(f1), fontsize=fontsize_*1.1)

    plt.tight_layout()

    # ---------------------------------------------------------------------------
    # f2
    fig2, ax = plt.subplots(nrows=2, figsize=(5,4), sharex=True)

    # phase angle and amplitude
    ax1 = ax[0].twinx()
    ax[0].plot(time_min2, df2['Amplitude [V]']*1000, color='navy', lw=1.)
    ax1.plot(time_min2, df2['Phi [deg]'], color='darkorange', lw=1.)

    ax[1].set_xlabel('Time [min]', fontsize=fontsize_)
    ax[0].set_ylabel('Amplitude [mV]', color='navy', fontsize=fontsize_)
    ax1.set_ylabel('phase angle [deg]', color='darkorange', fontsize=fontsize_)

    # -------------------------------------------
    # flow rate nitrogen and oxygen
    ax2 = ax[1].twinx()
    ax[1].plot(time_min2, df2['N2 [mL/min]'], color='k', ls='--', lw=1.)
    ax2.plot(time_min2, df2['O2 [mL/min]'], color='grey', lw=1.)
    ax[1].set_ylabel('flow rate N$_2$ [mL min$^{-1}$]', fontsize=fontsize_)
    ax2.set_ylabel('flow rate air [mL min$^{-1}$]', color='grey', fontsize=fontsize_)
    ax[0].set_title('Modulation frequency f = {:.2f}Hz'.format(f2), fontsize=fontsize_)
    plt.tight_layout()

    if plotting is False:
        plt.close(fig1)
        plt.close(fig2)
    plt.show()
    # ---------------------------------------------------------------------------
    # Saving
    if saving is True:
        # path = '/'.join(p1.split('/')[:-1]) + '/'
        path_fig = '/'.join(p1.split('/')[:-1]) + '/txt/Graphs/'
        directory = os.path.dirname(path_fig)

        if not os.path.exists(directory):
            os.makedirs(directory)
        saving_name_f1 = path_fig + p1.split('/')[-1] + '.png'
        saving_name_f2 = path_fig + p2.split('/')[-1] + '.png'
        fig1.savefig(saving_name_f1, dpi=300)
        fig2.savefig(saving_name_f2, dpi=300)

        path_txt = '/'.join(p1.split('/')[:-1]) + '/txt/'
        directory_txt = os.path.dirname(path_txt)
        if not os.path.exists(directory_txt):
            os.makedirs(directory_txt)

        saving_name1_f1 = path_txt + p1.split('/')[-1] + '.txt'
        saving_name1_f2 = path_txt + p2.split('/')[-1] + '.txt'
        df1.to_csv(saving_name1_f1, decimal='.', sep='\t')
        df2.to_csv(saving_name1_f2, decimal='.', sep='\t')
        print(saving_name_f1, 'saved')

    return df1, df2


def lifetime_intensity_ratio_lockIn(file, fontsize_=12, num_freq=2, zoom_in=True, saving=False, plotting=True):
    print(file.split('/')[-1])
    data = pd.read_csv(file, sep='\t', skiprows=num_freq, index_col=0, usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11]).dropna() #
    skip_ = len(pd.read_csv(file, sep='\t', header=None, usecols=[1])) - 2
    p = pd.read_csv(file, sep='\t', header=None, usecols=[1])[1].values
    freq = [np.float(i) for i in p[:len(p) - skip_]]
    temp = file.split('/')[-1].split('_')[-2]

    # store measurement data
    meas_data = pd.read_csv(file, sep='\t', skiprows=2, index_col=0, usecols=[3, 4, 6, 8, 10]).dropna()

    # calculate lifetime
    x1, x2 = af.lifetime(phi1=data['Phi(f1) [deg]'].values, phi2=data['Phi(f2) [deg]'].values, f1=freq[0], f2=freq[1])
    tauP = af.tau_selection(tau1=x1, tau2=x2)
    df_tauP = pd.DataFrame(tauP, index=data.index, columns=[temp])

    # calculate intensity ratio
    int_ratio11 = af.intensity_ratio(f=freq[0], tau=tauP, phi=data['Phi(f1) [deg]'].values)
    int_ratio12 = af.intensity_ratio(f=freq[1], tau=tauP, phi=data['Phi(f2) [deg]'].values)
    int_ratio1 = pd.concat([pd.DataFrame(int_ratio11), pd.DataFrame(int_ratio12)], axis=1).mean(axis=1)
    int_ratio1_std = pd.concat([pd.DataFrame(int_ratio11), pd.DataFrame(int_ratio12)], axis=1).std(axis=1)
    df_int_ratio = pd.DataFrame(int_ratio1.values, index=data.index, columns=[temp])

    # --------------------------------------------------------------------------------------------------------
    # plotting
    plt.ioff()
    fig3, ((ax2, ax3), (ax4, ax5)) = plt.subplots(figsize=(5, 4), nrows=2, ncols=2, sharex=True)

    # dphi
    ax2.plot(data.index, data['Phi(f1) [deg]'], lw=1., color='navy')
    ax2.plot(data.index, data['Phi(f2) [deg]'], lw=1., color='darkorange')
    ax2.tick_params(which='both', direction='in', top='on', right='on')

    # amplitude
    ax3.plot(data.index, data['A(f1) [V]']*1000, lw=1., color='navy')
    ax3.plot(data.index, data['A(f2) [V]']*1000, lw=1., color='darkorange')
    ax3.tick_params(which='both', direction='in', top='on', right='on')

    # ------------------------------------------------------
    # lifetime
    ax4.plot(data.index, tauP*1000, lw=1., color='k')
    ax4.tick_params(which='both', direction='in', top='on', right='on')

    # intensity ratio
    ax5.plot(data.index, int_ratio1, lw=1., color='k')
    ax5.tick_params(which='both', direction='in', top='on', right='on')

    # ------------------------------------------------------
    # legend
    ax3.legend(['f1 = {:.2f}Hz'.format(freq[0]), 'f2 = {:.2f}Hz'.format(freq[1])], loc='upper center',
              bbox_to_anchor=(0.55,1), frameon=True, fancybox=True, handlelength=1)

    # layout
    ax4.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
    ax5.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
    ax2.set_ylabel('Phase angle [deg]', fontsize=fontsize_)
    ax3.set_ylabel('Amplitude [mV]', fontsize=fontsize_)
    ax4.set_ylabel('Lifetime [ms]', fontsize=fontsize_)
    ax5.set_ylabel('Intensity ratio', fontsize=fontsize_)

    plt.tight_layout()

    # --------------------------------------------------
    if zoom_in is True:
        fig4, ax6 = plt.subplots(figsize=(3,2.5))

        majorLocatorx = MultipleLocator(50)
        minorLocatorx = MultipleLocator(10)
        minorLocatory = MultipleLocator(0.1)

        # lifetime
        ax6.plot(data.index, tauP[0] / tauP, color='k', lw=1.)

        # ------------------------------------------------------
        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        ax7 = plt.axes([0, 0, 3, 200])

        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax6, [0.4,0.18,0.5,0.5])
        ax7.set_axes_locator(ip)

        # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
        # mark_inset(ax6, ax7, loc1=2, loc2=4, fc="none", ec='0.5')
        ax7.plot(data.index[:5], tauP[0] / tauP[:5], color='navy', lw=1.)

        # ------------------------------------------------------
        # layout
        ax6.xaxis.set_major_locator(majorLocatorx)
        ax6.xaxis.set_minor_locator(minorLocatorx)
        ax6.yaxis.set_minor_locator(minorLocatory)
        ax6.tick_params(which='both', direction='in', top='on', right='on')

        ax7.set_xlim(-5, 40)
        ax7.tick_params(which='both', direction='in', top='on', right='on', labelsize=fontsize_*0.8)

        ax6.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
        ax6.set_ylabel('tau$_0$ / tau', fontsize=fontsize_)

        plt.tight_layout()

        # ---------------------------------------------------------------------------------
        fig5, ax8 = plt.subplots(figsize=(3,2.5))

        majorLocatorx = MultipleLocator(50)
        minorLocatorx = MultipleLocator(10)
        minorLocatory = MultipleLocator(0.1)

        # lifetime
        ax8.plot(data.index, int_ratio1, color='k', lw=1.)

        # ------------------------------------------------------
        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        ax9 = plt.axes([0, 0, 1, 1])

        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax8, [0.4,0.12,0.5,0.5])
        ax9.set_axes_locator(ip)

        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.
        ax9.plot(data.index[:5], tauP[0] / tauP[:5], color='navy', lw=1.)

        # ------------------------------------------------------
        # layout
        ax8.xaxis.set_major_locator(majorLocatorx)
        ax8.xaxis.set_minor_locator(minorLocatorx)
        ax8.yaxis.set_minor_locator(minorLocatory)
        ax8.tick_params(which='both', direction='in', top='on', right='on', labelsize=fontsize_)
        ax9.set_xlim(-5, 40)
        ax9.tick_params(which='both', direction='in', top='on', right='on', labelsize=fontsize_*0.8)

        ax8.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
        ax8.set_ylabel('Intensity ratio', fontsize=fontsize_)

        plt.tight_layout()
    else:
        fig4, ax6 = plt.subplots(figsize=(3,2.5))

        majorLocatorx = MultipleLocator(50)
        minorLocatorx = MultipleLocator(10)
        minorLocatory = MultipleLocator(0.1)

        # lifetime
        ax6.plot(data.index, tauP[0] / tauP, color='k', lw=1.)

        # layout
        ax6.xaxis.set_major_locator(majorLocatorx)
        ax6.xaxis.set_minor_locator(minorLocatorx)
        ax6.yaxis.set_minor_locator(minorLocatory)
        ax6.tick_params(which='both', direction='in', top='on', right='on')
        ax6.set_xlabel('Partial pressure [hPa]', fontsize=fontsize_)
        ax6.set_ylabel('tau$_0$ / tau', fontsize=fontsize_)

        plt.tight_layout()

    if plotting is True:
        pass
    else:
        plt.close(fig3)
        plt.close(fig4)
        if zoom_in is True:
            plt.close(fig5)
    plt.show()
    # ---------------------------------------------------------------------------
    # Saving
    if saving is True:
        path = '/'.join(file.split('/')[:-1]) + '/Graphs/'
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        saving_name = path + file.split('/')[-1].split('.')[0] + '_all.png'
        fig3.savefig(saving_name, dpi=300)

        if zoom_in is True:
            saving_name_fig4 = path + file.split('/')[-1].split('.')[0] + '_zoomIn_SternVolmer.png'
            saving_name_fig5 = path + file.split('/')[-1].split('.')[0] + '_zoomIn_intensityRatio.png'

            fig4.savefig(saving_name_fig4, dpi=300)
            fig5.savefig(saving_name_fig5, dpi=300)
        else:
            saving_name_fig4 = path + file.split('/')[-1].split('.')[0] + '_SternVolmer.png'
            fig4.savefig(saving_name_fig4, dpi=300)

    return df_tauP, df_int_ratio, meas_data


def generate_matrix_for_3dPlot(temp_slit, df_full, label_ox, label_temp, label_dPhi):
    l = []
    for i in range(len(temp_slit)-1):
        ind = df_full[label_ox].iloc[temp_slit[i]:temp_slit[i+1]].values
        col = df_full[label_temp].iloc[temp_slit[i]:temp_slit[i+1]].values[0]
        l.append(pd.DataFrame(df_full[label_dPhi].iloc[temp_slit[i]:temp_slit[i+1]].values, index=ind,
                              columns=[col]))

    ind = df_full[label_ox].iloc[temp_slit[-1]:].values
    col = df_full[label_temp].iloc[temp_slit[-1]:].values[0]
    l.append(pd.DataFrame(df_full[label_dPhi].iloc[temp_slit[-1]:].values, index=ind, columns=[col]))

    return l


def plot_3dsubplot_T_O2(temp, temp_slit, df_full, colors, ax, zlabel='Phi(f1) [deg]', xticks_major=2,
                        xlabel='pO2 (firesting) [hPa]', ylabel='Temperatur', fontsize=8, labelpad=4, tickspad=0,
                        yticks_major=50, zticks_major=10):
    if zlabel == 'tauP0 / tauP':
            df_full['tauP0 / tauP'] = df_full['tauP [ms]'].copy()
    if zlabel == '1/I-ratio':
            df_full['1/I-ratio'] = df_full['I-ratio'].copy()
    if zlabel == 'I0 / I':
            df_full['I0 / I'] = df_full['I-ratio'].copy()
    if zlabel == 'I / I0':
            df_full['I / I0'] = df_full['I-ratio'].copy()

    tauP0 = df_full['tauP [ms]'].iloc[np.flatnonzero([df_full[xlabel] == 0.])].values
    Iratio0 = df_full['I-ratio'].iloc[np.flatnonzero([df_full[xlabel] == 0.])].values

    for i in range(len(temp_slit)):
        slit0 = temp_slit[i]
        if i != len(temp_slit)-1:
            slit1 = temp_slit[i+1]
            if zlabel == 'tauP0 / tauP':
                df_full['tauP0 / tauP'].iloc[slit0:slit1] = tauP0[i] / df_full['tauP [ms]'].iloc[slit0:slit1].dropna()
            if zlabel == '1/I-ratio':
                df_full['1/I-ratio'].iloc[slit0:slit1] = 1. / df_full['I-ratio'].iloc[slit0:slit1].dropna()
            if zlabel == 'I0 / I':
                df_full['I0 / I'].iloc[slit0:slit1] = Iratio0[i] / df_full['I-ratio'].iloc[slit0:slit1].dropna()
            if zlabel == 'I / I0':
                df_full['I / I0'].iloc[slit0:slit1] = df_full['I-ratio'].iloc[slit0:slit1].dropna() / Iratio0[i]
            if isinstance(colors, str):
                ax.scatter(xs=temp[i], ys=df_full[xlabel].iloc[slit0:slit1],
                           zs=df_full[zlabel].iloc[slit0:slit1], c=[colors])
            else:
                ax.scatter(xs=temp[i], ys=df_full[xlabel].iloc[slit0:slit1],
                           zs=df_full[zlabel].iloc[slit0:slit1], c=colors[i])
        else:
            if zlabel == 'tauP0 / tauP':
                df_full['tauP0 / tauP'].iloc[slit0:] = tauP0[i] / df_full['tauP [ms]'].iloc[slit0:].dropna()
            if zlabel == '1/I-ratio':
                df_full['1/I-ratio'].iloc[slit0:] = 1. / df_full['I-ratio'].iloc[slit0:].dropna()
            if zlabel == 'I0 / I':
                df_full['I0 / I'].iloc[slit0:] = Iratio0[i] / df_full['I-ratio'].iloc[slit0:].dropna()
            if zlabel == 'I / I0':
                df_full['I / I0'].iloc[slit0:] = df_full['I-ratio'].iloc[slit0:].dropna() / Iratio0[i]
            if isinstance(colors, str):
                ax.scatter(xs=temp[i], ys=df_full[xlabel].iloc[slit0:], zs=df_full[zlabel].iloc[slit0:],
                           c=[colors])
            else:
                ax.scatter(xs=temp[i], ys=df_full[xlabel].iloc[slit0:], zs=df_full[zlabel].iloc[slit0:],
                           c=colors[i])

    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=tickspad)
    ax.set_xlabel(ylabel, labelpad=labelpad, fontsize=fontsize)
    ax.set_ylabel(xlabel, labelpad=labelpad, fontsize=fontsize)
    ax.set_zlabel(zlabel, labelpad=labelpad, fontsize=fontsize)

    if xticks_major is None:
        pass
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xticks_major))
    if yticks_major is None:
        pass
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(yticks_major))
    if zticks_major is None:
        pass
    else:
        ax.zaxis.set_major_locator(plt.MultipleLocator(zticks_major))

    plt.tight_layout(w_pad=1.5)
    plt.show()

    return df_full, ax


def plot_3dsubplot_T_O2_continuous(df_full, colors, ax, limits, zlabel='Phi(f1) [deg]', fontsize=8, tau0=None, I0=None,
                                   tickspad=0, ylabel='pO2 (firesting) [hPa]', xlabel='Temperatur', labelpad=4):

    if zlabel == 'tauP0 / tauP':
            df_full['tauP0 / tauP'] = df_full['tauP [ms]'].copy()
    if zlabel == '1/I-ratio':
            df_full['1/I-ratio'] = df_full['I-ratio'].copy()
    if zlabel == 'I0 / I':
            df_full['I0 / I'] = df_full['I-ratio'].copy()
    if zlabel == 'I / I0':
            df_full['I / I0'] = df_full['I-ratio'].copy()

    if 'Time' in ylabel:
        tauP0 = tau0
        Iratio0 = I0
    else:
        tauP0 = df_full[df_full[ylabel] <= 0.01]['tauP [ms]'].mean()
        Iratio0 = df_full[df_full[ylabel] <= 0.01]['I-ratio'].mean()

    if zlabel == 'tauP0 / tauP':
        df_full['tauP0 / tauP'] = tauP0 / df_full['tauP [ms]']
    if zlabel == '1/I-ratio':
        df_full['1/I-ratio'] = 1. / df_full['I-ratio']
    if zlabel == 'I0 / I':
        df_full['I0 / I'] = Iratio0 / df_full['I-ratio']
    if zlabel == 'I / I0':
        df_full['I / I0'] = df_full['I-ratio'] / Iratio0

    z = df_full[zlabel]
    y = df_full.loc[z.index, ylabel]

    for i, c in enumerate(df_full.columns):
        if 'Temp' in c:
            xlabel = c
    x = df_full.loc[z.index, xlabel]

    if isinstance(colors, str):
        ax.scatter(xs=x, ys=y, zs=z, c=[colors], marker='o', s=10, edgecolors='lightgrey', linewidth=0.2)
    else:
        ax.scatter(xs=x, ys=y, zs=z, c=colors[0], marker='o', s=10, edgecolors='lightgrey', linewidth=0.2)

    if limits is None:
        pass
    else:
        ax.set_xlim(limits['xmin'], limits['xmax'])
        ax.set_ylim(limits['ymin'], limits['ymax'])
        ax.set_zlim(limits['zmin'], limits['zmax'])
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=tickspad)

    if limits is None:
        pass
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(limits['T_major']))
        ax.yaxis.set_major_locator(plt.MultipleLocator(limits['pO2_major']))
        ax.zaxis.set_major_locator(plt.MultipleLocator(limits['z_major']))

    if 'Temp' in xlabel:
        xlabel_ = 'Temperature [°C]'
    else:
        ylabel_ = ylabel
    if 'pO2' in ylabel:
        ylabel_ = 'pO2 [hPa]'
    else:
        ylabel_ = ylabel
    ax.set_xlabel(xlabel_, labelpad=labelpad, fontsize=fontsize)
    ax.set_ylabel(ylabel_, labelpad=labelpad, fontsize=fontsize)
    ax.set_zlabel(zlabel, labelpad=labelpad, fontsize=fontsize)

    plt.tight_layout(w_pad=.5)
    plt.show()

    return df_full, ax


def span_area_between_data_points(lifetime_calc, intensity_calc, df_full, temp_slit, temp, mesh_tau=50, mesh_int=50,
                                  labelpad=2.5, tickspad=0, zticks_major=None, fontsize=11, ylabel='Temperatur',
                                  xlabel='pO2 (firesting) [hPa]', path=None, z_tau1='tauP [ms]', z_tau2='tauP0 / tauP',
                                  z_int1='I-ratio', z_int2='I0 / I', z_int3='I / I0',  save_=False, xlim=None,
                                  ylim=None, zlim=None):
    # lifetime tauP
    Z_tau = []
    Z_tau0_tauP = []
    for i in range(len(temp_slit)):
        slit0 = temp_slit[i]
        if i != len(temp_slit)-1:
            slit1 = temp_slit[i+1]
            df = df_full[z_tau1].iloc[slit0:slit1]
            df1 = df_full[z_tau2].iloc[slit0:slit1]
            df.reset_index(drop=True, inplace=True)
            df1.reset_index(drop=True, inplace=True)
            Z_tau.append(df)
            Z_tau0_tauP.append(df1)
        else:
            df = df_full[z_tau1].iloc[slit0:]
            df1 = df_full[z_tau2].iloc[slit0:]
            df.reset_index(drop=True, inplace=True)
            df1.reset_index(drop=True, inplace=True)
            Z_tau.append(df)
            Z_tau0_tauP.append(df1)
    Z_tau = pd.concat(Z_tau, axis=1, ignore_index=True).dropna()
    Z_tau0_tauP = pd.concat(Z_tau0_tauP, axis=1, ignore_index=True).dropna()
    print(Z_tau)
    # -------------------------------------------
    # intensity ratio
    Z_iratio = []
    Z_int0_int = []
    Z_int_int0 = []

    for i in range(len(temp_slit)):
        slit0 = temp_slit[i]
        if i != len(temp_slit)-1:
            slit1 = temp_slit[i+1]
            df2 = df_full[z_int1].iloc[slit0:slit1]
            df3 = df_full[z_int2].iloc[slit0:slit1]
            df4 = df_full[z_int3].iloc[slit0:slit1]
            df2.reset_index(drop=True, inplace=True)
            df3.reset_index(drop=True, inplace=True)
            df4.reset_index(drop=True, inplace=True)
            Z_iratio.append(df2)
            Z_int0_int.append(df3)
            Z_int_int0.append(df4)
        else:
            df2 = df_full[z_int1].iloc[slit0:]
            df3 = df_full[z_int2].iloc[slit0:]
            df4 = df_full[z_int3].iloc[slit0:]
            df2.reset_index(drop=True, inplace=True)
            df3.reset_index(drop=True, inplace=True)
            df4.reset_index(drop=True, inplace=True)
            Z_iratio.append(df2)
            Z_int0_int.append(df3)
            Z_int_int0.append(df4)
    Z_iratio = pd.concat(Z_iratio, axis=1, ignore_index=True).dropna()
    Z_int0_int = pd.concat(Z_int0_int, axis=1, ignore_index=True).dropna()
    Z_int_int0 = pd.concat(Z_int_int0, axis=1, ignore_index=True).dropna()

    # ========================================================================================
    # Plotting
    fig = plt.figure(figsize=plt.figaspect(0.9))
    ax0 = fig.add_subplot(2, 1, 1, projection='3d')

    temp_x = temp
    pO2_y = df_full[xlabel].iloc[0:8].values
    temp_X, pO2_Y = np.meshgrid(temp_x, pO2_y)

    # lifetime
    if len(lifetime_calc) == 0.:
        lifetime_calc = '1'
    if len(intensity_calc) == 0.:
        intensity_calc = '1'
    if lifetime_calc == '1' or lifetime_calc == 'tauP' or lifetime_calc == 'tauP in [ms]' or lifetime_calc == '(1)':
        time = 'tauP'
        print(temp_X)
        ax0.contour3D(temp_X, pO2_Y, Z_tau, mesh_tau, cmap='viridis')
        df1, ax1 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax0, labelpad=-5,
                                       zlabel=z_tau1, colors='k', xlabel=xlabel, ylabel=ylabel, xticks_major=2,
                                       yticks_major=50, zticks_major=zticks_major)
        ax0.set_zlabel('tauP [ms]', labelpad=labelpad, fontsize=fontsize)
    elif lifetime_calc == '2' or lifetime_calc == '(2)' or lifetime_calc == 'tau0 / tauP' or lifetime_calc == 'tau0/tauP':
        time = 'tau_quot'
        ax0.contour3D(temp_X, pO2_Y, Z_tau0_tauP, mesh_tau, cmap='viridis')
        df1, ax1 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax0, labelpad=-5,
                                       zlabel=z_tau2, xlabel=xlabel, ylabel=ylabel, colors='k', xticks_major=2,
                                       yticks_major=50, zticks_major=zticks_major)
        ax0.set_zlabel('tau0 / tauP', labelpad=labelpad, fontsize=fontsize)
    else:
        raise ValueError('Initial error - what kind of lifetime is required for plotting? --> (1) or (2)')

    ax0.tick_params(axis='both', which='major', labelsize=fontsize, pad=tickspad)
    ax0.set_ylabel('$pO_2$ [hPa]', labelpad=labelpad, fontsize=fontsize)
    ax0.set_xlabel('Temperature [°C]', labelpad=labelpad, fontsize=fontsize)
    if xlim is not None:
        ax0.set_xlim(xlim[0], xlim[1])
        ax1.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax0.set_ylim(ylim[0], ylim[1])
        ax1.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax0.set_zlim(zlim[0], zlim[1])
        ax1.set_zlim(zlim[0], zlim[1])

    # ---------------------------------------------------
    # Intensity
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    if intensity_calc == '1' or intensity_calc == 'I-ratio' or intensity_calc == 'i-ratio' or intensity_calc == '(1)':
        intensity = 'I-ratio'
        ax2.contour3D(temp_X, pO2_Y, Z_iratio, mesh_int, cmap='inferno')
        df1, ax3 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax2, labelpad=-5,
                                       zlabel=z_int1, xlabel=xlabel, ylabel=ylabel, colors='k', xticks_major=2,
                                       yticks_major=50, zticks_major=zticks_major)
        ax2.set_zlabel('I-ratio', labelpad=labelpad, fontsize=fontsize)
    elif intensity_calc == '2' or intensity_calc == '(2)' or intensity_calc == 'I0 / I-ratio' or intensity_calc == 'I0/I':
        intensity = 'I0 / I'
        ax2.contour3D(temp_X, pO2_Y, Z_int0_int, mesh_int, cmap='inferno')
        df1, ax3 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax2, labelpad=-5,
                                       zlabel=z_int2, xlabel=xlabel, ylabel=ylabel, colors='k', xticks_major=2,
                                       yticks_major=50, zticks_major=zticks_major)
        ax2.set_zlabel('I0 /I-ratio', labelpad=labelpad, fontsize=fontsize)
    elif intensity_calc == '3' or intensity_calc == '(3)' or intensity_calc == 'I / I0' or intensity_calc == 'I/I0':
        intensity = 'I / I0'
        ax2.contour3D(temp_X, pO2_Y, Z_int_int0, mesh_int, cmap='inferno')
        df1, ax3 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax2, labelpad=-5,
                                       zlabel=z_int3, xlabel=xlabel, ylabel=ylabel, colors='k', xticks_major=2,
                                       yticks_major=50, zticks_major=zticks_major)
        ax2.set_zlabel('I-ratio /I0', labelpad=labelpad, fontsize=fontsize)
    else:
        raise ValueError('Initial error - what kind of intensity ratio is required for plotting? --> (1) or (2)')

    ax2.tick_params(axis='both', which='major', labelsize=fontsize, pad=tickspad)
    ax2.set_ylabel('$pO_2$ [hPa]', labelpad=labelpad, fontsize=fontsize)
    ax2.set_xlabel('Temperature [°C]', labelpad=labelpad, fontsize=fontsize)

    if xlim is not None:
        ax2.set_xlim(xlim[0], xlim[1])
        ax3.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax2.set_ylim(ylim[0], ylim[1])
        ax3.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax2.set_zlim(zlim[0], zlim[1])
        ax3.set_zlim(zlim[0], zlim[1])

    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.92, top=0.9, hspace=0.2)
    plt.show()

    # combine to pd.Series for saving
    ZnOS_calibration = pd.Series({'Temperature [°C]': temp_x, 'pO2 [hPa]': pO2_y, 'I-ratio': Z_iratio,
                                  'I0 / I-ratio': Z_int0_int, 'tauP [ms]': Z_tau, 'tau0 / tauP': Z_tau0_tauP})

    if save_ is True:
        if path is None:
            path = input('Define path where to store your figures')
        else:
            pass
        if 'Graph' in path:
            path_calib_file = '/'.join(path.split('/')[:-1]) + '/'
            if path[-1] == '/':
                path_folder = path
            else:
                path_folder = path + '/'
        else:
            path_folder = path + '/Graph/'
            path_calib_file = path + '/'

        directory = os.path.dirname(path_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        saving_name = path_folder + '/' + str(date.today().year) + '-' + str(date.today().month) + '-' +\
                      str(date.today().day) + '_' + time + '-' + intensity.replace(' / ', '-') + '_spanned_area_inbetween.png'
        fig.savefig(saving_name)

        sav_name_csv = path_calib_file + str(date.today().year) + '-' + str(date.today().month) + '-' +\
                       str(date.today().day) + '_' + time + '-' + intensity.replace('/', '-') + '_calibrationfile.txt'
        ZnOS_calibration.to_csv(sav_name_csv)
        print(sav_name_csv)
    return ZnOS_calibration


def calib_plot_measurement_data(temp, temp_slit, df_full, limits_dPhi=None, limits_tauP=None, fontsize_=8,
                                limits_tau_quot=None, limits_ampl=None, limits_Iratio=None, tickspad_=0,
                                limits_norm=None, plot_phi=True, plot_ampl=True, limits_Inorm=None, color_palette=None,
                                ylabel='pO2 (firesting) [hPa]', xlabel_pO2='Temperatur', xticks_major_=2,
                                zlabel1='Phi(f1) [deg]', plateau=True, zlabel2='Phi(f2) [deg]', path=None,
                                yticks_major_=50, zticks_major_=None, savefig=False):
    if 'Time' in ylabel:
        for j, k in enumerate(df_full.columns):
            if 'O2' in k and 'mean' in k:
                pO2_label = k
        t0 = df_full[np.abs(df_full[pO2_label]) <= 0.01].mean()
        tau0 = t0['tauP [ms]']
        I0 = t0['I-ratio']
    else:
        tau0 = None
        I0 = None

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.set_facecolor('white')

    if color_palette is None:
        color_palette = ['k', 'k', 'k', 'k', 'navy', 'dodgerblue', 'forestgreen', 'lime green' ]
    else:
        color_palette = color_palette

    ampl_col = []
    for i, c in enumerate(df_full.columns):
        if 'A(' in c:
            ampl_col.append(c)
    dPhi_col = []
    for i, c in enumerate(df_full.columns):
        if 'Phi(' in c:
            dPhi_col.append(c)

    if plot_phi is True and plot_ampl is True:
        ax = fig.add_subplot(3, 4, 1, projection='3d')      # dPhi(f1)
        ax1 = fig.add_subplot(3, 4, 2, projection='3d')     # dPhi(f2)
        ax2 = fig.add_subplot(3, 4, 3, projection='3d')     # A(f1)
        ax3 = fig.add_subplot(3, 4, 4, projection='3d')     # A(f2)
        ax4 = fig.add_subplot(3, 4, 5, projection='3d')     # tauP
        ax5 = fig.add_subplot(3, 4, 6, projection='3d')     # tau0 / tauP
        ax6 = fig.add_subplot(3, 4, 9, projection='3d')     # I-ratio
        ax7 = fig.add_subplot(3, 4, 10, projection='3d')     # I0 / I-ratio
        ax8 = fig.add_subplot(3, 4, 11, projection='3d')     # I-ratio / I0
    else:
        ax = fig.add_subplot(3, 2, 1, projection='3d')      # tauP
        ax1 = fig.add_subplot(3, 2, 2, projection='3d')     # tau0 / tauP
        ax2 = fig.add_subplot(3, 2, 3, projection='3d')     # I-ratio
        ax3 = fig.add_subplot(3, 2, 4, projection='3d')     # I0 / I-ratio
        ax4 = fig.add_subplot(3, 2, 6, projection='3d')     # I-ratio / I0

    if limits_dPhi is None:
        pass
    else:
        t = df_full[df_full[ylabel] >= limits_dPhi['ymin']]
        tt_o2 = t[t[ylabel] <= limits_dPhi['ymax']]
        tt = tt_o2[tt_o2[xlabel_pO2] >= limits_dPhi['xmin']]
        df_full = tt[tt[xlabel_pO2] <= limits_dPhi['xmax']]

    if plot_phi is True and plot_ampl is True:
        # plotting phase angles dPhi(f1) and dPhi(f2)
        if plateau is True:
            df, ax = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax, labelpad=-2,
                                         zlabel=dPhi_col[0], xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[0],
                                         tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                         zticks_major=zticks_major_, fontsize=fontsize_)
            df1, ax1 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax1, labelpad=-2,
                                           zlabel=dPhi_col[1], xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[1],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df, ax = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax, fontsize=fontsize_, tau0=tau0,
                                                    I0=I0, xlabel=xlabel_pO2, limits=limits_dPhi, zlabel=dPhi_col[0],
                                                    tickspad=tickspad_, colors=color_palette[0], ylabel=ylabel)
            df1, ax1 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax1, fontsize=fontsize_, I0=I0,
                                                      tau0=tau0, xlabel=xlabel_pO2, ylabel=ylabel, limits=limits_dPhi,
                                                      zlabel=dPhi_col[1], tickspad=tickspad_, colors=color_palette[1])
        # ----------------------------------------------------------------------------
        # plotting amplitudes A(f1) and A(f2)
        # pre-check if mV or V is used in the data frame
        if plateau is True:
            df2, ax2 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax2, labelpad=-2,
                                           zlabel=ampl_col[0], xlabel=ylabel, ylabel=xlabel_pO2,
                                           colors=color_palette[0], tickspad=tickspad_, xticks_major=xticks_major_,
                                           yticks_major=yticks_major_, zticks_major=zticks_major_, fontsize=fontsize_)
            df3, ax3 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax3, labelpad=-1,
                                           zlabel=ampl_col[1], xlabel=ylabel, ylabel=xlabel_pO2,
                                           colors=color_palette[1], tickspad=tickspad_, xticks_major=xticks_major_,
                                           yticks_major=yticks_major_, zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df2, ax2 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax2,colors=color_palette[0],
                                                      fontsize=fontsize_, xlabel=xlabel_pO2, ylabel=ylabel,I0=I0,
                                                      tau0=tau0, limits=limits_ampl, zlabel=ampl_col[0],
                                                      tickspad=tickspad_)
            df3, ax3 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax3, colors=color_palette[1],
                                                      fontsize=fontsize_, xlabel=xlabel_pO2, ylabel=ylabel,I0=I0,
                                                      tau0=tau0, limits=limits_ampl, zlabel=ampl_col[1],
                                                      tickspad=tickspad_)
        # ----------------------------------------------------------------------------
        # plotting calculated lifetime tauP and tau0 / tauP
        if plateau is True:
            df4, ax4 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax4, labelpad=-2,
                                           zlabel='tauP [ms]', xlabel=ylabel, ylabel=xlabel_pO2,
                                           colors=color_palette[4], tickspad=tickspad_, xticks_major=xticks_major_,
                                           yticks_major=yticks_major_, zticks_major=zticks_major_, fontsize=fontsize_)
            df5, ax5 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax5, labelpad=-2,
                                           zlabel='tauP0 / tauP', xlabel=ylabel, ylabel=xlabel_pO2,
                                           colors=color_palette[5], tickspad=tickspad_, xticks_major=xticks_major_,
                                           yticks_major=yticks_major_, zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df4, ax4 = plot_3dsubplot_T_O2_continuous(df_full=df_full,labelpad=-2, ax=ax4, fontsize=fontsize_, I0=I0,
                                                      tau0=tau0, xlabel=xlabel_pO2, ylabel=ylabel, limits=limits_tauP,
                                                      zlabel='tauP [ms]', tickspad=tickspad_, colors=color_palette[4])
            df5, ax5 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax5, fontsize=fontsize_, I0=I0,
                                                      tau0=tau0, xlabel=xlabel_pO2, ylabel=ylabel, zlabel='tauP0 / tauP',
                                                      colors=color_palette[5], limits=limits_tau_quot,
                                                      tickspad=tickspad_)

        # ----------------------------------------------------------------------------
        # plotting calculated I-ratio and I0 / I-ratio
        if plateau is True:
            df6, ax6 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax6, labelpad=-5,
                                           zlabel='I-ratio', xlabel=ylabel, ylabel=xlabel_pO2,  colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
            df7, ax7 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax7, labelpad=-2,
                                           zlabel='I / I0', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
            df8, ax8 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax8, labelpad=-2,
                                           zlabel='I0 / I', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df6, ax6 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-5, ax=ax6, fontsize=fontsize_, I0=I0,
                                                      tau0=tau0, xlabel=xlabel_pO2, ylabel=ylabel, zlabel='I-ratio',
                                                      colors=color_palette[6], limits=limits_Iratio, tickspad=tickspad_)
            df7, ax7 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax7, fontsize=fontsize_,
                                                      xlabel=xlabel_pO2, ylabel=ylabel,colors=color_palette[6], I0=I0,
                                                      tau0=tau0, limits=limits_norm, zlabel='I / I0',
                                                      tickspad=tickspad_)
            df8, ax8 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax8, fontsize=fontsize_, I0=I0,
                                                      tau0=tau0, xlabel=xlabel_pO2, ylabel=ylabel, zlabel='I0 / I',
                                                      colors=color_palette[6], limits=limits_Inorm, tickspad=tickspad_)
    else:
        # plotting calculated lifetime tauP and tau0 / tauP
        if plateau is True:
            df4, ax = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax, labelpad=-2,
                                          zlabel='tauP [ms]', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[4],
                                          tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                          zticks_major=zticks_major_, fontsize=fontsize_)
            df5, ax1 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax1, labelpad=-2,
                                           zlabel='tauP0 / tauP', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[5],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df4, ax = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax, colors=color_palette[4],
                                                    fontsize=8, xlabel=xlabel_pO2, ylabel=ylabel, zlabel=zlabel1,
                                                    tickspad=tickspad_, limit=limits_tauP)
            df5, ax1 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax1, colors=color_palette[5],
                                                      fontsize=8, xlabel=xlabel_pO2, ylabel=ylabel, zlabel=zlabel2,
                                                      tickspad=tickspad_, limits=limits_tau_quot)
        # ----------------------------------------------------------------------------
        # plotting calculated I-ratio and I0 / I-ratio
        if plateau is True:
            df6, ax2 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax2, labelpad=-5,
                                           zlabel='I-ratio', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
            df7, ax3 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax3, labelpad=-2,
                                           zlabel='I / I0', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
            df8, ax4 = plot_3dsubplot_T_O2(temp=temp, temp_slit=temp_slit, df_full=df_full, ax=ax4, labelpad=-2,
                                           zlabel='I0 / I', xlabel=ylabel, ylabel=xlabel_pO2, colors=color_palette[6],
                                           tickspad=tickspad_, xticks_major=xticks_major_, yticks_major=yticks_major_,
                                           zticks_major=zticks_major_, fontsize=fontsize_)
        else:
            df6, ax2 = plot_3dsubplot_T_O2_continuous(df_full=df_full, ax=ax2, labelpad=-5, colors=color_palette[6],
                                                      fontsize=fontsize_, xlabel=xlabel_pO2, ylabel=ylabel,
                                                      zlabel='I-ratio', tickspad=tickspad_, limits=limits_Iratio)
            df7, ax3 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax3, colors=color_palette[6],
                                                      fontsize=fontsize_, xlabel=xlabel_pO2, ylabel=ylabel,
                                                      zlabel='I / I0', tickspad=tickspad_, limits=limits_norm)
            df8, ax4 = plot_3dsubplot_T_O2_continuous(df_full=df_full, labelpad=-2, ax=ax4, colors=color_palette[6],
                                                      fontsize=fontsize_, xlabel=xlabel_pO2, ylabel=ylabel,
                                                      zlabel='I0 / I', tickspad=tickspad_, limits=limits_Inorm)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.25)
    plt.show()

    if savefig is True:
        if path is None:
            path = input('Define path where to store your figures')
        else:
            pass
        if 'Graph' in path:
            if path[-1] == '/':
                path_folder = path
            else:
                path_folder = path + '/'
        else:
            path_folder = path + '/Graph/'

        directory = os.path.dirname(path_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        saving_name = path_folder + '/' + str(date.today().year) + '-' + str(date.today().month) + '-' +\
                      str(date.today().day) + '_measurement-data_evaluation_all.png'
        fig.savefig(saving_name)

    return df7


def creat_common_file(files_sync):
    temp_av_f1 = []
    pO2_ist_f1 = []
    pO2_soll_f1 = []
    pO2_slit_f1 = []
    temp_av_f2 = []
    pO2_ist_f2 = []
    pO2_soll_f2 = []
    pO2_slit_f2 = []

    # files_sync[s] = files_sync
    for i,j in enumerate(files_sync[0]['pO2_f1 soll [hPa]'].values):
        if np.isnan(j) == True:
            pass
        else:
            if j in pO2_soll_f1:
                pass
            else:
                pO2_soll_f1.append(j)
                pO2_slit_f1.append(i)
    for i,j in enumerate(files_sync[1]['pO2_f2 soll [hPa]'].values):
        if np.isnan(j) == True:
            pass
        else:
            if j in pO2_soll_f2:
                pass
            else:
                pO2_soll_f2.append(j)
                pO2_slit_f2.append(i)

    for i,k in enumerate(pO2_slit_f1):
        if i == len(pO2_slit_f1)-1:
            temp_av_f1.append(files_sync[0].iloc[k:].mean()['T_f1 [°C]'])
            pO2_ist_f1.append(files_sync[0].iloc[k:].mean()['pO2_f1 [hPa]'])
        else:
            temp_av_f1.append(files_sync[0].iloc[k: pO2_slit_f1[i+1]].mean()['T_f1 [°C]'])
            pO2_ist_f1.append(files_sync[0].iloc[k: pO2_slit_f1[i+1]].mean()['pO2_f1 [hPa]'])
    for i,k in enumerate(pO2_slit_f2):
        if i == len(pO2_slit_f2)-1:
            temp_av_f2.append(files_sync[1].iloc[k:].mean()['T_f2 [°C]'])
            pO2_ist_f2.append(files_sync[1].iloc[k:].mean()['pO2_f2 [hPa]'])
        else:
            temp_av_f2.append(files_sync[1].iloc[k: pO2_slit_f2[i+1]].mean()['T_f2 [°C]'])
            pO2_ist_f2.append(files_sync[1].iloc[k: pO2_slit_f2[i+1]].mean()['pO2_f2 [hPa]'])

    pO2_ist_av = [(i+j)/2 for (i,j) in zip(pO2_ist_f1, pO2_ist_f2)]
    pO2_soll_av = [(i+j)/2 for (i,j) in zip(pO2_soll_f1, pO2_soll_f2)]
    temp_av = [(i+j)/2 for (i,j) in zip(temp_av_f1, temp_av_f2)]

    return pO2_ist_av, pO2_soll_av, temp_av


def combining_measurement(run_files, set_T_constant, f1, f2):
    a = {}
    for i in range(len(run_files)):
        a[i] = pd.read_csv(run_files[i], sep='\t', decimal='.').sort_values(by='pO2.mean')
        a[i] = pd.DataFrame(a[i])

        if set_T_constant is True:
            a[i]['Temperature'] = [a[i]['Temp. Probe'].mean()] * len(a[i].index)
        else:
            a[i]['Temperature'] = a[i]['Temp. Probe']

        # re-name columns
        col_ls = a[i].columns.tolist()
        l_new = []
        for n in col_ls:
            if 'f1' in n:
                l_new.append(n.replace('f1', '{:.0f}Hz'.format(f1)))
            elif 'f2' in n:
                l_new.append(n.replace('f2', '{:.0f}Hz'.format(f2)))
            else:
                l_new.append(n)
        a[i].columns = l_new

        # calculate timedelta
        t1 = datetime.datetime.strptime(a[i]['DateTime'][0], "%Y-%m-%d %H:%M:%S")
        time_list_1 = []
        for c in a[i]['DateTime']:
            t_ = datetime.datetime.strptime(c, "%Y-%m-%d %H:%M:%S")
            time_list_1.append((t_ - t1).total_seconds())
        a[i]['Time [s]'] = time_list_1

    # combine to total dataframe
    df_all = a[0]
    for i in range(len(run_files) - 1):
        df_all = pd.concat([df_all, a[i + 1]], ignore_index=True)

    df_full_ = df_all[['Time [s]', 'Temperature', 'pO2.max', 'pO2.mean', 'pO2.min', 'dPhi({:.0f}Hz) [deg]'.format(f2),
                       'dPhi({:.0f}Hz) [deg]'.format(f1), 'tau.dual [s]', 'A({:.0f}Hz) [mV]'.format(f2),
                       'A({:.0f}Hz) [mV]'.format(f1), 'I pF/dF']]

    # convert tau [s] in [ms] and rename columns
    for i in df_full_.columns:
        if '[s]' in i:
            df_full_.loc[:, 'tauP [ms]'] = df_full_.loc[:, i] * 1000
        if 'I pF/dF' in i:
            df_full_.loc[:, 'I-ratio'] = df_full_.loc[:, i]

    df_full = df_full_[['Time [s]', 'Temperature', 'pO2.max', 'pO2.mean', 'pO2.min', 'dPhi({:.0f}Hz) [deg]'.format(f2),
                       'dPhi({:.0f}Hz) [deg]'.format(f1), 'tau.dual [s]', 'A({:.0f}Hz) [mV]'.format(f2),
                        'A({:.0f}Hz) [mV]'.format(f1), 'tauP [ms]', 'I-ratio']]
    df_full = df_full.sort_values(by='pO2.mean')
    df_full.index = np.arange(len(df_full.index))

    # ===============================================================================================
    drop_line = []
    for i in range(len(df_full['tauP [ms]'])):
        if np.isnan(df_full['tauP [ms]'].iloc[i]):
            drop_line.append(i)

    if len(drop_line) == 0.:
        df_new = df_full
    else:
        for i in drop_line:
            df_new = df_full.drop(index=i)

    temp = []
    temp_slit = []
    for i, j in enumerate(df_new['Temperature'].values):
        if np.isnan(j) == True:
            pass
        else:
            temp.append(j)
            temp_slit.append(i)

    return df_new, temp, temp_slit


def combine_para_slicing(limits_xy, limit_para):
    return limit_para.append(limits_xy)


def slice_dataframe_dualsensor(df_full, ylabel, xlabel, limits_xy):
    # slice according to the first parameter defined by ylabel
    df_sliced_ = df_full[df_full[ylabel] >= limits_xy['ymin']]
    df_sliced_y = df_sliced_[df_sliced_[ylabel] <= limits_xy['ymax']]

    # slice according to the second parameter defined by xlabel
    ddf_ = df_sliced_y[df_sliced_y[xlabel] >= limits_xy['xmin']]
    df_sliced = ddf_[ddf_[xlabel] <= limits_xy['xmax']].sort_values(by='Time [s]')
    df_sliced.index = np.arange(len(df_sliced))

    return df_sliced


# --------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------
def two_gas_mixture_to_total_pressure_hPa(flow_gas1, flow_gas2, conc_gas1_max, conc_gas2_max, flow_combined):
    part_gas1 = flow_gas1 / flow_combined
    conc_gas1 = part_gas1 * conc_gas1_max
    part_gas2 = flow_gas2 / flow_combined
    conc_gas2 = part_gas2 * conc_gas2_max

    pressure_total = conc_gas1 + conc_gas2
    return pressure_total


def sync_LockIn_firesting(file_lockIn_f1, file_lockIn_f2, file_FS_f1, file_FS_f2, usecols_lockIn=None, saving=False,
                          index_lockIn='Uhrzeit', conc_gas1_max=0., conc_gas2_max=198.85, flow_combined=100.):
    if usecols_lockIn is None:
        usecols_lockIn = [0, 1, 2, 4, 6, 8]

    # pre-check if the samples were measured at the same temperature
    a = file_lockIn_f1.split('/')[-1].split('_')[-2]
    b = file_lockIn_f2.split('/')[-1].split('_')[-2]
    c = file_FS_f1.split('/')[-1].split('_')[-2]
    d = file_FS_f2.split('/')[-1].split('_')[-2]
    if a == b == c == d:
        pass
    else:
        raise ValueError('The samples ought to be measured at the same temperature!')

    # data lockIn
    df_lockIn1 = convert_austrian_timestamp_LockIn(p=file_lockIn_f1,
                                                   df=read_data(p=file_lockIn_f1, usecols=usecols_lockIn,
                                                                index_=index_lockIn))
    df_lockIn2 = convert_austrian_timestamp_LockIn(p=file_lockIn_f2,
                                                   df=read_data(p=file_lockIn_f2, usecols=usecols_lockIn,
                                                                index_=index_lockIn))

    # ----------------------------------------------------------------

    flow_gas11 = df_lockIn1[df_lockIn1.columns[-2]].values # flow N2
    flow_gas12 = df_lockIn1[df_lockIn1.columns[-1]].values # flow O2
    pressure_soll1 = two_gas_mixture_to_total_pressure_hPa(flow_gas11, flow_gas12, conc_gas1_max, conc_gas2_max,
                                                           flow_combined)
    flow_gas21 = df_lockIn2[df_lockIn2.columns[-2]].values # flow N2
    flow_gas22 = df_lockIn2[df_lockIn2.columns[-1]].values # flow O2
    pressure_soll2 = two_gas_mixture_to_total_pressure_hPa(flow_gas21, flow_gas22, conc_gas1_max, conc_gas2_max,
                                                           flow_combined)

    # ========================================================================================================================
    # Firesting
    df_f1 = write_txt_to_df(p=file_FS_f1)
    df_f2 = write_txt_to_df(p=file_FS_f2)

    data_oxygen1 = df_f1[df_f1.columns[4]].values
    data_oxygen2 = df_f2[df_f2.columns[4]].values

    temperature1 = df_f1[df_f1.columns[14]]
    temperature2 = df_f2[df_f2.columns[14]]

    time1 = pd.DatetimeIndex(df_f1[df_f1.columns[1]].values)
    time2 = pd.DatetimeIndex(df_f2[df_f2.columns[1]].values)
    date_meas1 = df_f1[df_f1.columns[0]].values[0].split('.')
    date_meas2 = df_f2[df_f2.columns[0]].values[0].split('.')
    time1_FS = time1.map(lambda t: t.replace(year=int(date_meas1[2]), month=int(date_meas1[1]), day=int(date_meas1[0])))
    time2_FS = time2.map(lambda t: t.replace(year=int(date_meas2[2]), month=int(date_meas2[1]), day=int(date_meas2[0])))

    # Shift FireSting to timestamp of LockIn
    timedelta1 = time1_FS[0] - df_lockIn1.index[0]
    time_FS_new1 = time1_FS - timedelta1

    timedelta2 = time2_FS[0] - df_lockIn2.index[0]
    time_FS_new2 = time2_FS - timedelta2

    dd1 = pd.concat([pd.DataFrame(data_oxygen1, index=time_FS_new1, columns=['pO2_f1 [hPa]']),
                pd.DataFrame(temperature1.values, index=time_FS_new1, columns=['T_f1 [°C]']),
                pd.DataFrame(pressure_soll1, index=df_lockIn1.index, columns=['pO2_f1 soll [hPa]'])], axis=1)

    dd2 = pd.concat([pd.DataFrame(data_oxygen2, index=time_FS_new2, columns=['pO2_f2 [hPa]']),
                    pd.DataFrame(temperature2.values, index=time_FS_new2, columns=['T_f2 [°C]']),
                    pd.DataFrame(pressure_soll2, index=df_lockIn2.index, columns=['pO2_f2 soll [hPa]'])], axis=1)
    prev_folder = None

    if saving is True:
        for i, j in enumerate(file_FS_f1.split('/')):
            if j == 'FireSting':
                prev_folder = i
        if prev_folder == None:
                raise ValueError('Error - no folder named FireSting found')

        path = '/'.join(file_FS_f1.split('/')[:prev_folder]) + '/'

        savingname1 = path + date_meas1[2] + '-' + date_meas1[1] + '-' + date_meas1[0] +\
                      '_pO2_LockIn_Firesting_synchronization_f1_' + file_FS_f1.split('/')[-1].split('_')[2] + '.txt'
        savingname2 = path + date_meas1[2] + '-' + date_meas1[1] + '-' + date_meas1[0] +\
                      '_pO2_LockIn_Firesting_synchronization_f2_' + file_FS_f2.split('/')[-1].split('_')[2] + '.txt'
        dd1.to_csv(savingname1, sep='\t', decimal='.')
        dd2.to_csv(savingname2, sep='\t', decimal='.')

    return dd1, dd2


def individual_evaluation_continuous(data_all, dphi, day, timerange_0pO2, timerange_2pO2, meas_range, freq_pair, freq_Hz,
                                     pO2_hPa, m_fs=0.093, f_fs=0.72, pO2_2pc=18.942, fmod_fs=3000, plot_=True,
                                     save_results=True):

    # plateau and reference sensor calibration
    # sensor characterisation and sensor calibration
    [fs_2pc, fs_0pc,
     para_TSM_ref] = multi.characterisation_sensors(timerange_2pO2=timerange_2pO2, plotting=plot_, f_fs=f_fs,
                                                    timerange_0pO2=timerange_0pO2, df_dphi=data_all, m_fs=m_fs,
                                                    fmod_fs=fmod_fs, pO2_2pc=pO2_2pc)

    # plateau evaluation dual-sensor
    [err, tau0, tau2, int_ratio0,
     int_ratio2] = multi.dualsensor_evalaution_plateau(fs_2pc=fs_2pc, fs_0pc=fs_0pc, freq_Hz=freq_Hz,
                                                       freq_pair=freq_pair)

    # ================================================================================================================
    # Reference evaluation
    # lifetime reference
    tau_ref = np.tan(np.deg2rad(dphi['dphi1 raw'])) / (2 * np.pi * fmod_fs)
    df_tau_ref = pd.DataFrame(tau_ref)
    df_tau_ref.columns = ['tau.ref [s]']

    # tau0/tau quotient for tsm
    df_tau0_tau = pd.concat([t0 / df_tau_ref for t0 in para_TSM_ref['tauP0']], axis=1)
    df_tau0_tau.columns = ['tau0/tau.min', 'tau0/tau.mean', 'tau0/tau.max']

    # pO2 evaluation
    pO2_1_calc = []
    pO2_2_calc = []
    for t in tau_ref.values:
        pO2_1, pO2_2 = multi.twoSiteModel_evaluation_(tau0=para_TSM_ref['tauP0'], tau=t, m=para_TSM_ref['prop Ksv'],
                                                f=para_TSM_ref['slope'], ksv=para_TSM_ref['Ksv_fit1'],
                                                pO2_range=pO2_hPa)
        pO2_1_calc.append(pO2_1)
        pO2_2_calc.append(pO2_2)

    pO2_ref_calc = pd.DataFrame(pO2_1_calc, columns=['pO2.min', 'pO2.mean', 'pO2.max'], index=tau_ref.index)
    df_reference_ = pd.concat([df_tau_ref, df_tau0_tau, pO2_ref_calc, dphi['Temp. Probe']], axis=1, sort=True)
    df_reference_['tau.ref [µs]'] = df_reference_['tau.ref [s]'] * 1e6
    df_reference = df_reference_.sort_values(by='pO2.mean')

    # ================================================================================================================
    # lifetime dualsensor
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

    dphi_columns = ['dPhi(f{}) [deg]'.format(freq_pair[0]), 'dPhi(f{}) [deg]'.format(freq_pair[1]),
                    'A(f{}) [mV]'.format(freq_pair[0]), 'A(f{}) [mV]'.format(freq_pair[1])]
    df_dphi_eval = data_all[dphi_columns]

    tauP1 = []
    tauP2 = []
    for i, v in enumerate(df_dphi_eval.index):
        if isinstance(df_dphi_eval[df_dphi_eval.columns[0]].loc[df_dphi_eval.index[i]], np.float):
            p1 = np.deg2rad(df_dphi_eval[df_dphi_eval.columns[0]].loc[df_dphi_eval.index[i]])
        else:
            p1 = np.deg2rad(df_dphi_eval[df_dphi_eval.columns[0]].loc[df_dphi_eval.index[i]].mean())

        if isinstance(df_dphi_eval[df_dphi_eval.columns[1]].loc[df_dphi_eval.index[i]], np.float):
            p2 = np.deg2rad(df_dphi_eval[df_dphi_eval.columns[1]].loc[df_dphi_eval.index[i]])
        else:
            p2 = np.deg2rad(df_dphi_eval[df_dphi_eval.columns[1]].loc[df_dphi_eval.index[i]].mean())

        phi1 = [p1 - err, p1, p1 + err]
        phi2 = [p2 - err, p2, p2 + err]

        tau_1, tau_2 = af.two_frequency_lifetime(f1=f1, f2=f2, Phi_f1_rad=phi1, Phi_f2_rad=phi2)
        tauP1.append(tau_1)
        tauP2.append(tau_2)

    df_tau1 = pd.DataFrame(tauP1)
    df_tau_dual = pd.DataFrame(df_tau1[1])
    df_tau_dual.columns = ['tau.dual [s]']
    df_tau_dual.index = data_all.index
    l_ = [t0 / df_tau_dual for t0 in tau0]
    df_dual = pd.concat(l_, axis=1, sort=True)
    df_dual.columns = ['tau0/tau.dual.min', 'tau0/tau.dual.mean', 'tau0/tau.dual.max']

    df = pd.concat([df_reference_, df_tau_dual, df_dual, data_all[dphi_columns]], axis=1, sort=True)

    # slice evaluation range
    if len(meas_range) <= 1:
        measurement_range = dphi.index[0], dphi.index[-1]
    else:
        measurement_range = datetime.datetime.strptime(meas_range.split(',')[0], "%H:%M:%S"), \
                            datetime.datetime.strptime(meas_range.split(',')[1].split()[0], "%H:%M:%S")

    m_stemp = []
    for n, i in enumerate(dphi.index):
        if measurement_range[1].strftime('%H:%M:%S') >= i.strftime('%H:%M:%S') >= measurement_range[0].strftime('%H:%M:%S'):
            m_stemp.append(i)
    df_dphi_sliced = df.loc[m_stemp[0]:m_stemp[-1]]

    # intensity ratio
    dual_tau = df_dphi_sliced['tau.dual [s]'].dropna()
    dual_dphi1 = df_dphi_sliced.loc[dual_tau.index][dphi_columns[0]].values

    iratio = af.intensity_ratio(f=f1, tau=dual_tau, phi=dual_dphi1)
    df_iratio = pd.DataFrame(iratio, index=dual_tau.index)
    df_iratio.columns = ['I pF/dF']
    df_dphi_all = pd.concat([df_dphi_sliced, df_iratio], axis=1, sort=True)

    col_new = []
    for i, c in enumerate(df_dphi_all.columns):
        if 'dPhi' in c:
            num_ = np.int(c.split('f')[1].split(')')[0]) - 1
            col_new.append('dPhi({:.0f}Hz) [deg]'.format(freq_Hz[num_]))
        elif 'A(f' in c:
            num_ = np.int(c.split('f')[1].split(')')[0]) - 1
            col_new.append('A({:.0f}Hz) [mV]'.format(freq_Hz[num_]))
        else:
            col_new.append(c)

    df_dphi_all.columns = col_new

    if save_results is True:
        df_dphi_all.to_csv(day.strftime("%Y%m%d") + '_all-sensor_data_{:.0f}deg_f1-{:.0f}Hz_f2-{:.0f}Hz.txt'.format(
            df_reference['Temp. Probe'].mean(), freq_Hz[freq_pair[0] - 1], freq_Hz[freq_pair[1] - 1]), sep='\t',
                           decimal='.')

        df_reference.to_csv(day.strftime("%Y%m%d") + '_reference-sensor_{:.0f}deg_f1-{:.0f}Hz_f2-{:.0f}Hz.txt'.format(
            df_reference['Temp. Probe'].mean(), freq_Hz[freq_pair[0] - 1], freq_Hz[freq_pair[1] - 1]), sep='\t',
                            decimal='.')

    return df_dphi_all
