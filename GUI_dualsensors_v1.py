__author__ = 'szieger'
__project__ = 'dualsensor'

import matplotlib
matplotlib.use('Qt5Agg')
import additional_functions as fp
import Dualsensing_ph_oxygen as pHox
import Dualsensing_ph_T as pHtemp
import Dualsensing_CO2_O2 as cox
import Dualsensing_T_O2 as Tox
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow, QTabWidget, QProgressBar,
                             QGridLayout, QLabel, QLineEdit, QGroupBox, QTableWidget, QMessageBox,
                             QTableWidgetItem, QPushButton, QCheckBox, QFileDialog, QTextEdit)
from PyQt5.QtGui import QDoubleValidator, QRegExpValidator, QColor
from PyQt5.QtCore import Qt, QRegExp
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
import seaborn as sns
import os
from collections import OrderedDict
import statistics
import itertools
from scipy import interpolate
from scipy import stats
import math
import datetime
import matplotlib.pyplot as plt


sns.set_palette('Set1')
sns.set_context('paper', font_scale=1., rc={"grid.linewidth": 0.25})
sns.set_style("ticks", {"axes.facecolor": ".99", "image.cmap": 'rocket'})

plt.rcParams.update({'figure.max_open_warning': 0})

conv_temp = 273.15

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

color_phaseangle = {'τ1': '#06425C',
                    'τ2': '#CDBB79',
                    'τ3': '#B7695C'}

color_combined = {'τ1, I1':     '#003b46',  # blues
                  'τ2, I1':     '#45a0c1',
                  'τ3, I1':     '#a5b7ba',

                  'τ1, I2':     '#af4425',  # reds
                  'τ2, I2':     '#faaf08',
                  'τ3, I2':     '#c9a66b',

                  'τ1, I3':     '#00ad47',  # green
                  'τ2, I3':     '#68a225',
                  'τ3, I3':     '#52958b'}

class Gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.xcoords = []

    def initUI(self):
        # creating main window
        w = QWidget()
        self.setCentralWidget(w)

        self.setGeometry(20, 40, 100, 100)
        self.setWindowTitle('GUI - Dualsensor and error propagation')
        self.statusBar()

        # Set window background color
        self.setAutoFillBackground(True)
        self.setStyleSheet("selection-color:navy")

        # ------------------------------------------------------------------------------------
        # (Invisible) structure of main window (=grid)
        # -------------------------------------------------------------------------------------
        hbox = QHBoxLayout(w)

        vbox_left = QVBoxLayout()
        vbox_right = QVBoxLayout()

        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)

        # -------------------------------------------
        vbox_left.addWidget(w)
        vbox_left.setContentsMargins(2, 2, 2, 2)

        vbox_right.addWidget(w)
        vbox_right.setContentsMargins(2, 2, 2, 2)

        # --------------------------------------------------------------------------------------
        # left part (TOP)
        # --------------------------------------------------------------------------------------
        # Input data: f1, f2, tau, I-ratio, error
        grid_input_para = QGridLayout()
        vbox_left.addLayout(grid_input_para)

        # -------------------------------------------
        # lifetime phosphor
        lifetime_phosphor = QLabel(self)
        lifetime_unit = QLabel(self)
        lifetime_unit1 = QLabel(self)
        lifetime_unit2 = QLabel(self)
        lifetime_phosphor.setText('Lifetime')
        lifetime_unit.setText('µs')
        lifetime_unit1.setText('µs')
        lifetime_unit2.setText('µs')
        self.lifetime_phosphor_edit = QLineEdit(self)
        self.lifetime_phosphor_edit.setValidator(QDoubleValidator())
        self.lifetime_phosphor_edit.setMaximumWidth(40)
        self.lifetime_phosphor_edit.setText('50.')
        self.lifetime_phosphor_edit.setAlignment(Qt.AlignRight)
        self.lifetime_phosphor1_edit = QLineEdit(self)
        self.lifetime_phosphor1_edit.setValidator(QDoubleValidator())
        self.lifetime_phosphor1_edit.setMaximumWidth(40)
        self.lifetime_phosphor1_edit.setText('70.')
        self.lifetime_phosphor1_edit.setAlignment(Qt.AlignRight)
        self.lifetime_phosphor2_edit = QLineEdit(self)
        self.lifetime_phosphor2_edit.setValidator(QDoubleValidator())
        self.lifetime_phosphor2_edit.setMaximumWidth(40)
        self.lifetime_phosphor2_edit.setText('100.')
        self.lifetime_phosphor2_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # intensity ratios for simulation (I-1 to I-3)
        intensity_ratio = QLabel(self)
        intensity_ratio.setText('I ratio 1')
        intensity_ratio1 = QLabel(self)
        intensity_ratio1.setText('I ratio 2')
        intensity_ratio2 = QLabel(self)
        intensity_ratio2.setText('I ratio 3')
        self.intensity_ratio_edit = QLineEdit(self)
        self.intensity_ratio_edit.setMaximumWidth(40)
        self.intensity_ratio_edit.setValidator(QDoubleValidator())
        self.intensity_ratio_edit.setText('0.5')
        self.intensity_ratio_edit.setAlignment(Qt.AlignRight)
        self.intensity_ratio1_edit = QLineEdit(self)
        self.intensity_ratio1_edit.setValidator(QDoubleValidator())
        self.intensity_ratio1_edit.setMaximumWidth(40)
        self.intensity_ratio1_edit.setAlignment(Qt.AlignRight)
        self.intensity_ratio2_edit = QLineEdit(self)
        self.intensity_ratio2_edit.setValidator(QDoubleValidator())
        self.intensity_ratio2_edit.setMaximumWidth(40)
        self.intensity_ratio2_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # measurement uncertainty in phase angle
        error_assumed = QLabel(self)
        error_assumed_unit = QLabel(self)
        error_assumed.setText('Assumed uncertainty ±')
        error_assumed_unit.setText('°')
        self.error_assumed_edit = QLineEdit(self)
        self.error_assumed_edit.setValidator(QDoubleValidator())
        self.error_assumed_edit.setMaximumWidth(40)
        self.error_assumed_edit.setText('0.1')
        self.error_assumed_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # modulation frequency 1 for simulation (fixed)
        frequency1 = QLabel(self)
        frequency1.setText('Modulation frequency f1')
        frequency1_unit = QLabel(self)
        frequency1_unit.setText('Hz')
        self.frequency1_edit = QLineEdit(self)
        self.frequency1_edit.setValidator(QDoubleValidator())
        self.frequency1_edit.setMaximumWidth(40)
        self.frequency1_edit.setText('3000.')
        self.frequency1_edit.setAlignment(Qt.AlignRight)

        # modulation frequency f2 range for measurement simulation
        frequency2 = QLabel(self)
        frequency2_unit = QLabel(self)
        frequency2_unit2 = QLabel(self)
        frequency2_line = QLabel(self)
        frequency2.setText('Modulation frequency f2 range')
        frequency2_unit.setText('Hz - Steps')
        frequency2_unit2.setText('Hz')
        frequency2_line.setText('-')
        self.frequency2_min_edit = QLineEdit(self)
        self.frequency2_min_edit.setValidator(QDoubleValidator())
        self.frequency2_min_edit.setMaximumWidth(40)
        self.frequency2_min_edit.setText('500.')
        self.frequency2_min_edit.setAlignment(Qt.AlignRight)
        self.frequency2_max_edit = QLineEdit(self)
        self.frequency2_max_edit.setValidator(QDoubleValidator())
        self.frequency2_max_edit.setMaximumWidth(40)
        self.frequency2_max_edit.setText('20000.')
        self.frequency2_max_edit.setAlignment(Qt.AlignRight)
        self.frequency2_step_edit = QLineEdit(self)
        self.frequency2_step_edit.setValidator(QDoubleValidator())
        self.frequency2_step_edit.setMaximumWidth(40)
        self.frequency2_step_edit.setText('200.')
        self.frequency2_step_edit.setAlignment(Qt.AlignRight)

        # ------------------------------------------------------------
        # Select combination by checkboxes
        self.lifetime_checkbox = QCheckBox('τ1', self)
        self.lifetime_checkbox.toggle()
        self.lifetime2_checkbox = QCheckBox('τ2', self)
        self.lifetime3_checkbox = QCheckBox('τ3', self)
        if self.lifetime3_checkbox.checkState() != 2:
            self.lifetime2_checkbox.toggle()
        self.intensity_checkbox = QCheckBox('I-ratio1', self)
        self.intensity_checkbox.toggle()
        self.intensity2_checkbox = QCheckBox('I-ratio2', self)
        self.intensity3_checkbox = QCheckBox('I-ratio3', self)
        if self.intensity3_checkbox.checkState() != 2:
            self.lifetime2_checkbox.toggle()

        self.lifetime_checkbox.adjustSize()
        self.lifetime2_checkbox.adjustSize()
        self.lifetime3_checkbox.adjustSize()
        self.intensity_checkbox.adjustSize()
        self.intensity2_checkbox.adjustSize()
        self.intensity3_checkbox.adjustSize()

        # -------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_parameter_group = QGroupBox("Parameter for error propagation")
        grid_parameter = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        vbox_left.addWidget(set_parameter_group)
        set_parameter_group.setLayout(grid_parameter)

        grid_parameter.addWidget(lifetime_phosphor, 1, 0)
        grid_parameter.addWidget(self.lifetime_phosphor_edit, 1, 1)
        grid_parameter.addWidget(lifetime_unit, 1, 2)
        grid_parameter.addWidget(self.lifetime_phosphor1_edit, 1, 3)
        grid_parameter.addWidget(lifetime_unit1, 1, 4)
        grid_parameter.addWidget(self.lifetime_phosphor2_edit, 1, 5)
        grid_parameter.addWidget(lifetime_unit2, 1, 6)

        grid_parameter.addWidget(intensity_ratio, 2, 0)
        grid_parameter.addWidget(self.intensity_ratio_edit, 2, 1)
        grid_parameter.addWidget(intensity_ratio1, 2, 2)
        grid_parameter.addWidget(self.intensity_ratio1_edit, 2, 3)
        grid_parameter.addWidget(intensity_ratio2, 2, 4)
        grid_parameter.addWidget(self.intensity_ratio2_edit, 2, 5)

        grid_parameter.addWidget(error_assumed, 3, 0)
        grid_parameter.addWidget(error_assumed_unit, 3, 2)
        grid_parameter.addWidget(self.error_assumed_edit, 3, 1)

        grid_parameter.addWidget(frequency1, 4, 0)
        grid_parameter.addWidget(self.frequency1_edit, 4, 1)
        grid_parameter.addWidget(frequency1_unit, 4, 2)

        grid_parameter.addWidget(frequency2, 5, 0)
        grid_parameter.addWidget(self.frequency2_min_edit, 5, 1)
        grid_parameter.addWidget(frequency2_line, 5, 2)
        grid_parameter.addWidget(self.frequency2_max_edit, 5, 3)
        grid_parameter.addWidget(frequency2_unit, 5, 4)

        grid_parameter.addWidget(self.frequency2_step_edit, 5, 5)
        grid_parameter.addWidget(frequency2_unit2, 5, 6)

        set_parameter_group.setContentsMargins(2, 10, 2, 2)
        vbox_left.addSpacing(2)

        # -------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_combination_group = QGroupBox("Select combinations for error propagation")
        grid_combination = QGridLayout()

        grid_combination.addWidget(self.lifetime_checkbox, 0, 0)
        grid_combination.addWidget(self.lifetime2_checkbox, 0, 1)
        grid_combination.addWidget(self.lifetime3_checkbox, 0, 2)
        grid_combination.addWidget(self.intensity_checkbox, 1, 0)
        grid_combination.addWidget(self.intensity2_checkbox, 1, 1)
        grid_combination.addWidget(self.intensity3_checkbox, 1, 2)

        # add GroupBox to layout and load buttons in GroupBox
        vbox_left.addWidget(set_combination_group)
        set_combination_group.setLayout(grid_combination)

        set_combination_group.setContentsMargins(2, 10, 2, 2)
        vbox_left.addSpacing(2)

        # -----------------------------------------------------------------------------------
        # connect LineEdit with function (read_csv and do stuff)
        # updating lineEdit
        self.lifetime_phosphor_edit.editingFinished.connect(self.print_lifetime_phosphor)
        self.lifetime_phosphor1_edit.editingFinished.connect(self.print_lifetime_phosphor1)
        self.lifetime_phosphor2_edit.editingFinished.connect(self.print_lifetime_phosphor2)

        self.intensity_ratio_edit.editingFinished.connect(self.print_intensity_ratio)
        self.intensity_ratio1_edit.editingFinished.connect(self.print_intensity_ratio1)
        self.intensity_ratio2_edit.editingFinished.connect(self.print_intensity_ratio2)

        self.error_assumed_edit.editingFinished.connect(self.print_error_assumed)
        self.frequency1_edit.editingFinished.connect(self.print_frequency1)
        self.frequency2_min_edit.editingFinished.connect(self.print_frequency2_range)
        self.frequency2_max_edit.editingFinished.connect(self.print_frequency2_range)
        self.frequency2_step_edit.editingFinished.connect(self.print_frequency2_range)

        # connect with function
        self.lifetime_checkbox.clicked.connect(self.reportInput_lifetime)
        self.lifetime_checkbox.clicked.connect(self.control_input)
        self.lifetime2_checkbox.clicked.connect(self.reportInput_lifetime2)
        self.lifetime2_checkbox.clicked.connect(self.control_input)
        self.lifetime3_checkbox.clicked.connect(self.reportInput_lifetime3)
        self.lifetime3_checkbox.clicked.connect(self.control_input)

        self.intensity_checkbox.clicked.connect(self.reportInput_intensity)
        self.intensity_checkbox.clicked.connect(self.control_input)
        self.intensity2_checkbox.clicked.connect(self.reportInput_intensity2)
        self.intensity2_checkbox.clicked.connect(self.control_input)
        self.intensity3_checkbox.clicked.connect(self.reportInput_intensity3)
        self.intensity3_checkbox.clicked.connect(self.control_input)

        # --------------------------------------------------------------------------------------
        # left part (MIDDLE)
        # --------------------------------------------------------------------------------------
        # simulation or measurement
        # Select combination by checkboxes
        sim_or_meas = QLabel(self)
        sim_or_meas.setText('Simulation or measurement')
        self.simulation_checkbox = QCheckBox('simulation', self)
        self.simulation_checkbox.toggle()
        self.measurement_checkbox = QCheckBox('measurement', self)

        # ---------------------------------------------------------------------
        # modulation frequencies f1&f2 fixed for measurement evaluation
        frequency1_fix = QLabel(self)
        frequency1fix_unit = QLabel(self)
        frequency1_fix.setText('Modulation frequency f1')
        frequency1fix_unit.setText('Hz \t I ratio max I_F / I_P')
        self.frequency1fix_edit = QLineEdit(self)
        self.frequency1fix_edit.setValidator(QDoubleValidator())
        self.frequency1fix_edit.setMaximumWidth(40)
        self.frequency1fix_edit.setText('3000.')
        self.frequency1fix_edit.setAlignment(Qt.AlignRight)

        frequency2_fix = QLabel(self)
        frequency2fix_unit = QLabel(self)
        frequency2_fix.setText('Modulation frequency f2')
        frequency2fix_unit.setText('Hz \t Lifetime tauP')
        self.frequency2fix_edit = QLineEdit(self)
        self.frequency2fix_edit.setValidator(QDoubleValidator())
        self.frequency2fix_edit.setMaximumWidth(40)
        self.frequency2fix_edit.setText('7000.')
        self.frequency2fix_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # measurement uncertainty in phase angle
        error_assumed_meas = QLabel(self)
        error_assumed_meas_unit = QLabel('°', self)
        error_assumed_meas.setText('Assumed uncertainty ±')
        self.error_assumed_meas_edit = QLineEdit(self)
        self.error_assumed_meas_edit.setValidator(QDoubleValidator())
        self.error_assumed_meas_edit.setMaximumWidth(40)
        self.error_assumed_meas_edit.setText('0.1')
        self.error_assumed_meas_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # maximal intensity of the fluorophor and phosphor
        self.int_ratio_dualsens_edit = QLineEdit(self)
        self.int_ratio_dualsens_edit.setValidator(QDoubleValidator())
        self.int_ratio_dualsens_edit.setMaximumWidth(40)
        self.int_ratio_dualsens_edit.setText('0.5')
        self.int_ratio_dualsens_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # lifetime phosphor
        lifetime_phos_dualsens_unit = QLabel('µs', self)
        lifetime_phos_dualsens_unit.move(50, 50)
        self.lifetime_phos_dualsens_edit = QLineEdit(self)
        self.lifetime_phos_dualsens_edit.setValidator(QDoubleValidator())
        self.lifetime_phos_dualsens_edit.setMaximumWidth(40)
        self.lifetime_phos_dualsens_edit.setText('51.86')
        self.lifetime_phos_dualsens_edit.setAlignment(Qt.AlignRight)

        # -------------------------------------------
        # open temperature compensation file
        self.load_temp_comp_button = QPushButton('Load T compensation', self)
        self.load_temp_comp_button.setStyleSheet("color: white; background-color:"
                                                 "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                 "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                 "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px;"
                                                 "padding-left: 1px; padding-right: 5px; min-height: 10px;"
                                                 "max-height: 18px;")
        self.compensation_edit = QTextEdit(self)
        self.compensation_edit.setReadOnly(True)
        self.compensation_edit.setMaximumHeight(25)

        # -------------------------------------------
        # open tauP -> IntP conversion compensation file
        self.load_tauP_intP_conv_button = QPushButton('Load tauP -> IntP conversion', self)
        self.load_tauP_intP_conv_button.setStyleSheet("color: white; background-color:"
                                                      "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                      "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                      "border-style: solid; border-radius: 7; padding: 5px; "
                                                      "font-size: 10px; padding-left: 1px; padding-right: 5px; "
                                                      "min-height: 10px; max-height: 18px;")
        self.conversion_edit = QTextEdit(self)
        self.conversion_edit.setReadOnly(True)
        self.conversion_edit.setMaximumHeight(25)

        # -------------------------------------------
        # open calibration file for T/O2
        self.load_O_T_calib_button = QPushButton('Load calibration T/O2', self)
        self.load_O_T_calib_button.setStyleSheet("color: white; background-color:"
                                                 "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                 "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                 "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px;"
                                                 "padding-left: 1px; padding-right: 5px; min-height: 10px;"
                                                 "max-height: 18px;")
        self.calib_edit = QTextEdit(self)
        self.calib_edit.setReadOnly(True)
        self.calib_edit.setMaximumHeight(25)

        # -------------------------------------------
        # What information is given during measurement -  intensity ratio or total amplitude
        intensity_amplitude_measured = QLabel(self)
        intensity_amplitude_measured.setText('I-ratio or total amplituide')
        self.int_ratio_checkbox = QCheckBox('I-ratio', self)
        self.int_ratio_checkbox.toggle()
        self.total_amplitude_checkbox = QCheckBox('A total', self)

        # -------------------------------------------
        # Table for calibration
        table_title_calib = QLabel(self)
        table_title_calib.setText('Calibration')
        self.tableCalibration = QTableWidget(parent=self)
        self.tableCalibration.setColumnCount(7)
        self.tableCalibration.setRowCount(4)
        self.tableCalibration.move(30, 30)
        self.tableCalibration.setColumnWidth(0, 69)
        self.tableCalibration.setColumnWidth(1, 80)
        self.tableCalibration.setColumnWidth(2, 90)
        self.tableCalibration.setColumnWidth(3, 80)
        self.tableCalibration.setColumnWidth(4, 80)
        self.tableCalibration.setColumnWidth(5, 85)
        self.tableCalibration.setColumnWidth(6, 85)
        self.tableCalibration.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                         "dPhi(f2) [°];  I-ratio").split(";"))
        self.tableCalibration.setStyleSheet("QHeaderView { font-size: 8pt; }")
        self.tableCalibration.resizeColumnToContents(0)

        # -------------------------------------------
        # Measurement input
        table_title_meas = QLabel(self)
        table_title_meas.setText('Input simulation / measurement')

        self.tableINPUT = QTableWidget(parent=self)
        self.tableINPUT.setColumnCount(7)
        self.tableINPUT.setRowCount(1)
        # self.tableINPUT.setFixedSize(580, 66)
        self.tableINPUT.setColumnWidth(0, 69)
        self.tableINPUT.setColumnWidth(1, 80)
        self.tableINPUT.setColumnWidth(2, 90)
        self.tableINPUT.setColumnWidth(3, 80)
        self.tableINPUT.setColumnWidth(4, 80)
        self.tableINPUT.setColumnWidth(5, 85)
        self.tableINPUT.setColumnWidth(6, 85)
        self.tableINPUT.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; dPhi(f2) [°]; "
                                                   "I-ratio").split(";"))
        self.tableINPUT.setStyleSheet("QHeaderView { font-size: 8pt; }")
        self.tableINPUT.resizeColumnToContents(0)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_dualsensor_group = QGroupBox("Dualsensor simulation or measurement")
        grid_dualsensor = QGridLayout()

        # simulation / measurement selection
        grid_dualsensor.addWidget(sim_or_meas, 1, 0)
        grid_dualsensor.addWidget(self.simulation_checkbox, 1, 2)
        grid_dualsensor.addWidget(self.measurement_checkbox, 1, 3)

        # general input
        grid_dualsensor.addWidget(frequency1_fix, 4, 0)
        grid_dualsensor.addWidget(self.frequency1fix_edit, 4, 1)
        grid_dualsensor.addWidget(frequency1fix_unit, 4, 2)
        grid_dualsensor.addWidget(frequency2_fix, 5, 0)
        grid_dualsensor.addWidget(self.frequency2fix_edit, 5, 1)
        grid_dualsensor.addWidget(frequency2fix_unit, 5, 2)

        grid_dualsensor.addWidget(self.int_ratio_dualsens_edit, 4, 3)
        grid_dualsensor.addWidget(self.lifetime_phos_dualsens_edit, 5, 3)
        grid_dualsensor.addWidget(lifetime_phos_dualsens_unit, 5, 4)

        grid_dualsensor.addWidget(error_assumed_meas, 8, 0)
        grid_dualsensor.addWidget(self.error_assumed_meas_edit, 8, 1)
        grid_dualsensor.addWidget(error_assumed_meas_unit, 8, 2)

        grid_dualsensor.addWidget(self.load_temp_comp_button, 9, 0)
        grid_dualsensor.addWidget(self.compensation_edit, 9, 1, 2, 4)

        grid_dualsensor.addWidget(self.load_tauP_intP_conv_button, 11, 0)
        grid_dualsensor.addWidget(self.conversion_edit, 11, 1, 2, 4)

        grid_dualsensor.addWidget(self.load_O_T_calib_button, 13, 0)
        grid_dualsensor.addWidget(self.calib_edit, 13, 1, 2, 4)

        # add GroupBox to layout and load buttons in GroupBox
        vbox_left.addWidget(set_dualsensor_group)
        set_dualsensor_group.setLayout(grid_dualsensor)

        set_dualsensor_group.setContentsMargins(2, 5, 2, 2)

        # create GroupBox to structure the layout
        set_dualsensor_group2 = QGroupBox()
        grid_dualsensor2 = QGridLayout()

        grid_dualsensor2.addWidget(intensity_amplitude_measured, 0, 0)
        grid_dualsensor2.addWidget(self.int_ratio_checkbox, 0, 1)
        grid_dualsensor2.addWidget(self.total_amplitude_checkbox, 0, 2)
        grid_dualsensor2.addWidget(table_title_calib, 1, 0, 1, 2)
        grid_dualsensor2.addWidget(self.tableCalibration, 2, 0, 1, 3)

        grid_dualsensor2.addWidget(table_title_meas, 3, 0, 1, 2)
        grid_dualsensor2.addWidget(self.tableINPUT, 4, 0, 1, 3)

        # add GroupBox to layout and load buttons in GroupBox
        vbox_left.addWidget(set_dualsensor_group2)
        set_dualsensor_group2.setLayout(grid_dualsensor2)

        set_dualsensor_group2.setContentsMargins(2, 10, 2, 2)

        # ------------------------------------------------------------------------------------
        # updating lineEdit
        self.frequency1fix_edit.editingFinished.connect(self.print_frequency_fixed_1)
        self.frequency2fix_edit.editingFinished.connect(self.print_frequency_fixed_2)
        self.int_ratio_dualsens_edit.editingFinished.connect(self.print_Intensity_ratio_dualsens)
        self.error_assumed_meas_edit.editingFinished.connect(self.print_error_assumption)

        # connect run button with function
        self.int_ratio_checkbox.clicked.connect(self.amplitude_to_intensity)
        self.total_amplitude_checkbox.clicked.connect(self.intensity_to_amplitude)

        self.simulation_checkbox.clicked.connect(self.simulation_to_evaluation)
        self.measurement_checkbox.clicked.connect(self.evaluation_to_simulation)

        # connect button with function
        self.load_temp_comp_button.clicked.connect(self.open_compensation)
        self.load_tauP_intP_conv_button.clicked.connect(self.open_conversion)
        self.load_O_T_calib_button.clicked.connect(self.open_calibration)

        self.clip = QApplication.clipboard()

    # --------------------------------------------------------------------------------------
    # left part (BOTTOM)
    # --------------------------------------------------------------------------------------
        # create button for updating if required
        self.add_row_button = QPushButton('Add measurement', self)
        self.add_row_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                          "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487;"
                                          "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                          "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")

        self.clear_button = QPushButton('Clear plots', self)
        self.clear_button.setStyleSheet("border-width: 1px; border-color: #a5b6bb; border-style: solid; "
                                        "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                        "padding-right: 5px; min-height: 10px; max-height: 18px;")
        self.save_button = QPushButton('Save', self)
        self.save_button.setStyleSheet("border-width: 1px; border-color: #a5b6bb; border-style: solid; "
                                       "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                       "padding-right: 5px; min-height: 10px; max-height: 18px;")
        self.clear_files_button = QPushButton('Clear files', self)
        self.clear_files_button.setStyleSheet("border-width: 1px; border-color: #a5b6bb; border-style: solid; "
                                              "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                              "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # clear button for calibration and input table
        self.clear_calib_button = QPushButton('Clear calibration', self)
        self.clear_calib_button.setStyleSheet("border-width: 1px; border-color: #a5b6bb; border-style: solid; "
                                              "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                              "padding-right: 5px; min-height: 10px; max-height: 18px;")
        self.clear_input_button = QPushButton('Clear simulation/measurement', self)
        self.clear_input_button.setStyleSheet("border-width: 1px; border-color: #a5b6bb; border-style: solid; "
                                              "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                              "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # -----------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_bottom_group = QGroupBox()
        grid_bottom = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        vbox_left.addWidget(set_bottom_group)
        set_bottom_group.setLayout(grid_bottom)

        grid_bottom.addWidget(self.add_row_button, 2, 0)
        grid_bottom.addWidget(self.save_button, 2, 1)
        grid_bottom.addWidget(self.clear_files_button, 2, 2)
        grid_bottom.addWidget(self.clear_button, 3, 0)
        grid_bottom.addWidget(self.clear_calib_button, 3, 1)
        grid_bottom.addWidget(self.clear_input_button, 3, 2)

        set_bottom_group.setContentsMargins(2, 2, 2, 2)

        # -----------------------------------------------------------------------------------
        # connect LineEdit with function (read_csv and do stuff)
        self.add_row_button.clicked.connect(self.insertRows)
        self.save_button.clicked.connect(self.save_report)
        self.clear_button.clicked.connect(self.clear_all)
        self.clear_calib_button.clicked.connect(self.clear_calib)
        self.clear_input_button.clicked.connect(self.clear_input)
        self.clear_files_button.clicked.connect(self.clear_files)

        # =============================================================================================================
        # Tabs - right part
        # =============================================================================================================
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab6 = QWidget()
        # self.tabs.setMinimumWidth(1200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Simulation τ and Φ")
        self.tabs.addTab(self.tab2, "Simulation I-ratio")
        self.tabs.addTab(self.tab3, "pH/O2")
        self.tabs.addTab(self.tab4, "pH/T")
        self.tabs.addTab(self.tab5, "(C)O2")
        self.tabs.addTab(self.tab6, "O2/T")

        # Add tabs to widget
        vbox_right.addWidget(self.tabs)
        self.setLayout(vbox_right)

        # ==================================================================
        # 1st tab Simulation τ and Φ
        # ==================================================================
        self.tab1.layout = QVBoxLayout(self)
        self.tab1.setLayout(self.tab1.layout)

        tab1_hbox_top = QHBoxLayout()
        tab1_hbox_bottom = QHBoxLayout()
        self.tab1.layout.addLayout(tab1_hbox_top)
        self.tab1.layout.addLayout(tab1_hbox_bottom)

        # -----------------------------------------------------------
        # PLOT - superimposed phase angle (left side bottom)
        self.fig_phaseangle, self.ax_phaseangle = plt.subplots()
        self.canvas_phaseangle = FigureCanvasQTAgg(self.fig_phaseangle)
        self.navi_timedrive = NavigationToolbar2QT(self.canvas_phaseangle, w)

        self.ax_phaseangle.set_xlim(0, 20)
        self.ax_phaseangle.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_phaseangle.set_ylabel('Superimposed phase angle Phi [°]', fontsize=9)
        self.ax_phaseangle.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_phaseangle.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ---------------------
        # PLOT - lifetime
        self.fig_lifetime, self.ax_lifetime = plt.subplots()
        self.canvas_lifetime = FigureCanvasQTAgg(self.fig_lifetime)
        self.navi_lifetime = NavigationToolbar2QT(self.canvas_lifetime, w)

        x = np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))
        self.ax_lifetime.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
        self.ax_lifetime.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_lifetime.set_ylabel('Lifetime tau [µs]', fontsize=9)
        self.ax_lifetime.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_lifetime.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ---------------------
        # Message box
        self.message = QTextEdit(self)
        self.message.setReadOnly(True)

        # ---------------------
        # Run simulation botton
        self.run_sim_button = QPushButton('Run simulation', self)
        self.run_sim_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                          "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487; "
                                          "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                          "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - relative error life time
        self.fig_lifetime_err, self.ax_lifetime_er = plt.subplots()
        self.canvas_lifetime_err = FigureCanvasQTAgg(self.fig_lifetime_err)
        self.navi_lifetime_err = NavigationToolbar2QT(self.canvas_lifetime_err, w)

        self.ax_lifetime_er.set_xlim(0, 20)
        self.ax_lifetime_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_lifetime_er.set_ylabel('Rel. error rate [%]', fontsize=9)
        self.ax_lifetime_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_lifetime_err.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        # left part of tab1 - top
        set_phaseangle_group = QGroupBox("Superimposed phase angle Φ [°]")
        set_phaseangle_group.setMinimumHeight(200)
        set_phaseangle_group.setMinimumWidth(400)
        grid_phaseangle = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab1_hbox_top.addWidget(set_phaseangle_group)
        set_phaseangle_group.setLayout(grid_phaseangle)

        grid_phaseangle.addWidget(self.canvas_phaseangle)
        grid_phaseangle.addWidget(self.navi_timedrive)

        # left part of tab1 - bottom
        set_message_group = QGroupBox("Report messages")
        grid_message = QGridLayout()
        set_message_group.setMinimumHeight(150)
        set_message_group.setMinimumWidth(400)

        # add GroupBox to layout and load buttons in GroupBox
        tab1_hbox_bottom.addWidget(set_message_group)
        set_message_group.setLayout(grid_message)

        grid_message.addWidget(self.message, 2, 0)
        grid_message.addWidget(self.run_sim_button, 3, 0)

        # ----------------------------------
        # right part of tab1
        set_lifetime_group = QGroupBox("Lifetime phosphor τ [µs]")
        set_lifetime_group.setMinimumHeight(200)
        set_lifetime_group.setMinimumWidth(420)
        grid_lifetime = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab1_hbox_top.addWidget(set_lifetime_group)
        set_lifetime_group.setLayout(grid_lifetime)

        grid_lifetime.addWidget(self.canvas_lifetime)
        grid_lifetime.addWidget(self.navi_lifetime)

        # right part tab1 bottom
        set_lifetime_er_group = QGroupBox("Rel. error for the lifetime [%]")
        set_lifetime_er_group.setMinimumHeight(150)
        set_lifetime_er_group.setMinimumWidth(400)
        grid_lifetime_er = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab1_hbox_bottom.addWidget(set_lifetime_er_group)
        set_lifetime_er_group.setLayout(grid_lifetime_er)

        grid_lifetime_er.addWidget(self.canvas_lifetime_err)
        grid_lifetime_er.addWidget(self.navi_lifetime_err)

        set_lifetime_er_group.setContentsMargins(2, 2, 2, 2)

        # ----------------------------------------------------------------------
        # connect run button with function
        self.run_sim_button.clicked.connect(self.error_propagation)

        # ==================================================================
        # 2nd tab Simulation I-ratio
        # ==================================================================
        # Create first tab
        self.tab2.layout = QVBoxLayout(self)
        self.tab2.setLayout(self.tab2.layout)

        tab2_hbox_top = QHBoxLayout()
        tab2_hbox_bottom = QHBoxLayout()
        self.tab2.layout.addLayout(tab2_hbox_top)
        self.tab2.layout.addLayout(tab2_hbox_bottom)

        # -----------------------------------------------------------
        # PLOT - superimposed phase angle (left side bottom)
        self.fig_intensity, self.ax_intensity = plt.subplots()
        self.canvas_intensity = FigureCanvasQTAgg(self.fig_intensity)
        self.navi_intensity = NavigationToolbar2QT(self.canvas_intensity, w)

        self.ax_intensity.set_xlim(0, 20)
        self.ax_intensity.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity.set_ylabel('Intensity ratio', fontsize=9)
        self.ax_intensity.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ---------------------
        # PLOT - lifetime
        self.fig_intensity_abs, self.ax_intensity_abs = plt.subplots()
        self.canvas_intensity_abs = FigureCanvasQTAgg(self.fig_intensity_abs)
        self.navi_intensity_abs = NavigationToolbar2QT(self.canvas_intensity_abs, w)

        x = np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))
        self.ax_intensity_abs.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
        self.ax_intensity_abs.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity_abs.set_ylabel('Abs. deviation intensity ratio', fontsize=9)
        self.ax_intensity_abs.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity_abs.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ---------------------
        # Message box
        self.message_int = QTextEdit(self)
        self.message_int.setReadOnly(True)

        # ---------------------
        # Run simulation button
        self.run_sim_button2 = QPushButton('Run simulation', self)
        self.run_sim_button2.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                          "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487; "
                                          "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                          "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - relative error life time
        self.fig_intensity_er, self.ax_intensity_er = plt.subplots()
        self.canvas_intensity_err = FigureCanvasQTAgg(self.fig_intensity_er)
        self.navi_intensity_err = NavigationToolbar2QT(self.canvas_intensity_err, w)

        self.ax_intensity_er.set_xlim(0, 20)
        self.ax_intensity_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity_er.set_ylabel('Rel. error rate [%]', fontsize=9)
        self.ax_intensity_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity_er.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        # left part of tab1 - top
        set_intensity_group = QGroupBox("Intensity ratio")
        set_intensity_group.setMinimumHeight(200)
        set_intensity_group.setMinimumWidth(400)
        grid_intensity = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab2_hbox_top.addWidget(set_intensity_group)
        set_intensity_group.setLayout(grid_intensity)

        grid_intensity.addWidget(self.canvas_intensity)
        grid_intensity.addWidget(self.navi_intensity)

        # add GroupBox to layout and load buttons in GroupBox
        set_message_int_group = QGroupBox("Report messages")
        grid_message_int = QGridLayout()
        set_message_int_group.setMinimumHeight(150)
        set_message_int_group.setMinimumWidth(400)

        # add GroupBox to layout and load buttons in GroupBox
        tab2_hbox_bottom.addWidget(set_message_int_group)
        set_message_int_group.setLayout(grid_message_int)

        grid_message_int.addWidget(self.message_int, 2, 0)
        grid_message_int.addWidget(self.run_sim_button2, 3, 0)
        # ----------------------------------
        # right part of tab1
        set_intensity_abs_group = QGroupBox("Abs. deviation intensity ratio")
        set_intensity_abs_group.setMinimumHeight(200)
        set_intensity_abs_group.setMinimumWidth(420)
        grid_intensity_abs = QGridLayout()

        tab2_hbox_top.addWidget(set_intensity_abs_group)
        set_intensity_abs_group.setLayout(grid_intensity_abs)

        grid_intensity_abs.addWidget(self.canvas_intensity_abs)
        grid_intensity_abs.addWidget(self.navi_intensity_abs)

        # right part tab1 bottom
        set_intensity_er_group = QGroupBox("Rel. error for the intensity ratio [%]")
        set_intensity_er_group.setMinimumHeight(150)
        set_intensity_er_group.setMinimumWidth(400)
        grid_intensity_er = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab2_hbox_bottom.addWidget(set_intensity_er_group)
        set_intensity_er_group.setLayout(grid_intensity_er)

        grid_intensity_er.addWidget(self.canvas_intensity_err)
        grid_intensity_er.addWidget(self.navi_intensity_err)

        set_intensity_er_group.setContentsMargins(2, 2, 2, 2)

        # ----------------------------------------------------------------------
        # connect run button with function
        self.run_sim_button2.clicked.connect(self.error_propagation)

        # ==================================================================
        # 3rd tab pH/O2
        # ==================================================================
        # Create 3rd tab
        self.tab3.layout = QVBoxLayout(self)
        self.tab3.setLayout(self.tab3.layout)

        # split in top - bottom
        tab3_vbox_top = QHBoxLayout()
        tab3_vbox_bottom = QHBoxLayout()
        self.tab3.layout.addLayout(tab3_vbox_top)
        self.tab3.layout.addLayout(tab3_vbox_bottom)

        # split top part into left and right
        tab3_vbox_middle = QVBoxLayout()
        tab3_vbox_right = QVBoxLayout()
        tab3_vbox_top.addLayout(tab3_vbox_middle)
        tab3_vbox_top.addLayout(tab3_vbox_right)

        # split bottom part into left, middle and right
        tab3_vbox_run = QVBoxLayout()
        tab3_vbox_input = QVBoxLayout()
        tab3_vbox_output = QVBoxLayout()
        tab3_vbox_bottom.addLayout(tab3_vbox_run)
        tab3_vbox_bottom.addLayout(tab3_vbox_input)
        tab3_vbox_bottom.addLayout(tab3_vbox_output)

        # -----------------------------------------------------------
        # Input grid
        # pH sensing
        slope_tab3 = QLabel(self)
        slope_tab3.setText('pH - slope')
        self.slope_tab3_edit = QLineEdit(self)
        self.slope_tab3_edit.setValidator(QDoubleValidator())
        self.slope_tab3_edit.setMaximumWidth(50)
        self.slope_tab3_edit.setText('1.0')
        self.slope_tab3_edit.setAlignment(Qt.AlignRight)

        pka_tab3 = QLabel(self)
        pka_tab3.setText('pH - pKa')
        self.pka_tab3_edit = QLineEdit(self)
        self.pka_tab3_edit.setValidator(QDoubleValidator())
        self.pka_tab3_edit.setMaximumWidth(50)
        self.pka_tab3_edit.setText('7.4')
        self.pka_tab3_edit.setAlignment(Qt.AlignRight)

        # O2 sensing
        Ksv1_tab3 = QLabel(self)
        Ksv1_tab3.setText('O2 - Ksv1')
        self.Ksv1_tab3_edit = QLineEdit(self)
        self.Ksv1_tab3_edit.setValidator(QDoubleValidator())
        self.Ksv1_tab3_edit.setMaximumWidth(50)
        self.Ksv1_tab3_edit.setText('0.0339')
        self.Ksv1_tab3_edit.setAlignment(Qt.AlignRight)

        Ksv2_tab3 = QLabel(self)
        Ksv2_tab3.setText('O2 - prop factor Ksv2')
        self.Ksv2_tab3_edit = QLineEdit(self)
        self.Ksv2_tab3_edit.setValidator(QDoubleValidator())
        self.Ksv2_tab3_edit.setMaximumWidth(50)
        self.Ksv2_tab3_edit.setText('0.106')
        self.Ksv2_tab3_edit.setAlignment(Qt.AlignRight)

        curv_O2_tab3 = QLabel(self)
        curv_O2_tab3.setText('O2 - curvature f')
        self.curv_O2_tab3_edit = QLineEdit(self)
        self.curv_O2_tab3_edit.setValidator(QDoubleValidator())
        self.curv_O2_tab3_edit.setMaximumWidth(50)
        self.curv_O2_tab3_edit.setText('0.8')
        self.curv_O2_tab3_edit.setAlignment(Qt.AlignRight)

        # ---------------------
        # Message box
        self.message_tab3 = QTextEdit(self)
        self.message_tab3.setReadOnly(True)
        self.message_tab3.setMinimumHeight(5)
        self.message_tab3.setMinimumWidth(10)

        # ---------------------
        # update or run button
        self.run_tab3_button = QPushButton('Run pH/O2', self)
        self.run_tab3_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                          "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487; "
                                          "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                          "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")

        self.progress_tab3 = QProgressBar(self)
        self.progress_tab3.setStyleSheet("border-width: 1px; border-color: #077487; border-style: solid; "
                                         "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                         "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs pH
        self.fig_pH_calib, self.ax_pH_calib = plt.subplots()
        self.canvas_pH_calib = FigureCanvasQTAgg(self.fig_pH_calib)
        self.navi_pH_calib = NavigationToolbar2QT(self.canvas_pH_calib, w)

        self.ax_pH_calib.set_xlim(0, 15)
        self.ax_pH_calib.set_ylim(-0.5, 105)
        self.ax_pH_calib.set_xlabel('pH', fontsize=9)
        self.ax_pH_calib.set_ylabel('Rel. intensity I$_F$ [%]', fontsize=9)
        self.ax_pH_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - Dualsensor - pH calculation
        self.fig_pH, self.ax_pH = plt.subplots()
        self.canvas_pH = FigureCanvasQTAgg(self.fig_pH)
        self.navi_pH = NavigationToolbar2QT(self.canvas_pH, w)

        self.ax_pH.set_xlim(0, 15)
        self.ax_pH.set_ylim(-0.5, 105)
        self.ax_pH.set_xlabel('pH', fontsize=9)
        self.ax_pH.set_ylabel('Rel. intensity I$_F$ [%]', fontsize=9)
        self.ax_pH.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs pO2
        self.fig_pO2_calib, self.ax_pO2_calib = plt.subplots()
        self.canvas_pO2_calib = FigureCanvasQTAgg(self.fig_pO2_calib)
        self.navi_pO2_calib = NavigationToolbar2QT(self.canvas_pO2_calib, w)

        self.ax_pO2_calib.set_xlim(0, 100)
        self.ax_pO2_calib.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2_calib.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
        self.ax_pO2_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - Dualsensor - pO2 calculation
        self.fig_pO2, self.ax_pO2 = plt.subplots()
        self.canvas_pO2 = FigureCanvasQTAgg(self.fig_pO2)
        self.navi_pO2 = NavigationToolbar2QT(self.canvas_pO2, w)

        self.ax_pO2.set_xlim(0, 100)
        self.ax_pO2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
        self.ax_pO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_tab3_run_group = QGroupBox()
        set_tab3_run_group.setMinimumHeight(100)
        set_tab3_run_group.setStyleSheet("QGroupBox { border: 1px solid white;}")
        grid_run_tab3 = QGridLayout()

        set_input_group = QGroupBox("Input")
        set_input_group.setMinimumHeight(120)
        set_input_group.setMinimumWidth(150)
        grid_input = QGridLayout()

        set_output_group = QGroupBox("Output")
        set_output_group.setMinimumHeight(80)
        set_output_group.setMinimumWidth(200)

        grid_output = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab3_vbox_run.addWidget(set_tab3_run_group)
        set_tab3_run_group.setLayout(grid_run_tab3)
        tab3_vbox_input.addWidget(set_input_group)
        set_input_group.setLayout(grid_input)
        tab3_vbox_output.addWidget(set_output_group)
        set_output_group.setLayout(grid_output)

        # run button
        grid_run_tab3.addWidget(self.progress_tab3)
        grid_run_tab3.addWidget(self.run_tab3_button)
        # input
        grid_input.addWidget(slope_tab3, 1, 0)
        grid_input.addWidget(self.slope_tab3_edit, 1, 1)
        grid_input.addWidget(pka_tab3, 2, 0)
        grid_input.addWidget(self.pka_tab3_edit, 2, 1)
        grid_input.addWidget(Ksv1_tab3, 1, 3)
        grid_input.addWidget(self.Ksv1_tab3_edit, 1, 4)
        grid_input.addWidget(Ksv2_tab3, 2, 3)
        grid_input.addWidget(self.Ksv2_tab3_edit, 2, 4)
        grid_input.addWidget(curv_O2_tab3, 3, 3)
        grid_input.addWidget(self.curv_O2_tab3_edit, 3, 4)

        # output results
        grid_output.addWidget(self.message_tab3, 1, 2)

        # ----------------------------------
        # create GroupBox to structure the layout
        # left part of tab1 - top
        set_pHcalib_group = QGroupBox("Boltzmann fit for calibration")
        set_pHcalib_group.setMinimumHeight(150)
        grid_pHcalib = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab3_vbox_middle.addWidget(set_pHcalib_group)
        set_pHcalib_group.setLayout(grid_pHcalib)

        grid_pHcalib.addWidget(self.canvas_pH_calib)
        grid_pHcalib.addWidget(self.navi_pH_calib)

        # left part of tab1 - bottom
        set_pH_group = QGroupBox("pH sensing")
        set_pH_group.setMinimumHeight(150)
        grid_pH = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab3_vbox_middle.addWidget(set_pH_group)
        set_pH_group.setLayout(grid_pH)

        grid_pH.addWidget(self.canvas_pH)
        grid_pH.addWidget(self.navi_pH)

        # ----------------------------------
        # right part of tab1
        set_pO2calib_group = QGroupBox("Two-site-model for calibration")
        set_pO2calib_group.setMinimumHeight(150)
        grid_pO2calib = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab3_vbox_right.addWidget(set_pO2calib_group)
        set_pO2calib_group.setLayout(grid_pO2calib)

        grid_pO2calib.addWidget(self.canvas_pO2_calib)
        grid_pO2calib.addWidget(self.navi_pO2_calib)
        set_pO2calib_group.setContentsMargins(2, 10, 2, 2)

        # right part tab1 bottom
        set_pO2_group = QGroupBox("O2 sensing")
        set_pO2_group.setMinimumHeight(150)
        grid_pO2 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab3_vbox_right.addWidget(set_pO2_group)
        set_pO2_group.setLayout(grid_pO2)

        grid_pO2.addWidget(self.canvas_pO2)
        grid_pO2.addWidget(self.navi_pO2)
        set_pO2_group.setContentsMargins(2, 2, 2, 2)

        # ------------------------------------------------------------------------------------
        # connect LineEdit with function for updating
        self.slope_tab3_edit.editingFinished.connect(self.print_tab3_slope)
        self.pka_tab3_edit.editingFinished.connect(self.print_tab3_pKa)
        self.Ksv1_tab3_edit.editingFinished.connect(self.print_tab3_Ksv1)
        self.Ksv2_tab3_edit.editingFinished.connect(self.print_tab3_Ksv2)
        self.curv_O2_tab3_edit.editingFinished.connect(self.print_tab3_curv)

        # connect run button with function
        self.run_tab3_button.clicked.connect(self.pH_oxygen_sensing)

        # ==================================================================
        # 4th tab pH/T
        # ==================================================================
        # Create 4th tab
        self.tab4.layout = QVBoxLayout(self)
        self.tab4.setLayout(self.tab4.layout)

        tab4_vbox_top = QHBoxLayout()
        tab4_vbox_bottom = QHBoxLayout()
        self.tab4.layout.addLayout(tab4_vbox_top)
        self.tab4.layout.addLayout(tab4_vbox_bottom)

        # split top part into left and right
        tab4_vbox_middle = QVBoxLayout()
        tab4_vbox_right = QVBoxLayout()
        tab4_vbox_top.addLayout(tab4_vbox_middle)
        tab4_vbox_top.addLayout(tab4_vbox_right)

        # split bottom part into left, middle and right
        tab4_vbox_run = QVBoxLayout()
        tab4_vbox_input = QVBoxLayout()
        tab4_vbox_output = QVBoxLayout()
        tab4_vbox_bottom.addLayout(tab4_vbox_run)
        tab4_vbox_bottom.addLayout(tab4_vbox_input)
        tab4_vbox_bottom.addLayout(tab4_vbox_output)

        # -----------------------------------------------------------
        # Input grid
        self.tab4_fit_slope = QCheckBox('Slope', self)
        self.tab4_fit_bottom = QCheckBox('Bottom', self)
        self.tab4_fit_top = QCheckBox('Top', self)
        self.tab4_fit_v50 = QCheckBox('V50', self)
        self.tab4_fit_bottom.toggle()
        self.tab4_fit_top.toggle()
        self.tab4_fit_v50.toggle()

        # ---------------------
        # Message box
        self.message_tab4 = QTextEdit(self)
        self.message_tab4.setReadOnly(True)
        self.message_tab4.setMinimumHeight(5)
        self.message_tab4.setMinimumWidth(10)

        # ---------------------
        # update or run button
        self.run_tab4_button = QPushButton('Run pH/T', self)
        self.run_tab4_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                           "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487; "
                                           "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                           "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")

        self.progress_tab4 = QProgressBar(self)
        self.progress_tab4.setStyleSheet("border-width: 1px; border-color: #077487; border-style: solid; "
                                         "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                         "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs pH
        self.fig_pH_calib2, self.ax_pH_calib2 = plt.subplots()
        self.ax_pH_calib2_mir = self.ax_pH_calib2.twinx()
        self.canvas_pH_calib2 = FigureCanvasQTAgg(self.fig_pH_calib2)
        self.navi_pH_calib2 = NavigationToolbar2QT(self.canvas_pH_calib2, w)

        self.ax_pH_calib2.set_xlim(0, 15)
        self.ax_pH_calib2.set_xlabel('pH', fontsize=9)
        self.ax_pH_calib2.set_ylabel('cot(Φ)', fontsize=9)
        self.ax_pH_calib2_mir.set_ylabel('Φ [deg]', fontsize=9)
        self.ax_pH_calib2.tick_params(axis='both', which='both', direction='in', top=True)
        self.ax_pH_calib2_mir.tick_params(axis='both', which='both', direction='in')
        self.fig_pH_calib2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - Dualsensor - pH calculation
        self.fig_pH2, self.ax_pH2 = plt.subplots()
        self.ax_pH2_mir = self.ax_pH2.twinx()
        self.canvas_pH2 = FigureCanvasQTAgg(self.fig_pH2)
        self.navi_pH2 = NavigationToolbar2QT(self.canvas_pH2, w)

        self.ax_pH2.set_xlim(0, 15)
        self.ax_pH2.set_xlabel('pH', fontsize=9)
        self.ax_pH2.set_ylabel('cot(Φ)', fontsize=9)
        self.ax_pH2_mir.set_ylabel('Φ [deg]', fontsize=9)
        self.ax_pH2.tick_params(axis='both', which='both', direction='in', top=True)
        self.ax_pH2_mir.tick_params(axis='both', which='both', direction='in')
        self.fig_pH2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs pO2
        self.fig_temp_calib, self.ax_temp_calib = plt.subplots(nrows=2, ncols=2, sharex=True)
        self.canvas_temp_calib = FigureCanvasQTAgg(self.fig_temp_calib)
        self.navi_temp_calib = NavigationToolbar2QT(self.canvas_temp_calib, w)

        self.ax_temp_calib[1][0].set_xlim(0, 50)
        self.ax_temp_calib[1][1].set_xlim(0, 50)
        self.ax_temp_calib[1][0].set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp_calib[1][1].set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp_calib[0][0].set_ylabel('slope', fontsize=9)
        self.ax_temp_calib[0][1].set_ylabel('bottom', fontsize=9)
        self.ax_temp_calib[1][0].set_ylabel('pka', fontsize=9)
        self.ax_temp_calib[1][1].set_ylabel('top', fontsize=9)
        self.ax_temp_calib[0][0].tick_params(axis='both', which='both', labelsize=7)
        self.ax_temp_calib[1][0].tick_params(axis='both', which='both', labelsize=7)
        self.ax_temp_calib[0][1].tick_params(axis='both', which='both', labelsize=7)
        self.ax_temp_calib[1][1].tick_params(axis='both', which='both', labelsize=7)
        self.ax_temp_calib[0][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.ax_temp_calib[0][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.ax_temp_calib[1][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.ax_temp_calib[1][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_temp_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98, wspace=.3, hspace=.2)

        # ---------------------
        # PLOT - Dualsensor - pO2 calculation
        self.fig_temp, self.ax_temp = plt.subplots()
        self.canvas_temp = FigureCanvasQTAgg(self.fig_temp)
        self.navi_temp = NavigationToolbar2QT(self.canvas_temp, w)

        self.ax_temp.set_xlim(0, 50)
        self.ax_temp.set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp.set_ylabel('$τ_P$ [µs]', fontsize=9)
        self.ax_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_temp.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_tab4_run_group = QGroupBox()
        set_tab4_run_group.setMinimumHeight(100)
        set_tab4_run_group.setStyleSheet("QGroupBox { border: 1px solid white;}")
        grid_run_tab4 = QGridLayout()

        set_input_group_tab4 = QGroupBox("Fitting parameter")
        set_input_group_tab4.setMinimumHeight(120)
        set_input_group_tab4.setMinimumWidth(150)
        grid_input_tab4 = QGridLayout()

        set_tab4_report_group = QGroupBox("Output")
        set_tab4_report_group.setMinimumHeight(80)
        grid_tab4_report = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab4_vbox_run.addWidget(set_tab4_run_group)
        set_tab4_run_group.setLayout(grid_run_tab4)
        tab4_vbox_input.addWidget(set_input_group_tab4)
        set_input_group_tab4.setLayout(grid_input_tab4)
        tab4_vbox_output.addWidget(set_tab4_report_group)
        set_tab4_report_group.setLayout(grid_tab4_report)

        # run button
        grid_run_tab4.addWidget(self.progress_tab4)
        grid_run_tab4.addWidget(self.run_tab4_button)
        # input
        grid_input_tab4.addWidget(self.tab4_fit_bottom, 1, 1)
        grid_input_tab4.addWidget(self.tab4_fit_top, 2, 1)
        grid_input_tab4.addWidget(self.tab4_fit_v50, 2, 0)
        grid_input_tab4.addWidget(self.tab4_fit_slope, 1, 0)
        # output results
        grid_tab4_report.addWidget(self.message_tab4, 1, 2)

        # ----------------------------------
        # create GroupBox to structure the layout
        # left part of tab1 - top
        set_pH2calib_group = QGroupBox("pH calibration")
        set_pH2calib_group.setMinimumHeight(150)
        grid_pH2calib = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab4_vbox_middle.addWidget(set_pH2calib_group)
        set_pH2calib_group.setLayout(grid_pH2calib)

        grid_pH2calib.addWidget(self.canvas_pH_calib2)
        grid_pH2calib.addWidget(self.navi_pH_calib2)

        # left part of tab4 - bottom
        set_pH2_group = QGroupBox("Boltzmann fit")
        set_pH2_group.setMinimumHeight(150)
        grid_pH2 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab4_vbox_middle.addWidget(set_pH2_group)
        set_pH2_group.setLayout(grid_pH2)

        grid_pH2.addWidget(self.canvas_pH2)
        grid_pH2.addWidget(self.navi_pH2)

        # ----------------------------------
        # right part of tab4
        set_temp_calib_group = QGroupBox("Temperature compensation pH sensing")
        set_temp_calib_group.setMinimumHeight(150)
        grid_tempcalib = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab4_vbox_right.addWidget(set_temp_calib_group)
        set_temp_calib_group.setLayout(grid_tempcalib)

        grid_tempcalib.addWidget(self.canvas_temp_calib)
        grid_tempcalib.addWidget(self.navi_temp_calib)
        set_temp_calib_group.setContentsMargins(2, 10, 2, 2)

        # right part tab4 bottom
        set_temp_group = QGroupBox("Arrhenius equation")
        set_temp_group.setMinimumHeight(150)
        grid_temp = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab4_vbox_right.addWidget(set_temp_group)
        set_temp_group.setLayout(grid_temp)

        grid_temp.addWidget(self.canvas_temp)
        grid_temp.addWidget(self.navi_temp)
        set_temp_group.setContentsMargins(2, 2, 2, 2)

        # ------------------------------------------------------------------------------------
        # connect run button with function
        self.run_tab4_button.clicked.connect(self.pH_temp_sensing)

        # ==================================================================
        # 5th tab (C)O2
        # ==================================================================
        # Create 5th tab
        self.tab5.layout = QVBoxLayout(self)
        self.tab5.setLayout(self.tab5.layout)

        tab5_vbox_top = QHBoxLayout()
        tab5_vbox_bottom = QHBoxLayout()
        self.tab5.layout.addLayout(tab5_vbox_top)
        self.tab5.layout.addLayout(tab5_vbox_bottom)

        # split top part into left and right
        tab5_vbox_middle = QVBoxLayout()
        tab5_vbox_right = QVBoxLayout()
        tab5_vbox_top.addLayout(tab5_vbox_middle)
        tab5_vbox_top.addLayout(tab5_vbox_right)

        # split bottom part into left, middle and right
        tab5_vbox_run = QVBoxLayout()
        tab5_vbox_input = QVBoxLayout()
        tab5_vbox_output = QVBoxLayout()
        tab5_vbox_bottom.addLayout(tab5_vbox_run)
        tab5_vbox_bottom.addLayout(tab5_vbox_input)
        tab5_vbox_bottom.addLayout(tab5_vbox_output)

        # -----------------------------------------------------------
        # Input grid
        # pCO2 sensing
        Ksv1_tab5_CO2 = QLabel(self)
        Ksv1_tab5_CO2.setText('CO2 - Ksv1')
        self.Ksv1_CO2_tab5_edit = QLineEdit(self)
        self.Ksv1_CO2_tab5_edit.setValidator(QDoubleValidator())
        self.Ksv1_CO2_tab5_edit.setMaximumWidth(50)
        self.Ksv1_CO2_tab5_edit.setText('0.13')
        self.Ksv1_CO2_tab5_edit.setAlignment(Qt.AlignRight)

        Ksv2_CO2_tab5 = QLabel(self)
        Ksv2_CO2_tab5.setText('CO2 - prop factor')
        self.Ksv2_CO2_tab5_edit = QLineEdit(self)
        self.Ksv2_CO2_tab5_edit.setValidator(QDoubleValidator())
        self.Ksv2_CO2_tab5_edit.setMaximumWidth(50)
        self.Ksv2_CO2_tab5_edit.setText('1.')
        self.Ksv2_CO2_tab5_edit.setAlignment(Qt.AlignRight)

        curv_CO2_tab5 = QLabel(self)
        curv_CO2_tab5.setText('CO2 - curvature f')
        self.curv_CO2_tab5_edit = QLineEdit(self)
        self.curv_CO2_tab5_edit.setValidator(QDoubleValidator())
        self.curv_CO2_tab5_edit.setMaximumWidth(50)
        self.curv_CO2_tab5_edit.setText('0.8')
        self.curv_CO2_tab5_edit.setAlignment(Qt.AlignRight)

        # pO2 sensing
        # What ought to be calibrated for O2 sensing - Ksv or tauP
        calibration_type = QLabel(self)
        calibration_type.setText('Optimization')
        self.calib_ksv_checkbox = QCheckBox('Ksv', self)
        self.calib_ksv_checkbox.toggle()
        self.calib_tauP_checkbox = QCheckBox('tauP', self)

        Ksv1_tab5_O2 = QLabel(self)
        Ksv1_tab5_O2.setText('O2 - Ksv1')
        self.Ksv1_O2_tab5_edit = QLineEdit(self)
        self.Ksv1_O2_tab5_edit.setValidator(QDoubleValidator())
        self.Ksv1_O2_tab5_edit.setMaximumWidth(50)
        self.Ksv1_O2_tab5_edit.setText('0.019')
        self.Ksv1_O2_tab5_edit.setAlignment(Qt.AlignRight)

        Ksv2_O2_tab5 = QLabel(self)
        Ksv2_O2_tab5.setText('O2 - prop factor')
        self.Ksv2_O2_tab5_edit = QLineEdit(self)
        self.Ksv2_O2_tab5_edit.setValidator(QDoubleValidator())
        self.Ksv2_O2_tab5_edit.setMaximumWidth(50)
        self.Ksv2_O2_tab5_edit.setText('0.125')
        self.Ksv2_O2_tab5_edit.setAlignment(Qt.AlignRight)

        curv_O2_tab5 = QLabel(self)
        curv_O2_tab5.setText('O2 - curvature f')
        self.curv_O2_tab5_edit = QLineEdit(self)
        self.curv_O2_tab5_edit.setValidator(QDoubleValidator())
        self.curv_O2_tab5_edit.setMaximumWidth(50)
        self.curv_O2_tab5_edit.setText('0.85')
        self.curv_O2_tab5_edit.setAlignment(Qt.AlignRight)

        # ---------------------
        # Message box
        self.message_tab5 = QTextEdit(self)
        self.message_tab5.setReadOnly(True)
        self.message_tab5.setMinimumHeight(5)
        self.message_tab5.setMinimumWidth(10)

        # ---------------------
        # update or run button
        self.run_tab5_button = QPushButton('Run (C)O2', self)
        self.run_tab5_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, "
                                           "y2: 1, stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; "
                                           "border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
                                           "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; "
                                           "max-height: 18px;")
        self.progress_tab5 = QProgressBar(self)
        self.progress_tab5.setStyleSheet("border-width: 1px; border-color: #077487; border-style: solid; "
                                         "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                         "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs CO2
        self.fig_CO2calib, self.ax_CO2calib = plt.subplots(figsize=(5, 4))
        self.canvas_CO2calib = FigureCanvasQTAgg(self.fig_CO2calib)
        self.navi_CO2calib = NavigationToolbar2QT(self.canvas_CO2calib, w)

        self.ax_CO2calib.set_xlim(0, 100)
        self.ax_CO2calib.set_ylim(0, 100)
        self.ax_CO2calib.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
        self.ax_CO2calib.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
        self.ax_CO2calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_CO2calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - Dualsensor - CO2 calculation
        self.fig_CO2, self.ax_CO2 = plt.subplots(figsize=(5, 4))
        self.canvas_CO2 = FigureCanvasQTAgg(self.fig_CO2)
        self.navi_CO2 = NavigationToolbar2QT(self.canvas_CO2, w)

        self.ax_CO2.set_xlim(0, 100)
        self.ax_CO2.set_ylim(0, 100)
        self.ax_CO2.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
        self.ax_CO2.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
        self.ax_CO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_CO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - CALIBRATION - Intensity vs pO2
        self.fig_pO2calib2, self.ax_pO2calib_2 = plt.subplots(figsize=(5, 4))
        self.canvas_pO2calib2 = FigureCanvasQTAgg(self.fig_pO2calib2)
        self.navi_pO2calib2 = NavigationToolbar2QT(self.canvas_pO2calib2, w)

        self.ax_pO2calib_2.set_xlim(0, 100)
        self.ax_pO2calib_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2calib_2.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
        self.ax_pO2calib_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2calib2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ---------------------
        # PLOT - Dualsensor - pO2 calculation
        self.fig_pO2_2, self.ax_pO2_2 = plt.subplots(figsize=(5, 4))
        self.canvas_pO2_2 = FigureCanvasQTAgg(self.fig_pO2_2)
        self.navi_pO2_2 = NavigationToolbar2QT(self.canvas_pO2_2, w)

        self.ax_pO2_2.set_xlim(0, 100)
        self.ax_pO2_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2_2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
        self.ax_pO2_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2_2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_tab5_run_group = QGroupBox()
        set_tab5_run_group.setMinimumHeight(100)
        set_tab5_run_group.setStyleSheet("QGroupBox { border: 1px solid white;}")
        grid_run_tab5 = QGridLayout()

        set_input_group5 = QGroupBox("Input")
        set_input_group5.setMinimumHeight(120)
        set_input_group5.setMinimumWidth(200)
        grid_input5 = QGridLayout()

        set_output_group5 = QGroupBox("Output")
        set_output_group5.setMinimumHeight(80)
        grid_output5 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab5_vbox_run.addWidget(set_tab5_run_group)
        set_tab5_run_group.setLayout(grid_run_tab5)
        tab5_vbox_input.addWidget(set_input_group5)
        set_input_group5.setLayout(grid_input5)
        tab5_vbox_output.addWidget(set_output_group5)
        set_output_group5.setLayout(grid_output5)

        # run button
        grid_run_tab5.addWidget(self.progress_tab5)
        grid_run_tab5.addWidget(self.run_tab5_button)
        # input
        grid_input5.addWidget(Ksv1_tab5_CO2, 1, 0)
        grid_input5.addWidget(self.Ksv1_CO2_tab5_edit, 1, 2)
        grid_input5.addWidget(Ksv2_CO2_tab5, 2, 0)
        grid_input5.addWidget(self.Ksv2_CO2_tab5_edit, 2, 2)
        grid_input5.addWidget(curv_CO2_tab5, 3, 0)
        grid_input5.addWidget(self.curv_CO2_tab5_edit, 3, 2)

        grid_input5.addWidget(Ksv1_tab5_O2, 1, 3)
        grid_input5.addWidget(self.Ksv1_O2_tab5_edit, 1, 4)
        grid_input5.addWidget(Ksv2_O2_tab5, 2, 3)
        grid_input5.addWidget(self.Ksv2_O2_tab5_edit, 2, 4)
        grid_input5.addWidget(curv_O2_tab5, 3, 3)
        grid_input5.addWidget(self.curv_O2_tab5_edit, 3, 4)
        grid_input5.addWidget(calibration_type, 4, 3)
        grid_input5.addWidget(self.calib_ksv_checkbox, 4, 4)
        grid_input5.addWidget(self.calib_tauP_checkbox, 4, 5)

        # output results
        grid_output5.addWidget(self.message_tab5, 1, 2)

        # ----------------------------------
        # create GroupBox to structure the layout
        # left part of tab5 - top
        set_CO2calib_group = QGroupBox("Two-site-model (intensity) for calibration")
        set_CO2calib_group.setMinimumHeight(150)
        grid_CO2calib = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab5_vbox_middle.addWidget(set_CO2calib_group)
        set_CO2calib_group.setLayout(grid_CO2calib)

        grid_CO2calib.addWidget(self.canvas_CO2calib)
        grid_CO2calib.addWidget(self.navi_CO2calib)

        # left part of tab5 - bottom
        set_CO2_group = QGroupBox("CO2 sensing")
        set_CO2_group.setMinimumHeight(150)
        grid_CO2 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab5_vbox_middle.addWidget(set_CO2_group)
        set_CO2_group.setLayout(grid_CO2)

        grid_CO2.addWidget(self.canvas_CO2)
        grid_CO2.addWidget(self.navi_CO2)

        # ----------------------------------
        # right part of tab5
        set_pO2calib2_group = QGroupBox("Two-site-model for calibration")
        set_pO2calib2_group.setMinimumHeight(150)
        grid_pO2calib2 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab5_vbox_right.addWidget(set_pO2calib2_group)
        set_pO2calib2_group.setLayout(grid_pO2calib2)

        grid_pO2calib2.addWidget(self.canvas_pO2calib2)
        grid_pO2calib2.addWidget(self.navi_pO2calib2)
        set_pO2calib2_group.setContentsMargins(2, 10, 2, 2)

        # right part tab1 bottom
        set_pO2_group2 = QGroupBox("O2 sensing")
        set_pO2_group2.setMinimumHeight(150)
        grid_pO2_2 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab5_vbox_right.addWidget(set_pO2_group2)
        set_pO2_group2.setLayout(grid_pO2_2)

        grid_pO2_2.addWidget(self.canvas_pO2_2)
        grid_pO2_2.addWidget(self.navi_pO2_2)

        set_pO2_group2.setContentsMargins(2, 2, 2, 2)

        self.Ksv1_O2_tab5_edit.editingFinished.connect(self.print_tab3_Ksv1)
        self.Ksv2_O2_tab5_edit.editingFinished.connect(self.print_tab3_Ksv2)
        self.curv_O2_tab5_edit.editingFinished.connect(self.print_tab3_curv)
        self.Ksv1_CO2_tab5_edit.editingFinished.connect(self.print_tab3_Ksv1)
        self.Ksv2_CO2_tab5_edit.editingFinished.connect(self.print_tab3_Ksv2)
        self.curv_CO2_tab5_edit.editingFinished.connect(self.print_tab3_curv)
        self.calib_ksv_checkbox.clicked.connect(self.calib_ksv_to_tauP)
        self.calib_tauP_checkbox.clicked.connect(self.calib_tauP_to_ksv)

        # ------------------------------------------------------------------------------------
        # connect run button with function
        self.run_tab5_button.clicked.connect(self.CO2_O2_sensing)

        # ==================================================================
        # 6th tab O2 / T
        # ==================================================================
        # Create 6th tab
        self.tab6.layout = QVBoxLayout(self)
        self.tab6.setLayout(self.tab6.layout)

        tab6_hbox_top = QHBoxLayout()
        tab6_hbox_bottom = QHBoxLayout()
        self.tab6.layout.addLayout(tab6_hbox_top)
        self.tab6.layout.addLayout(tab6_hbox_bottom)

        # split top part into left and right
        tab6_vbox_top_left = QVBoxLayout()
        tab6_vbox_top_right = QVBoxLayout()
        tab6_hbox_top.addLayout(tab6_vbox_top_left)
        tab6_hbox_top.addLayout(tab6_vbox_top_right)

        # split bottom part into left and right
        tab6_vbox_bottom_left = QVBoxLayout()
        tab6_vbox_bottom_right = QVBoxLayout()
        tab6_hbox_bottom.addLayout(tab6_vbox_bottom_left)
        tab6_hbox_bottom.addLayout(tab6_vbox_bottom_right)

        # split bottom right into output and input part
        tab6_output = QHBoxLayout()
        tab6_input = QHBoxLayout()
        tab6_vbox_bottom_right.addLayout(tab6_output)
        tab6_vbox_bottom_right.addLayout(tab6_input)

        # split input part into input parameter and run button
        tab6_input_param = QVBoxLayout()
        tab6_input_run = QVBoxLayout()
        tab6_input.addLayout(tab6_input_param)
        tab6_input.addLayout(tab6_input_run)

        # -----------------------------------------------------------
        # Input grid
        # defining temperature and pO2 range for fitting
        regex = r"^(\s*(-|\+)?\d+(?:\.\d+)?\s*,\s*)+(-|\+)?\d+(?:\.\d+)?\s*$"
        validator = QRegExpValidator(QRegExp(regex), self)
        temp_range_tab6 = QLabel(self)
        temp_range_tab6.setText('Temperature [°C]')
        self.temp_range_tab6_edit = QLineEdit(self)
        self.temp_range_tab6_edit.setValidator(validator)
        self.temp_range_tab6_edit.setText('10, 33, 1')
        self.temp_range_tab6_edit.setAlignment(Qt.AlignRight)

        pO2_range_tab6 = QLabel(self)
        pO2_range_tab6.setText('pO2 [hPa]')
        self.pO2_range_tab6_edit = QLineEdit(self)
        self.pO2_range_tab6_edit.setValidator(validator)
        self.pO2_range_tab6_edit.setText('0, 6, 0.2')
        self.pO2_range_tab6_edit.setAlignment(Qt.AlignRight)

        plot_3d = QLabel(self)
        plot_3d.setText('Plotting')
        self.plot_3d_checkbox = QCheckBox('3D', self)
        self.plot_3d_checkbox.toggle()
        self.plot_2d_checkbox = QCheckBox('2D', self)

        temp_example_tab6 = QLabel(self)
        temp_example_tab6.setText('Temperature example')
        self.temp_example_tab6_edit = QLineEdit(self)
        self.temp_example_tab6_edit.setValidator(validator)
        self.temp_example_tab6_edit.setText('10, 26')
        self.temp_example_tab6_edit.setDisabled(True)
        self.temp_example_tab6_edit.setAlignment(Qt.AlignRight)
        ox_example_tab6 = QLabel(self)
        ox_example_tab6.setText('O2 example')
        self.ox_example_tab6_edit = QLineEdit(self)
        self.ox_example_tab6_edit.setValidator(validator)
        self.ox_example_tab6_edit.setText('0, 1.4')
        self.ox_example_tab6_edit.setDisabled(True)
        self.ox_example_tab6_edit.setAlignment(Qt.AlignRight)

        # ---------------------
        # Message box
        self.message_tab6 = QTextEdit(self)
        self.message_tab6.setReadOnly(True)

        # ---------------------
        # update or run button plus progressbar
        self.run_tab6_button = QPushButton('Run O2 / T', self)
        self.run_tab6_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, "
                                           "y2: 1, stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; "
                                           "border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
                                           "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; "
                                           "max-height: 18px;")

        self.progress_tab6 = QProgressBar(self)
        self.progress_tab6.setStyleSheet("border-width: 1px; border-color: #077487; border-style: solid; "
                                         "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; "
                                         "padding-right: 5px; min-height: 10px; max-height: 18px;")

        # ---------------------
        # PLOT - CALIBRATION plane - lifetime plus intensity
        self.fig_tau_tab6 = plt.figure(figsize=(5, 5))
        self.fig_int_tab6 = plt.figure()

        if self.plot_3d_checkbox.isChecked() is True:
            self.ax_tau_tab6 = self.fig_tau_tab6.add_subplot(111, projection='3d')
            self.canvas_tau_tab6 = FigureCanvasQTAgg(self.fig_tau_tab6)
            self.navi_tau_tab6 = NavigationToolbar2QT(self.canvas_tau_tab6, w)
            self.ax_tau_tab6.set_xlim(0, 50)
            self.ax_tau_tab6.set_ylim(0, 20)
            self.ax_tau_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6.set_zlabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)

            self.ax_int_tab6 = self.fig_int_tab6.add_subplot(111, projection='3d')
            self.canvas_int_tab6 = FigureCanvasQTAgg(self.fig_int_tab6)
            self.navi_int_tab6 = NavigationToolbar2QT(self.canvas_int_tab6, w)
            self.ax_int_tab6.set_xlim(0, 100)
            self.ax_int_tab6.set_ylim(0, 100)
            self.ax_int_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6.set_zlabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
        else:
            self.ax_tau_tab6_temp = self.fig_tau_tab6.add_subplot(211)
            self.ax_tau_tab6_o2 = self.fig_tau_tab6.add_subplot(212)
            self.ax_tau_tab6_o2.set_xlim(0, 20)
            self.ax_tau_tab6_temp.set_xlim(0, 50)
            self.ax_tau_tab6_o2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6_temp.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6_o2.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_temp.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_tau_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)

            self.ax_int_tab6_temp = self.fig_int_tab6.add_subplot(211)
            self.ax_int_tab6_o2 = self.fig_int_tab6.add_subplot(211)
            self.canvas_int_tab6 = FigureCanvasQTAgg(self.fig_int_tab6)
            self.navi_int_tab6 = NavigationToolbar2QT(self.canvas_int_tab6, w)
            self.ax_int_tab6_temp.set_xlim(0, 50)
            self.ax_int_tab6_o2.set_xlim(0, 20)
            self.ax_int_tab6_temp.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6_o2.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6_temp.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_o2.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_int_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)

        # PLOT - intersection of planes
        self.fig_pO2T_tab6, self.ax_pO2T_tab6 = plt.subplots() #figsize=(5, 50))
        self.canvas_pO2T_tab6 = FigureCanvasQTAgg(self.fig_pO2T_tab6)
        self.navi_pO2T_tab6 = NavigationToolbar2QT(self.canvas_pO2T_tab6, w)

        self.ax_pO2T_tab6.set_ylim(0, 100)
        self.ax_pO2T_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2T_tab6.set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_pO2T_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2T_tab6.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)

        # ------------------------------------------------------------------------------------
        # create GroupBox to structure the layout
        set_tab6_run_group = QGroupBox()
        set_tab6_run_group.setMinimumHeight(100)
        set_tab6_run_group.setStyleSheet("QGroupBox { border: 1px solid white;}")
        grid_run_tab6 = QGridLayout()

        set_input_group6 = QGroupBox("Input")
        set_input_group6.setMinimumHeight(100)
        set_input_group6.setMinimumWidth(100)
        grid_input6 = QGridLayout()

        set_output_group6 = QGroupBox("Output")
        set_output_group6.setMinimumHeight(100)
        set_output_group6.setMinimumWidth(420)
        grid_output6 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab6_input_run.addWidget(set_tab6_run_group)
        set_tab6_run_group.setLayout(grid_run_tab6)
        tab6_input_param.addWidget(set_input_group6)
        set_input_group6.setLayout(grid_input6)
        tab6_output.addWidget(set_output_group6)
        set_output_group6.setLayout(grid_output6)

        # run button
        grid_run_tab6.addWidget(self.progress_tab6)
        grid_run_tab6.addWidget(self.run_tab6_button)
        # input
        grid_input6.addWidget(temp_range_tab6, 0, 0)
        grid_input6.addWidget(self.temp_range_tab6_edit, 0, 1, 1, 4)
        grid_input6.addWidget(pO2_range_tab6, 1, 0)
        grid_input6.addWidget(self.pO2_range_tab6_edit, 1, 1, 1, 4)
        grid_input6.addWidget(plot_3d, 2, 0)
        grid_input6.addWidget(self.plot_3d_checkbox, 2, 1, 1, 2)
        grid_input6.addWidget(self.plot_2d_checkbox, 2, 3, 1, 4)
        grid_input6.addWidget(temp_example_tab6, 3, 0)
        grid_input6.addWidget(self.temp_example_tab6_edit, 3, 1, 1, 4)
        grid_input6.addWidget(ox_example_tab6, 4, 0)
        grid_input6.addWidget(self.ox_example_tab6_edit, 4, 1, 1, 4)

        # output results
        grid_output6.addWidget(self.message_tab6, 1, 2)

        # ----------------------------------
        # create GroupBox to structure the layout
        # left part of tab6 - top
        set_temp_calib_group_tab6 = QGroupBox("Lifetime calibration plane")
        set_temp_calib_group_tab6.setMinimumHeight(200)
        set_temp_calib_group_tab6.setMinimumWidth(400)
        grid_tempcalib_tab6 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab6_vbox_top_left.addWidget(set_temp_calib_group_tab6)
        set_temp_calib_group_tab6.setLayout(grid_tempcalib_tab6)

        grid_tempcalib_tab6.addWidget(self.canvas_tau_tab6)
        grid_tempcalib_tab6.addWidget(self.navi_tau_tab6)

        # left part of tab6 - bottom
        set_temp_group_tab6 = QGroupBox("Intensity calibration plane")
        set_temp_group_tab6.setMinimumHeight(200)
        set_temp_group_tab6.setMinimumWidth(400)
        grid_temp_group_tab6 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab6_vbox_top_right.addWidget(set_temp_group_tab6)
        set_temp_group_tab6.setLayout(grid_temp_group_tab6)

        grid_temp_group_tab6.addWidget(self.canvas_int_tab6)
        grid_temp_group_tab6.addWidget(self.navi_int_tab6)

        # ----------------------------------
        # create GroupBox to structure the layout
        # right part of tab6 - top
        set_pO2calib_group_tab6 = QGroupBox("Intersection for evaluation")
        set_pO2calib_group_tab6.setMinimumHeight(200)
        set_pO2calib_group_tab6.setMinimumWidth(400)
        grid_O2calib_tab6 = QGridLayout()

        # add GroupBox to layout and load buttons in GroupBox
        tab6_vbox_bottom_left.addWidget(set_pO2calib_group_tab6)
        set_pO2calib_group_tab6.setLayout(grid_O2calib_tab6)

        grid_O2calib_tab6.addWidget(self.canvas_pO2T_tab6)
        grid_O2calib_tab6.addWidget(self.navi_pO2T_tab6)
        set_pO2calib_group_tab6.setContentsMargins(2, 10, 2, 2)

        # ------------------------------------------------------------------------------------
        # connect run button with function
        self.run_tab6_button.clicked.connect(self.O2_temp_sensing)
        self.plot_3d_checkbox.clicked.connect(self.plotting_system_2Dto3D)
        self.plot_2d_checkbox.clicked.connect(self.plotting_system_3Dto2D)
        self.show()

# -------------------------------------------------------------------------------------------------------------
# Helping hands
# -------------------------------------------------------------------------------------------------------------
    def keyPressEvent(self, e):

        if (e.modifiers() & Qt.ControlModifier):
            selected = self.tableCalibration.selectedRanges()

            if e.key() == Qt.Key_C: #copy
                s = "\t".join([str(self.tableCalibration.horizontalHeaderItem(i).text())
                                    for i in range(selected[0].leftColumn(), selected[0].rightColumn()+1)])

                s = s + '\n'
                for r in range(selected[0].topRow(), selected[0].bottomRow()+1):
                    for c in range(selected[0].leftColumn(), selected[0].rightColumn()+1):
                        try:
                            s += str(self.tableCalibration.item(r, c).text()) + "\t"
                        except AttributeError:
                            s += "\t"
                    s = s + "\n" #eliminate last '\t'
                self.clip.setText(s)

# -------------------------------------------------------------------------------------------------------------
# Individual tabs connected with functions
# -------------------------------------------------------------------------------------------------------------
    def print_lifetime_phosphor(self):
        print('Lifetime phosphor: ', self.lifetime_phosphor_edit.text().replace(',', '.'), 'µs')
        return np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))

    def print_lifetime_phosphor1(self):
        print('Lifetime phosphor-1: ', self.lifetime_phosphor1_edit.text().replace(',', '.'), 'µs')
        return np.float64(self.lifetime_phosphor1_edit.text().replace(',', '.'))

    def print_lifetime_phosphor2(self):
        print('Lifetime phosphor-2: ', self.lifetime_phosphor2_edit.text().replace(',', '.'), 'µs')
        return np.float64(self.lifetime_phosphor2_edit.text().replace(',', '.'))

    def print_intensity_ratio(self):
        print('Intensity ratio-1: ', self.intensity_ratio_edit.text().replace(',', '.'))
        return np.float64(self.intensity_ratio_edit.text().replace(',', '.'))

    def print_intensity_ratio1(self):
        print('Intensity ratio-2: ', self.intensity_ratio1_edit.text().replace(',', '.'))
        return np.float64(self.intensity_ratio1_edit.text().replace(',', '.'))

    def print_intensity_ratio2(self):
        print('Intensity ratio-3: ', self.intensity_ratio2_edit.text().replace(',', '.'))
        return np.float64(self.intensity_ratio2_edit.text().replace(',', '.'))

    def reportInput_intensity3(self):
        print('Intensity ratio 3: ', self.intensity3_checkbox.isChecked())
        if self.intensity3_checkbox.isChecked() is True:
            if self.intensity_checkbox.isChecked() is True:
                self.intensity2_checkbox.setCheckState(False)
        return self.intensity3_checkbox.isChecked()

    def reportInput_intensity2(self):
        print('Intensity ratio 2: ', self.intensity2_checkbox.isChecked())
        if self.intensity2_checkbox.isChecked() is True:
            if self.intensity_checkbox.isChecked() is True:
                self.intensity3_checkbox.setCheckState(False)
        return self.intensity2_checkbox.isChecked()

    def reportInput_intensity(self):
        print('Intensity ratio 1: ', self.intensity_checkbox.isChecked())
        if self.intensity2_checkbox.isChecked() is True:
            self.intensity3_checkbox.setCheckState(False)
        return self.intensity_checkbox.isChecked()

    def reportInput_lifetime3(self):
        print('lifetime 3: ', self.lifetime3_checkbox.isChecked())
        if self.lifetime3_checkbox.isChecked() is True:
            if self.lifetime_checkbox.isChecked() is True:
                self.lifetime2_checkbox.setCheckState(False)
        return self.lifetime3_checkbox.isChecked()

    def reportInput_lifetime2(self):
        print('lifetime 2: ', self.lifetime2_checkbox.isChecked())
        if self.lifetime2_checkbox.isChecked() is True:
            if self.lifetime_checkbox.isChecked() is True:
                self.lifetime3_checkbox.setCheckState(False)
        return self.lifetime2_checkbox.isChecked()

    def reportInput_lifetime(self):
        print('lifetime 1: ', self.lifetime_checkbox.isChecked())
        if self.lifetime2_checkbox.isChecked() is True:
            self.lifetime3_checkbox.setCheckState(False)
        return self.lifetime_checkbox.isChecked()

    def plotting_system_3Dto2D(self):
        if self.plot_2d_checkbox.isChecked() is True:
            self.plot_3d_checkbox.setCheckState(False)
        self.plot_calibration_system()
        self.temp_example_tab6_edit.setEnabled(True)
        self.ox_example_tab6_edit.setEnabled(True)
        return self.plot_3d_checkbox.isChecked()

    def plotting_system_2Dto3D(self):
        if self.plot_3d_checkbox.isChecked() is True:
            self.plot_2d_checkbox.setCheckState(False)
        self.plot_calibration_system()
        self.temp_example_tab6_edit.setDisabled(True)
        self.ox_example_tab6_edit.setDisabled(True)
        return self.plot_2d_checkbox.isChecked()

    def plot_calibration_system(self):
        # clear - CALIBRATION plane - lifetime as it is comparable with the measurement results)
        self.ax_tau_tab6.cla()
        self.fig_tau_tab6.clear()
        if self.plot_3d_checkbox.isChecked() is True:
            self.ax_tau_tab6 = self.fig_tau_tab6.gca(projection='3d')
            self.ax_tau_tab6.set_xlim(0, 50)
            self.ax_tau_tab6.set_ylim(0, 20)
            self.ax_tau_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6.set_zlabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_tau_tab6.canvas.draw()
        else:
            self.ax_tau_tab6_temp = self.fig_tau_tab6.add_subplot(211)
            self.ax_tau_tab6_o2 = self.fig_tau_tab6.add_subplot(212)
            self.ax_tau_tab6_o2.set_xlim(0, 20)
            self.ax_tau_tab6_temp.set_xlim(0, 50)
            self.ax_tau_tab6_o2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6_temp.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6_o2.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_temp.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_tau_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98, hspace=.5)
            self.fig_tau_tab6.canvas.draw()

        # clear - CALIBRATION plane - intensity ratio (I prompt fluorescence vs delayed fluorescence as it is comparable
        # with the measurement results)
        self.ax_int_tab6.cla()
        self.fig_int_tab6.clear()
        if self.plot_3d_checkbox.isChecked() is True:
            self.ax_int_tab6 = self.fig_int_tab6.gca(projection='3d')
            self.ax_int_tab6.set_xlim(0, 100)
            self.ax_int_tab6.set_ylim(0, 100)
            self.ax_int_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6.set_zlabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_int_tab6.canvas.draw()
        else:
            self.ax_int_tab6_o2 = self.fig_int_tab6.add_subplot(212)
            self.ax_int_tab6_temp = self.fig_int_tab6.add_subplot(211)
            self.ax_int_tab6_o2.set_xlim(0, 20)
            self.ax_int_tab6_temp.set_xlim(0, 50)
            self.ax_int_tab6_o2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6_temp.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6_o2.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_temp.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_int_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98, hspace=.5)
            self.fig_int_tab6.canvas.draw()

    def print_error_assumed(self):
        print('Measurement uncertainty: ', self.error_assumed_edit.text().replace(',', '.'), '°')
        return np.float64(self.error_assumed_edit.text().replace(',', '.'))

    def print_frequency1(self):
        print('Modulation frequency f1: ', self.frequency1_edit.text().replace(',', '.'), '°')
        return np.float64(self.frequency1_edit.text().replace(',', '.'))

    def print_frequency2_range(self):
        print('Modulation frequency f2: ', self.frequency2_min_edit.text().replace(',', '.'), '-',
              self.frequency2_max_edit.text().replace(',', '.'), 'Hz')
        print('Modulation frequency f2 - steps: ', self.frequency2_step_edit.text().replace(',', '.'), 'Hz')
        return np.float64(self.frequency2_min_edit.text().replace(',', '.')), \
               np.float64(self.frequency2_max_edit.text().replace(',', '.')), \
               np.float64(self.frequency2_step_edit.text().replace(',', '.'))

    def control_input(self):
        # checkbox control - what should be analyzed
        # tau 1/2/3 in seconds
        if self.lifetime_checkbox.isChecked() is True:
            if self.lifetime_phosphor_edit.text() == '':
                lifetime_notgiven = QMessageBox()
                lifetime_notgiven.setIcon(QMessageBox.Information)
                lifetime_notgiven.setText("Provide lifetime in µs")
                lifetime_notgiven.setInformativeText("The lifetime of the phosphor 1 is required for simulation.")
                lifetime_notgiven.setWindowTitle("Error!")
                lifetime_notgiven.exec_()
                return
            tau_phos1 = self.print_lifetime_phosphor()*1E-6
        else:
            tau_phos1 = None
        if self.lifetime2_checkbox.isChecked() is True:
            if self.lifetime_phosphor1_edit.text() == '':
                lifetime_notgiven = QMessageBox()
                lifetime_notgiven.setIcon(QMessageBox.Information)
                lifetime_notgiven.setText("Provide lifetime in µs")
                lifetime_notgiven.setInformativeText("The lifetime of the phosphor 2 is required for simulation.")
                lifetime_notgiven.setWindowTitle("Error!")
                lifetime_notgiven.exec_()
                return
            tau_phos2 = self.print_lifetime_phosphor1()*1E-6
        else:
            tau_phos2 = None
        if self.lifetime3_checkbox.isChecked() is True:
            if self.lifetime_phosphor2_edit.text() == '':
                lifetime_notgiven = QMessageBox()
                lifetime_notgiven.setIcon(QMessageBox.Information)
                lifetime_notgiven.setText("Provide lifetime in µs")
                lifetime_notgiven.setInformativeText("The lifetime of the phosphor 3 is required for simulation.")
                lifetime_notgiven.setWindowTitle("Error!")
                lifetime_notgiven.exec_()
                return
            tau_phos3 = self.print_lifetime_phosphor2()*1E-6
        else:
            tau_phos3 = None

        # i-ratio 1/2/3
        if self.intensity_checkbox.isChecked() is True:
            if self.intensity_ratio_edit.text() == '':
                intensity_notgiven = QMessageBox()
                intensity_notgiven.setIcon(QMessageBox.Information)
                intensity_notgiven.setText("Provide intensity ratio")
                intensity_notgiven.setInformativeText("The intensity ratio I-ratio 1 is required for simulation.")
                intensity_notgiven.setWindowTitle("Error!")
                intensity_notgiven.exec_()
                return
            i_ratio1 = self.print_intensity_ratio()
        else:
            i_ratio1 = None
        if self.intensity2_checkbox.isChecked() is True:
            if self.intensity_ratio1_edit.text() == '':
                intensity_notgiven = QMessageBox()
                intensity_notgiven.setIcon(QMessageBox.Information)
                intensity_notgiven.setText("Provide intensity ratio")
                intensity_notgiven.setInformativeText("The intensity ratio I-ratio 2 is required for simulation.")
                intensity_notgiven.setWindowTitle("Error!")
                intensity_notgiven.exec_()
                return
            i_ratio2 = self.print_intensity_ratio1()
        else:
            i_ratio2 = None
        if self.intensity3_checkbox.isChecked() is True:
            if self.intensity_ratio2_edit.text() == '':
                intensity_notgiven = QMessageBox()
                intensity_notgiven.setIcon(QMessageBox.Information)
                intensity_notgiven.setText("Provide intensity ratio")
                intensity_notgiven.setInformativeText("The intensity ratio I-ratio 3 is required for simulation.")
                intensity_notgiven.setWindowTitle("Error!")
                intensity_notgiven.exec_()
                return
            i_ratio3 = self.print_intensity_ratio2()
        else:
            i_ratio3 = None

        return tau_phos1, tau_phos2, tau_phos3, i_ratio1, i_ratio2, i_ratio3

    def plot_phaseangle(self, lifetime_phosphor, signals, ls, color, label_, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        lines = []
        r = lifetime_phosphor
        if r >= 0.1:
            rounding = 3
        elif 0.01 < r < 0.1:
            rounding = 5
        else:
            rounding = 9
        label_f1 = 'phi$_{f1}$(' + label_ + ') [°]'
        label_f2 = 'phi$_{f2}$(' + label_ + ') [°]'

        line1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'phi(f1) [°]'].round(rounding), lw=1.75,
                         ls=linestyles[ls], color=color_phaseangle[color], alpha=0.5, label=label_f1)
        line2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'phi(f2) [°]'].round(rounding), lw=1.75,
                         ls=linestyles[ls], color=color_phaseangle[color], label=label_f2)
        lines.append(line1)
        lines.append(line2)
        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*self.signals.loc[:, 'f2 [Hz]'].values[-1]/1000)

        plt.autoscale(ax)
        ax.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)

        # legend adjustment
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 2:
            fontsize_adjust = 10
            loc_ = (0.5, 1.15)
            top_ = 0.89
            ncol_ = 3
        elif 4 > len(labels) > 2:
            # adjust fontsize and columns
            fontsize_adjust = 9
            loc_ = (0.5, 1.15)
            top_ = 0.88
            ncol_ = 3
        else:
            fontsize_adjust = 7
            loc_ = (0.5, 1.15)
            top_ = 0.87
            ncol_ = 4
        leg = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=ncol_, fancybox=True, shadow=True,
                        fontsize=fontsize_adjust)

        # dict mapping legend
        self.lined = dict()
        for self.legline, self.origline in zip(leg.get_lines(), lines):
            self.legline.set_picker(5)  # 5 pts tolerance
            self.lined[self.legline] = self.origline

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=top_)
        fig.canvas.draw()

    def plot_lifetime(self, lifetime_phosphor, error_lifetime, signals, color, label_, ylim, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        lifes = []
        er = error_lifetime
        r = lifetime_phosphor

        if r >= 1:
            rounding = 3
        elif 0.01 < r < 1:
            rounding = 4
        else:
            rounding = 5
        label_min = '(' + label_ + ')$_{min}$ [µs]'
        label_max = '(' + label_ + ')$_{max}$ [µs]'

        life1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'tau [µs]'].round(rounding), lw=1.,
                         ls=linestyles['densely dashed'], color=color_combined[color], alpha=1,
                         label='{} [µs]'.format(label_))
        life2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'tau(min) [µs]'].round(rounding), lw=1.5,
                         ls=linestyles['solid'], color=color_combined[color], alpha=1, label=label_min)
        life3, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'tau(max) [µs]'].round(rounding), lw=1.5,
                         ls=linestyles['solid'], color=color_combined[color], alpha=0.5, label=label_max)
        lifes.append(life1)
        lifes.append(life2)
        lifes.append(life3)
        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*signals.loc[:, 'f2 [Hz]'].values[-1]/1000)

        if not ylim:
            ymin_abs = signals.loc[:, 'tau(min) [µs]'].mean()
            ymax_abs = signals.loc[:, 'tau(max) [µs]'].mean()
        else:
            if ylim[0] < signals.loc[:, 'tau(min) [µs]'].mean():
                ymin_abs = ylim[0]
            else:
                ymin_abs = signals.loc[:, 'tau(min) [µs]'].mean()
            if ylim[1] > signals.loc[:, 'tau(max) [µs]'].mean():
                ymax_abs = ylim[1]
            else:
                ymax_abs = signals.loc[:, 'tau(max) [µs]'].mean()
        ylim = [ymin_abs, ymax_abs]
        ax.set_ylim(ymin_abs * (1 - er / 10), ymax_abs * (1 + er / 10))

        # legend adjustment
        handles, labels = self.ax_lifetime.get_legend_handles_labels()
        if len(labels) < 4:
            fontsize_adjust = 8
            loc_ = (0.9, 0.98)
        elif 6 >= len(labels) >= 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        else:
            fontsize_adjust = 6.8
            loc_ = (0.9, 1.05)
        leg_life = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=1, fancybox=True, shadow=True,
                             fontsize=fontsize_adjust)
        # dict mapping legend
        self.lined_life = dict()
        for self.legline_life, self.origline_life in zip(leg_life.get_lines(), lifes):
            self.legline_life.set_picker(5)  # 5 pts tolerance
            self.lined_life[self.legline_life] = self.origline_life

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        fig.canvas.draw()

        return ylim

    def plot_lifetime_err(self, lifetime_phosphor, er, signals, color, label_, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        r = lifetime_phosphor
        error_lifetime = []
        if r >= 1:
            rounding = 3
        elif 0.01 < r < 1:
            rounding = 4
        else:
            rounding = 5
        signals0 = signals.copy()
        label_min = 'er(' + label_ + ')$_{min}$ [%]'
        label_max = 'er(' + label_ + ')$_{max}$ [%]'

        li1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er(tau, min) [%]'].round(rounding), lw=1.5,
                       ls=linestyles['solid'], color=color_combined[color], label=label_min)
        li2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er(tau, max) [%]'].round(rounding), lw=1.5,
                       ls=linestyles['solid'], color=color_combined[color], alpha=0.5, label=label_max)
        error_lifetime.append(li1)
        error_lifetime.append(li2)
        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*signals.loc[:, 'f2 [Hz]'].values[-1]/1000)

        ax.set_ylim(-1*er, er)
        ax.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)

        # legend adjustment
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) < 4:
            fontsize_adjust = 8
            loc_ = (0.89, 0.98)
        elif 6 >= len(labels) >= 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        else:
            fontsize_adjust = 6.8
            loc_ = (0.9, 0.98)
        leg_life_er = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=1, fancybox=True, shadow=True,
                                fontsize=fontsize_adjust)
        ax.plot(signals0.loc[:, 'f2 [Hz]']/1000, [0]*len(signals0.loc[:, 'f2 [Hz]'].index), lw=1.,
                ls=linestyles['densely dashed'], color='grey')

        # dict mapping legend
        self.lined_life_er = dict()
        for self.legline_life_er, self.origline_life_er in zip(leg_life_er.get_lines(), error_lifetime):
            self.legline_life_er.set_picker(5)  # 5 pts tolerance
            self.lined_life_er[self.legline_life_er] = self.origline_life_er

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        fig.canvas.draw()

    def plot_intensity(self, error_intensity, color, signals, label_, ylim, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        err = error_intensity
        intens = []

        r = np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))
        if r >= 1:
            rounding = 4
        elif 0.01 < r < 1:
            rounding = 6
        else:
            rounding = 8
        label = label_.split(',')[1] + '({})'.format(label_.split(',')[0])
        label_min = label_.split(',')[1] + '({})'.format(label_.split(',')[0]) + '$_{min}$'
        label_max = label_.split(',')[1] + '({})'.format(label_.split(',')[0]) + '$_{max}$'

        in1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'i_ratio'].round(rounding), lw=1.,
                       ls=linestyles['densely dashed'], color='grey', label=label)
        in2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'i_ratio(min)'].round(rounding), lw=1.5,
                       ls=linestyles['solid'], color=color_combined[color], label=label_min)
        in3, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'i_ratio(max)'].round(rounding), lw=1.5,
                       ls=linestyles['solid'], color=color_combined[color], label=label_max)

        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*signals.loc[:, 'f2 [Hz]'].values[-1]/1000)

        if not ylim:
            ymin_abs = signals.loc[:, 'i_ratio(min)'].mean()
            ymax_abs = signals.loc[:, 'i_ratio(max)'].mean()
        else:
            if ylim[0] < signals.loc[:, 'i_ratio(min)'].mean():
                ymin_abs = ylim[0]
            else:
                ymin_abs = signals.loc[:, 'i_ratio(min)'].mean()
            if ylim[1] > signals.loc[:, 'i_ratio(max)'].mean():
                ymax_abs = ylim[1]
            else:
                ymax_abs = signals.loc[:, 'i_ratio(max)'].mean()

        ylim = [ymin_abs, ymax_abs]
        ax.set_ylim(ymin_abs * (1 - err / 100), ymax_abs * (1 + err / 100))

        # legend adjustment
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) < 4:
            fontsize_adjust = 8
            loc_ = (0.9, 0.98)
        elif 6 >= len(labels) >= 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        else:
            fontsize_adjust = 6.8
            loc_ = (0.9, 1.05)
        leg_intens = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=1, fancybox=True, shadow=True,
                               fontsize=fontsize_adjust)

        intens.append(in1)
        intens.append(in2)
        intens.append(in3)

        # dict mapping legend
        self.lined_intens = dict()
        for self.legline_intens, self.origline_intens in zip(leg_intens.get_lines(), intens):
            self.legline_intens.set_picker(5)  # 5 pts tolerance
            self.lined_intens[self.legline_intens] = self.origline_intens

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        fig.canvas.draw()

        return ylim

    def plot_intensity_abs(self, lifetime_phosphor, signals, er, color, label_, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        intens_dev = []
        r = lifetime_phosphor

        if r >= 1:
            rounding = 4
        elif 0.01 < r < 1:
            rounding = 6
        else:
            rounding = 8
        signals0 = signals.copy()
        label_min = 'dev(' + label_.split(',')[1] + ', {}'.format(label_.split(',')[0]) + ')$_{min}$'
        label_max = 'dev(' + label_.split(',')[1] + ', {}'.format(label_.split(',')[0]) + ')$_{max}$'

        dev1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er_abs(i, min) [%]'].round(rounding), lw=1.5,
                        ls=linestyles['solid'], color=color_combined[color], label=label_min)
        dev2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er_abs(i, max) [%]'].round(rounding), lw=1.5,
                        ls=linestyles['solid'], color=color_combined[color], label=label_max)
        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*signals.loc[:, 'f2 [Hz]'].values[-1]/1000)
        intens_dev.append(dev1)
        intens_dev.append(dev2)

        ax.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        ax.set_ylim(-1*er/100, er/100)

        # legend adjustment
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) < 4:
            fontsize_adjust = 8
            loc_ = (0.9, 0.98)
        elif 6 >= len(labels) >= 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        else:
            fontsize_adjust = 6.8
            loc_ = (0.9, 1.05)
        leg_dev = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=1, fancybox=True, shadow=True,
                            fontsize=fontsize_adjust)
        # dict mapping legend
        self.lined_dev = dict()
        for self.legline_dev, self.origline_dev in zip(leg_dev.get_lines(), intens_dev):
            self.legline_dev.set_picker(5)  # 5 pts tolerance
            self.lined_dev[self.legline_dev] = self.origline_dev

        ax.plot(signals0.loc[:, 'f2 [Hz]']/1000, [0]*len(signals0.loc[:, 'f2 [Hz]'].index), lw=1.,
                ls=linestyles['densely dashed'], color='grey')

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        fig.canvas.draw()

    def plot_intensity_rel(self, lifetime_phosphor, signals, er, color, label_, ax, fig):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        error_intensity = []
        r = lifetime_phosphor
        if r >= 1:
            rounding = 4
        elif 0.01 < r < 1:
            rounding = 6
        else:
            rounding = 8
        signals0 = signals.copy()
        label_min = 'er(' + label_ + ')$_{min}$ [%]'
        label_max = 'er(' + label_ + ')$_{max}$ [%]'

        er_int1, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er_rel(i, min) [%]'].round(rounding), lw=1.5,
                           ls=linestyles['solid'], color=color_combined[color], label=label_min)
        er_int2, = ax.plot(signals.loc[:, 'f2 [Hz]']/1000, signals.loc[:, 'er_rel(i, max) [%]'].round(rounding), lw=1.5,
                           ls=linestyles['solid'], color=color_combined[color], label=label_max)
        ax.set_xlim(signals.loc[:, 'f2 [Hz]'].values[0]/1000*0.8, 1.05*signals.loc[:, 'f2 [Hz]'].values[-1]/1000)
        error_intensity.append(er_int1)
        error_intensity.append(er_int2)

        ax.set_ylim(-er, er)

        # legend adjustment
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) < 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        elif 6 >= len(labels) >= 4:
            fontsize_adjust = 7
            loc_ = (0.9, 0.98)
        else:
            fontsize_adjust = 6.8
            loc_ = (0.9, 0.98)
        leg_intens_err = ax.legend(loc='upper center', bbox_to_anchor=loc_, ncol=1, fancybox=True, shadow=True,
                                   fontsize=fontsize_adjust)
        # dict mapping legend
        self.lined_int_er = dict()
        for self.legline_int_er, self.origline_int_er in zip(leg_intens_err.get_lines(), error_intensity):
            self.legline_int_er.set_picker(5)  # 5 pts tolerance
            self.lined_int_er[self.legline_int_er] = self.origline_int_er

        ax.plot(signals0.loc[:, 'f2 [Hz]']/1000, [0]*len(signals0.loc[:, 'f2 [Hz]'].index), lw=1.,
                ls=linestyles['densely dashed'], color='grey')

        ax.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        fig.canvas.draw()

    def report_messagebox(self, signals):
        f2_min1 = signals.loc[:, 'er(tau, min) [%]'].idxmax()
        f2_max1 = signals.loc[:, 'er(tau, max) [%]'].idxmin()
        f2_min2 = signals.loc[:, 'er_abs(i, min) [%]'].idxmax()
        f2_max2 = signals.loc[:, 'er_abs(i, max) [%]'].idxmin()

        if signals.loc[f2_min1, 'f2 [Hz]'] <= signals.loc[f2_max1, 'f2 [Hz]']:
            if (signals.loc[f2_min1, 'er_abs(i, min) [%]']) <= (signals.loc[f2_min2, 'er_abs(i, min) [%]']):
                f = f2_min1
            else:
                f = f2_min2
        else:
            if (signals.loc[f2_max1, 'er_abs(i, min) [%]']) <= (signals.loc[f2_max2, 'er_abs(i, min) [%]']):
                f = f2_max1
            else:
                f = f2_max2

        f1 = signals.loc[f, 'f1 [Hz]']
        f2_opt = signals.loc[f, 'f2 [Hz]']
        phi_f1 = signals.loc[f, 'phi(f1) [°]']
        phi_f2 = signals.loc[f, 'phi(f2) [°]']
        tau_min = signals.loc[f, 'tau(min) [µs]']
        tau_max = signals.loc[f, 'tau(max) [µs]']
        er_tau_min = signals.loc[f, 'er(tau, min) [%]']
        er_tau_max = signals.loc[f, 'er(tau, max) [%]']
        i_ratio_min = signals.loc[f, 'i_ratio(min)']
        i_ratio_max = signals.loc[f, 'i_ratio(max)']
        er_i_ratio_min = signals.loc[f, 'er_abs(i, min) [%]']
        er_i_ratio_max = signals.loc[f, 'er_abs(i, max) [%]']
        er_rel_i_min = signals.loc[f, 'er_rel(i, min) [%]']
        er_rel_i_max = signals.loc[f, 'er_rel(i, max) [%]']

        res = pd.Series({'f1': f1, 'f2': f2_opt, 'phi_f1': phi_f1, 'phi_f2': phi_f2, 'tau_min': tau_min,
                         'tau_max': tau_max, 'er(tau)_min': er_tau_min, 'er(tau)_max': er_tau_max,
                         'i-ratio(min)': i_ratio_min, 'i-ratio(max)': i_ratio_max, 'er(i-ratio)_min': er_i_ratio_min,
                         'er(i-ratio)_max': er_i_ratio_max, 'er_rel(i-ratio)_min': er_rel_i_min,
                         'er_rel(i-ratio)_max': er_rel_i_max})

        return res

    def report_save(self, t, I, signals, lifetime_phosphor, intensity_ratio, error_assumed, today, f):
        rep = fp.report(t=t, I=I, signals=signals, lifetime_phosphor=lifetime_phosphor, intensity_ratio=intensity_ratio,
                        error_assumed=error_assumed)

        filename = str(today[:10]) + '_' + str(today[11:13]) + '-' + str(today[14:16]) + '-' + str(today[17:19]) \
                   + 'h_report_τ{}-I{}-f{}Hz.txt'.format(t, I, f)

        return rep, filename

    def print_frequency_fixed_1(self):
        print('Modulation frequency f1: ', self.frequency1fix_edit.text().replace(',', '.'), 'Hz')
        return np.float64(self.frequency1fix_edit.text().replace(',', '.'))

    def print_frequency_fixed_2(self):
        print('Modulation frequency f2: ', self.frequency2fix_edit.text().replace(',', '.'), 'Hz')
        return np.float64(self.frequency2fix_edit.text().replace(',', '.'))

    def print_Intensity_ratio_dualsens(self):
        print('Intensity ratio $I_F$ / $I_P$: ', self.int_ratio_dualsens_edit.text().replace(',', '.'), '%')
        return np.float64(self.int_ratio_dualsens_edit.text().replace(',', '.'))

    def print_error_assumption(self):
        print('assumed measurement uncertainty: ', self.error_assumed_meas_edit.text().replace(',', '.'))
        return np.float64(self.error_assumed_meas_edit.text().replace(',', '.'))

    def print_tab3_slope(self):
        print('pH sensing slope: ', self.slope_tab3_edit.text().replace(',', '.'))
        return np.float64(self.slope_tab3_edit.text().replace(',', '.'))

    def print_tab3_pKa(self):
        print('pH sensing pKa: ', self.pka_tab3_edit.text().replace(',', '.'))
        return np.float64(self.pka_tab3_edit.text().replace(',', '.'))

    def print_tab3_Ksv1(self):
        Ksv1 = np.float64(self.Ksv1_tab3_edit.text().replace(',', '.'))
        print('TSM - Ksv1: ', self.Ksv1_tab3_edit.text().replace(',', '.'))
        return Ksv1

    def print_tab3_Ksv2(self):
        Ksv1 = np.float64(self.Ksv1_tab3_edit.text().replace(',', '.'))
        Ksv2 = np.float64(self.Ksv2_tab3_edit.text().replace(',', '.')) * Ksv1
        print('TSM - Ksv2: ', Ksv2)
        return Ksv2

    def print_tab3_curv(self):
        curv_O2 = np.float64(self.curv_O2_tab3_edit.text().replace(',', '.'))
        print('(C)O2 sensing curvature factor: ', curv_O2)
        return curv_O2

    def extract_calibration_points(self, para_list, para_order, cols, calib_type):
        if calib_type == '2point':
            for i in range(self.tableCalibration.rowCount()):
                para_order.append(np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')))

                if np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')) in para_list:
                    pass
                else:
                    para_list.append(np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')))
        else:
            for i in range(self.tableCalibration.rowCount()-2):
                para_order.append(np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')))

                if np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')) in para_list:
                    pass
                else:
                    para_list.append(np.float64(self.tableCalibration.item(i, cols).text().replace(',', '.')))
        para_list.sort()

        return para_list, para_order

    def extract_dPhi_calibration(self, pH_soll, temp_soll, row):
        for i in range(self.tableCalibration.rowCount()):
            if self.pH_order[i] == pH_soll:
                if self.temp_order[i] == temp_soll:
                    phi = np.float64(self.tableCalibration.item(i, row).text().replace(',', '.'))
                else:
                    pass
            else:
                pass
        if not phi:
            phi_return = None
        else:
            phi_return = phi
        return phi_return

    def clear_table_parts(self, table, rows, cols):
        for i in range(rows):
            if isinstance(cols, list):
                for j in cols:
                    if not table.item(i, j):
                        pass
                    else:
                        table.item(i, j).setText('')
            else:
                for j in range(cols):
                    if not table.item(i, j):
                        pass
                    else:
                        table.item(i, j).setText('')
        return

    def insertRows(self):
        position = self.tableINPUT.rowCount()
        self.tableINPUT.insertRow(position)
        self.tableINPUT.resizeColumnToContents(0)

        return True

# ---------------------------------------------------------------------------------------------------------------------
# error propagation 1st and 2nd tab
# ---------------------------------------------------------------------------------------------------------------------
    def error_propagation(self):
        print('#--------------------------------------')
        print('error propagation')
        self.run_sim_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px; border-color: "
                                          "#077487; border-style: solid; border-radius: 7; padding: 5px; font-size: "
                                          "10px; padding-left: 1px; padding-right: 5px; min-height: 10px; max-height:"
                                          " 18px;")
        # ------------------------------------------------------------------------------------------------------------
        # update figures - 1st tab
        self.message.clear()

        self.ax_phaseangle.cla()
        self.ax_phaseangle.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_phaseangle.set_ylabel('Superimposed phase angle Phi [°]', fontsize=9)
        self.ax_phaseangle.set_xlim(0, 20)
        self.ax_phaseangle.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_phaseangle.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_phaseangle.canvas.draw()

        x = np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))
        self.ax_lifetime.cla()
        self.ax_lifetime.set_xlabel('Modulation frequency f2 [Hz]', fontsize=9)
        self.ax_lifetime.set_ylabel('lifetime tau [µs]', fontsize=9)
        self.ax_lifetime.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
        self.ax_lifetime.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_lifetime.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_lifetime.canvas.draw()

        self.ax_lifetime_er.cla()
        self.ax_lifetime_er.set_xlim(0, 20)
        self.ax_lifetime_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_lifetime_er.set_ylabel('Rel. error rate [%]', fontsize=9)
        self.ax_lifetime_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_lifetime_err.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_lifetime_err.canvas.draw()

        # update figures - 2nd tab
        self.message_int.clear()

        self.ax_intensity.cla()
        self.ax_intensity.set_xlim(0, 20)
        self.ax_intensity.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity.set_ylabel('Intensity ratio', fontsize=9)
        self.ax_intensity.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_intensity.canvas.draw()

        self.ax_intensity_abs.cla()
        self.ax_intensity_abs.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
        self.ax_intensity_abs.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity_abs.set_ylabel('Abs. deviation intensity ratio', fontsize=9)
        self.ax_intensity_abs.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity_abs.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_intensity_abs.canvas.draw()

        self.ax_intensity_er.cla()
        self.ax_intensity_er.set_xlim(0, 20)
        self.ax_intensity_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
        self.ax_intensity_er.set_ylabel('Rel. error rate [%]', fontsize=9)
        self.ax_intensity_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_intensity_er.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
        self.fig_intensity_er.canvas.draw()

        # ------------------------------------------------------------------------------------------------------------
        # load all input parameters
        # error
        self.error_prop = self.print_error_assumed()
        self.f1_prop = self.print_frequency1()
        [f2_start, f2_stop, f2_step] = self.print_frequency2_range()

        # check if f2 inside of a reasonable range
        f2_prop = np.linspace(start=f2_start, stop=f2_stop, num=(f2_stop+f2_start)/f2_step, dtype=np.float64)
        if f2_prop[0] < 0 or f2_prop[0] == 0:
            try:
                f2_prop[0]
            except:
                frequency_outside_range = QMessageBox()
                frequency_outside_range.setIcon(QMessageBox.Information)
                frequency_outside_range.setText("Change range for modulation frequency 2!")
                frequency_outside_range.setInformativeText("Expected is a frequency higher than 0 Hz.")
                frequency_outside_range.setWindowTitle("Error!")
                frequency_outside_range.exec_()
                return
        # depending on f2 range, reduce the points/steps for f2
        if len(f2_prop) > 1000:
            f2_steps_new = np.int64(((f2_stop+f2_start)/f2_step)/1000)
            f2_prop = np.linspace(start=f2_start, stop=f2_stop, num=f2_steps_new, dtype=np.float64)
            self.frequency2_step_edit.setText(str(f2_steps_new))
        else:
            f2_steps_new = f2_step

        # -------------------------------------------------------------------
        # lifetime and intensity ratio according to selected checkboxes
        [self.tau_phos1, self.tau_phos2, self.tau_phos3, self.i_ratio1, self.i_ratio2,
         self.i_ratio3] = self.control_input()

        # ------------------------------------------------------------------------------------------------------------
        f_combination1 = []
        f_combination2 = []
        f_combination3 = []
        f_combination4 = []
        f_combination5 = []
        f_combination6 = []
        f_combination7 = []
        f_combination8 = []
        f_combination9 = []

        for xs in itertools.product([self.f1_prop], f2_prop):
            if xs[0] != xs[1]:
                f1_ = np.float64(xs[0])
                f2_ = np.float64(xs[1])

                if f1_ - f2_steps_new < f2_ < f1_ + f2_steps_new:
                    pass
                else:
                    # depending on selected i-ratio & lifetime
                    if self.lifetime_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                        f_combination1 = fp.GUI_visualisation(tau_p=self.tau_phos1, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio1, er_=self.error_prop,
                                                              f_combination=f_combination1)
                    if self.lifetime_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                        f_combination4 = fp.GUI_visualisation(tau_p=self.tau_phos1, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio2, er_=self.error_prop,
                                                              f_combination=f_combination4)
                    if self.lifetime_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                        f_combination7 = fp.GUI_visualisation(tau_p=self.tau_phos1, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio3, er_=self.error_prop,
                                                              f_combination=f_combination7)

                    if self.lifetime2_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                        f_combination2 = fp.GUI_visualisation(tau_p=self.tau_phos2, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio1, er_=self.error_prop,
                                                              f_combination=f_combination2)
                    if self.lifetime2_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                        f_combination5 = fp.GUI_visualisation(tau_p=self.tau_phos2, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio2, er_=self.error_prop,
                                                              f_combination=f_combination5)
                    if self.lifetime2_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                        f_combination8 = fp.GUI_visualisation(tau_p=self.tau_phos2, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio3, er_=self.error_prop,
                                                              f_combination=f_combination8)

                    if self.lifetime3_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                        f_combination3 = fp.GUI_visualisation(tau_p=self.tau_phos3, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio1, er_=self.error_prop,
                                                              f_combination=f_combination3)
                    if self.lifetime3_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                        f_combination6 = fp.GUI_visualisation(tau_p=self.tau_phos3, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio2, er_=self.error_prop,
                                                              f_combination=f_combination6)
                    if self.lifetime3_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                        f_combination9 = fp.GUI_visualisation(tau_p=self.tau_phos3, f1_=f1_, f2_=f2_,
                                                              i_ratio=self.i_ratio3, er_=self.error_prop,
                                                              f_combination=f_combination9)
            else:
                pass

        if self.lifetime_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            signals_ = pd.DataFrame(f_combination1)
            signals_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                'er_rel(i, max) [%]']
            self.signals = signals_.round(9)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            signals2_ = pd.DataFrame(f_combination2)
            signals2_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals2 = signals2_.round(9)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            signals3_ = pd.DataFrame(f_combination3)
            signals3_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals3 = signals3_.round(9)

        if self.lifetime_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            signals4_ = pd.DataFrame(f_combination4)
            signals4_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals4 = signals4_.round(9)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            signals5_ = pd.DataFrame(f_combination5)
            signals5_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals5 = signals5_.round(9)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            signals6_ = pd.DataFrame(f_combination6)
            signals6_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals6 = signals6_.round(9)

        if self.lifetime_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            signals7_ = pd.DataFrame(f_combination7)
            signals7_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals7 = signals7_.round(9)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            signals8_ = pd.DataFrame(f_combination8)
            signals8_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals8 = signals8_.round(9)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            signals9_ = pd.DataFrame(f_combination9)
            signals9_.columns = ['f1 [Hz]', 'f2 [Hz]', 'phi(f1) [°]', 'phi(f2) [°]', 'tau [µs]', 'tau(min) [µs]',
                                 'tau(max) [µs]', 'er(tau, min) [%]', 'er(tau, max) [%]', 'i_ratio', 'i_ratio(min)',
                                 'i_ratio(max)', 'er_abs(i, min) [%]', 'er_abs(i, max) [%]', 'er_rel(i, min) [%]',
                                 'er_rel(i, max) [%]']
            self.signals9 = signals9_.round(9)

    # -------------------------------------------------------------------------------------------------------------
    # visualisation of the superimposed self.signals
    # -------------------------------------------------------------------------------------------------------------
        ylim_t = []
        ylim_I = []
        if self.lifetime_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos1, signals=self.signals, ls='solid', color='τ1',
                                 label_='τ1, I1', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos1, error_lifetime=self.error_prop,
                                        signals=self.signals, ylim=ylim_t, label_='τ1, I1', color='τ1, I1',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos1, er=self.error_prop*80, signals=self.signals,
                                   label_='τ1, I1', color='τ1, I1', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals, label_='τ1, I1',
                                         ylim=ylim_I, color='τ1, I1', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos1, signals=self.signals, er=self.error_prop*80,
                                    label_='τ1, I1',  color='τ1, I1', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos1, signals=self.signals, er=self.error_prop*80,
                                    label_='τ1, I1', color='τ1, I1', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos2, signals=self.signals2, ls='solid', color='τ2',
                                 label_='τ2, I1', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos2, error_lifetime=self.error_prop,
                                        signals=self.signals2, ylim=ylim_t, label_='τ2, I1', color='τ2, I1',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos2, er=self.error_prop*80, signals=self.signals2,
                                   label_='τ2, I1', color='τ2, I1', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals2, label_='τ2, I1',
                                         ylim=ylim_I, color='τ2, I1', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos2, signals=self.signals2, er=self.error_prop*80,
                                    label_='τ2, I1', color='τ2, I1', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos2, signals=self.signals2, er=self.error_prop*80,
                                    label_='τ2, I1', color='τ2, I1', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos3, signals=self.signals3, ls='solid', color='τ3',
                                 label_='τ3, I1', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos3, error_lifetime=self.error_prop,
                                        signals=self.signals3, ylim=ylim_t, label_='τ3, I1', color='τ3, I1',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos3, er=self.error_prop*80, signals=self.signals3,
                                   label_='τ3, I1', color='τ3, I1', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals3, color='τ3, I1',
                                         label_='τ3, I1', ylim=ylim_I, ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos3, signals=self.signals3, er=self.error_prop*80,
                                    label_='τ3, I1', color='τ3, I1', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos3, signals=self.signals3, er=self.error_prop*80,
                                    label_='τ3, I1', color='τ3, I1', ax=self.ax_intensity_er, fig=self.fig_intensity_er)

        if self.lifetime_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos1, signals=self.signals4, ls='densely dotted', color='τ1',
                                 label_='τ1, I2', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos1, error_lifetime=self.error_prop,
                                        signals=self.signals4, ylim=ylim_t, label_='τ1, I2', color='τ1, I2',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos1, er=self.error_prop*80, signals=self.signals4,
                                   label_='τ1, I2', color='τ1, I2', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals4, color='τ1, I2',
                                         ylim=ylim_I, label_='τ1, I2', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos1, signals=self.signals4, er=self.error_prop*80,
                                    label_='τ1, I2', color='τ1, I2', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos1, signals=self.signals4, er=self.error_prop*80,
                                    label_='τ1, I2', color='τ1, I2', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos2, signals=self.signals5, ls='densely dotted',
                                 color='τ2', label_='τ2, I2', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos2, error_lifetime=self.error_prop,
                                        signals=self.signals5, ylim=ylim_t, label_='τ2, I2', color='τ2, I2',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos2, er=self.error_prop*80, signals=self.signals5,
                                   label_='τ2, I2', color='τ2, I2', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals5, label_='τ2, I2',
                                         ylim=ylim_I, color='τ2, I2', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos2, signals=self.signals5, er=self.error_prop*80,
                                    label_='τ2, I2', color='τ2, I2', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos2, signals=self.signals5, er=self.error_prop*80,
                                    label_='τ2, I2', color='τ2, I2', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos3, signals=self.signals6, ls='densely dotted',
                                 color='τ3', label_='τ3, I2', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos3, error_lifetime=self.error_prop,
                                        signals=self.signals6, ylim=ylim_t, label_='τ3, I2', color='τ3, I2',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos3, er=self.error_prop*80, signals=self.signals6,
                                   label_='τ3, I2', color='τ3, I2', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals6, color='τ3, I2',
                                         ylim=ylim_I, label_='τ3, I2', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos3, signals=self.signals6, er=self.error_prop*80,
                                    label_='τ3, I2', color='τ3, I2', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos3, signals=self.signals6, er=self.error_prop*80,
                                    label_='τ3, I2', color='τ3, I2', ax=self.ax_intensity_er, fig=self.fig_intensity_er)

        if self.lifetime_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos1, signals=self.signals7, ls='densely dashdotdotted',
                                 label_='τ1, I3', color='τ1', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos1, error_lifetime=self.error_prop,
                                        signals=self.signals7, ylim=ylim_t, label_='τ1, I3', color='τ1, I3',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos1, er=self.error_prop*80, signals=self.signals7,
                                   label_='τ1, I3', color='τ1, I3', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals7, label_='τ1, I3',
                                         ylim=ylim_I, color='τ1, I3', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos1, signals=self.signals7, er=self.error_prop*80,
                                    label_='τ1, I3', color='τ1, I3', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos1, signals=self.signals7, er=self.error_prop*80,
                                    label_='τ1, I3', color='τ1, I3', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime2_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos2, signals=self.signals8, ls='densely dashdotdotted',
                                 label_='τ2, I3', color='τ2', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos2, error_lifetime=self.error_prop,
                                        signals=self.signals8, ylim=ylim_t, label_='τ2, I3', color='τ2, I3',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos2, er=self.error_prop*80, signals=self.signals8,
                                   label_='τ2, I3', color='τ2, I3', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals8, label_='τ2, I3',
                                         ylim=ylim_I, color='τ2, I3', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos2, signals=self.signals8, er=self.error_prop*80,
                                    label_='τ2, I3', color='τ2, I3', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos2, signals=self.signals8, er=self.error_prop*80,
                                    label_='τ2, I3', color='τ2, I3', ax=self.ax_intensity_er, fig=self.fig_intensity_er)
        if self.lifetime3_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            # phase angle
            self.plot_phaseangle(lifetime_phosphor=self.tau_phos3, signals=self.signals9, ls='densely dashdotdotted',
                                 label_='τ3, I3', color='τ3', ax=self.ax_phaseangle, fig=self.fig_phaseangle)
            # lifetime
            ylim_t = self.plot_lifetime(lifetime_phosphor=self.tau_phos3, error_lifetime=self.error_prop,
                                        signals=self.signals9, ylim=ylim_t, label_='τ3, I3', color='τ3, I3',
                                        ax=self.ax_lifetime, fig=self.fig_lifetime)
            # life time relative error
            self.plot_lifetime_err(lifetime_phosphor=self.tau_phos3, er=self.error_prop*80, signals=self.signals9,
                                   label_='τ3, I3', color='τ3, I3', ax=self.ax_lifetime_er, fig=self.fig_lifetime_err)
            # intensity ratio
            ylim_I = self.plot_intensity(error_intensity=self.error_prop*150, signals=self.signals9, label_='τ3, I3',
                                         ylim=ylim_I, color='τ3, I3', ax=self.ax_intensity, fig=self.fig_intensity)
            # absolute error intensity ratio
            self.plot_intensity_abs(lifetime_phosphor=self.tau_phos3, signals=self.signals9, er=self.error_prop*80,
                                    label_='τ3, I3', color='τ3, I3', ax=self.ax_intensity_abs,
                                    fig=self.fig_intensity_abs)
            # relative error intensity ratio
            self.plot_intensity_rel(lifetime_phosphor=self.tau_phos3, signals=self.signals9, er=self.error_prop*80,
                                    label_='τ3, I3', color='τ3, I3', ax=self.ax_intensity_er, fig=self.fig_intensity_er)

    # -------------------------------------------------------------------------------------------------------------
    # results reported in message box
    # -------------------------------------------------------------------------------------------------------------
        self.run_sim_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                          "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: "
                                          "#077487; border-style: solid; border-radius: 7; padding: 5px; font-size: "
                                          "10px; padding-left: 1px; padding-right: 5px; min-height: 10px; max-height:"
                                          " 18px;")

        self.message.setText('Error propagation successfully conducted.\n')
        self.message.append('-----------------------------------------------------------------')
        self.message.append('INPUT parameter')
        self.message.append('f1:\t\t {:.0f} Hz'.format(self.f1_prop))
        self.message.append('assumed uncertainty:\t {} °'.format(self.error_prop))

        if self.lifetime_checkbox.isChecked() is True:
            self.message.append('tau 1:\t\t {:.2f} µs'.format((self.tau_phos1*1E6)))
        if self.lifetime2_checkbox.isChecked() is True:
            self.message.append('tau 2:\t\t {:.2f} µs'.format((self.tau_phos2*1E6)))
        if self.lifetime3_checkbox.isChecked() is True:
            self.message.append('tau 3:\t\t {:.2f} µs'.format((self.tau_phos3*1E6)))
        self.message.append('-----------------------------------------------------------------')
        self.message.append('OUTPUT results')

        self.message_int.setText('Error propagation successfully conducted.\n')
        self.message_int.append('-----------------------------------------------------------------')
        self.message_int.append('INPUT parameter')
        self.message_int.append('f1:\t\t {:.0f} Hz'.format(self.f1_prop))
        self.message_int.append('assumed uncertainty:\t {:.2f} °'.format(self.error_prop))

        if self.intensity_checkbox.isChecked() is True:
            self.message_int.append('I-ratio 1:\t\t {:.2e}'.format(self.i_ratio1))
        if self.intensity2_checkbox.isChecked() is True:
            self.message_int.append('I-ratio 2:\t\t {:.2e}'.format(self.i_ratio2))
        if self.intensity3_checkbox.isChecked() is True:
            self.message_int.append('I-ratio 3:\t\t {:.2e}'.format(self.i_ratio3))
        self.message_int.append('-----------------------------------------------------------------')
        self.message_int.append('OUTPUT results')

        # ---------------------------------------------------------------------------
        # report for saving
        highlightColor = QColor(222, 122, 34)
        blackColor = QColor(0, 0, 0)

        if self.lifetime_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            self.res = self.report_messagebox(self.signals)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 1, I-ratio 1')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 1, I-ratio 1')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime2_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            self.res2 = self.report_messagebox(self.signals2)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 2, I-ratio 1')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res2['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res2['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res2['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res2['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res2['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res2['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res2['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 2, I-ratio 1')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res2['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res2['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res2['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res2['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res2['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res2['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res2['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime3_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
            self.res3 = self.report_messagebox(self.signals3)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 3, I-ratio 1')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res3['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res3['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res3['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res3['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res3['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res3['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res3['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 3, I-ratio 1')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res3['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res3['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res3['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res3['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res3['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res3['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res3['er_rel(i-ratio)_max']))
            self.message_int.append('\n')

        if self.lifetime_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            self.res4 = self.report_messagebox(self.signals4)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 1, I-ratio 2')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res4['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res4['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res4['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res4['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res4['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res4['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res4['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 1, I-ratio 2')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res4['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res4['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res4['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res4['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res4['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res4['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res4['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime2_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            self.res5 = self.report_messagebox(self.signals5)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 2, I-ratio 2')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res5['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res5['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res5['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res5['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res5['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res5['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res5['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 2, I-ratio 2')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res5['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res5['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res5['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res5['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res5['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res5['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res5['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime3_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
            self.res6 = self.report_messagebox(self.signals6)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 3, I-ratio 2')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res6['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res6['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res6['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res6['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res6['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res6['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res6['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 3, I-ratio 2')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res6['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res6['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res6['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res6['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res6['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res6['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res6['er_rel(i-ratio)_max']))
            self.message_int.append('\n')

        if self.lifetime_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            self.res7 = self.report_messagebox(self.signals7)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 1, I-ratio 3')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res7['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res7['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res7['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res7['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res7['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res7['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res7['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 1, I-ratio 3')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res7['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res7['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res7['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res7['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res7['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res7['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res7['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime2_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            self.res8 = self.report_messagebox(self.signals8)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 2, I-ratio 3')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res8['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res8['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res8['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res8['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res8['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res8['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res8['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 2, I-ratio 3')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res8['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res8['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res8['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res8['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res8['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res8['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res8['er_rel(i-ratio)_max']))
            self.message_int.append('\n')
        if self.lifetime3_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
            self.res9 = self.report_messagebox(self.signals9)
            self.message.setTextColor(highlightColor)
            self.message.append('tau 3, I-ratio 3')
            self.message.setTextColor(blackColor)
            self.message.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res9['f2']))
            self.message.append('phi_f1:\t\t {:.2f} °'.format(self.res9['phi_f1']))
            self.message.append('phi_f2:\t\t {:.2f} °'.format(self.res9['phi_f2']))
            self.message.append('tau(min):\t\t {:.2f} µs'.format(self.res9['tau_min']))
            self.message.append('tau(max):\t\t {:.2f} µs'.format(self.res9['tau_max']))
            self.message.append('rel. error tau(min):\t {:.2f} %'.format(self.res9['er(tau)_min']))
            self.message.append('rel. error tau(max):\t {:.2f} %'.format(self.res9['er(tau)_max']))
            self.message.append('\n')
            self.message_int.setTextColor(highlightColor)
            self.message_int.append('tau 3, I-ratio 3')
            self.message_int.setTextColor(blackColor)
            self.message_int.append('f2 optimal:\t\t {:.1f} Hz'.format(self.res9['f2']))
            self.message_int.append('I ratio (min):\t\t {:.2e}'.format(self.res9['i-ratio(min)']))
            self.message_int.append('I ratio (max):\t\t {:.2e}'.format(self.res9['i-ratio(max)']))
            self.message_int.append('deviation I ratio (min):\t {:.2e}'.format(self.res9['er(i-ratio)_min']))
            self.message_int.append('deviation I ratio (max):\t {:.2e}'.format(self.res9['er(i-ratio)_max']))
            self.message_int.append('rel. error I ratio (min):\t {:.2e}'.format(self.res9['er_rel(i-ratio)_min']))
            self.message_int.append('rel. error I ratio (max):\t {:.2e}'.format(self.res9['er_rel(i-ratio)_max']))
            self.message_int.append('\n')

        print('error propagation finished')
        print('#-------------------------------------------------------------------')

# ---------------------------------------------------------------------------------------------------------------------
# pH / O2 sensing
# ---------------------------------------------------------------------------------------------------------------------
    def plot_calibration_pH_pO2(self, pH_calib, pO2_calib, pk_a, Ksv1, Ksv2, int_f, tau_quot, f1, ax1, f2, ax2):
        # preparation of figure plot
        if ax1 is None:
            f1 = plt.figure()
            ax1 = f1.gca()
        if ax2 is None:
            f2 = plt.figure()
            ax2 = f2.gca()

        # ---------------------------------------------------------------------------------
        # pH sensing
        ax1.plot(int_f.index, int_f, lw=0.75, color='#07575B')
        ax1.axvline(pH_calib[0], color='k', lw=0.5, ls='--')
        ax1.axvline(pH_calib[1], color='k', lw=0.5, ls='--')
        ax1.set_xlim(0, 15)

        ax1.legend(['pK$_a$ = {:.2f}'.format(pk_a), 'pH$_1$ = {:.1f}'.format(pH_calib[0]),
                    'pH$_2$ = {:.1f}'.format(pH_calib[1])], frameon=True, framealpha=0.5, fancybox=True, loc=0,
                   fontsize=9)

        # ---------------------------------------------------------------------------------
        # O2 sensing
        ax2.plot(tau_quot.index, 1/tau_quot, color='#6fb98f', lw=1.) #004445
        ax2.axvline(pO2_calib[0], color='black', lw=0.5, ls='--')
        ax2.axvline(pO2_calib[1], color='black', lw=0.5, ls='--')

        xmax = tau_quot.index[-1]*1.05
        ymax = 1/tau_quot.values[-1][0]*1.05
        ymin = 1/tau_quot.values[0][0]*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        ax2.legend(['Ksv1 = {:.2e}, Ksv2 = {:.2e}'.format(Ksv1, Ksv2),
                    'pO$_{}$ = {:.1f} hPa'.format('2,1', pO2_calib[0]),
                    'pO$_{}$ = {:.1f} hPa'.format('2,2', pO2_calib[1])], frameon=True, framealpha=0.5, fancybox=True,
                   loc=0, fontsize=8)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_calibration_pH_pO2_measurement(self, pH_calib, pO2_calib, int_f_c0, tau_quot, pk_a, K_sv, f1, ax1, f2,
                                            ax2):
        # preparation of figure plot
        if ax1 is None:
            f1 = plt.figure()
            ax1 = f1.gca()
        if ax2 is None:
            f2 = plt.figure()
            ax2 = f2.gca()

        # ---------------------------------------------------------------------------------
        # pH sensing at 2 different pO2 contents
        p1, = ax1.plot(int_f_c0.index, int_f_c0['mean'], color='navy', lw=0.5, label='{:.1f}hPa'.format(pO2_calib[0]))
        ax1.plot(int_f_c0.index, int_f_c0['min'], color='k', ls='--', lw=0.5)
        ax1.plot(int_f_c0.index, int_f_c0['max'], color='k', ls='--', lw=0.5)

        ax1.legend(['pK$_a$ = {:.2f}'.format(pk_a), 'pH$_1$ = {:.1f}'.format(pH_calib[0]),
                    'pH$_2$ = {:.1f}'.format(pH_calib[1])], frameon=True, framealpha=0.5, fancybox=True, loc=0,
                   fontsize=9)

        ax1.fill_between(int_f_c0.index, int_f_c0['min'], int_f_c0['max'], color='grey', alpha=0.15)
        ax1.axvline(pH_calib[0], color='black', lw=0.5, ls='--')
        ax1.axvline(pH_calib[1], color='black', lw=0.5, ls='--')
        ax1.set_xlim(0, 15)

        # ---------------------------------------------------------------------------------
        # pO2 sensing
        ax2.plot(tau_quot.index, 1/tau_quot[1], color='#2a3132') #6fb98f
        ax2.plot(tau_quot.index, 1/tau_quot[0], color='k', ls='--', lw=0.5)
        ax2.plot(tau_quot.index, 1/tau_quot[2], color='k', ls='--', lw=0.5)

        ax2.legend(['Ksv = {:.2f}'.format(K_sv[0]), 'pO$_{}$ = {:.1f} hPa'.format('2,1', pO2_calib[0]),
                    'pO$_{}$ = {:.1f} hPa'.format('2,2', pO2_calib[1])], frameon=True, framealpha=0.5, fancybox=True,
                   loc=0, fontsize=9)

        ax2.fill_between(tau_quot.index, 1/tau_quot[0], 1/tau_quot[2], color='grey', alpha=0.2)
        ax2.axvline(pO2_calib[0], color='black', lw=0.5, ls='--')
        ax2.axvline(pO2_calib[1], color='black', lw=0.5, ls='--')

        xmax = tau_quot.index[-1]*1.05
        xmin = tau_quot.index[0]*0.95
        if tau_quot.index[0] == 0.0:
            xmin = -5.

        ymax = 1/tau_quot.values[-1][0]*1.05
        ymin = 1/tau_quot.values[0][0]*0.95
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_results_pH_pO2(self, pH, pH_calib, pO2_calc, pO2_calib, tauP, intF, f1, ax1, f2, ax2):

        # pH sensing - calibration points
        ax1.plot(intF.index, intF['mean'], color='navy', lw=0.75)
        ax1.plot(intF.index, intF['min'], color='k', ls='--', lw=0.5)
        ax1.plot(intF.index, intF['max'], color='k', ls='--', lw=0.5)
        ax1.fill_between(intF.index, intF['min'], intF['max'], color='grey', alpha=0.15)

        # measurement points
        ax1.axvline(pH[0], color='k', ls='--', lw=.4)
        ax1.axvline(pH[2], color='k', ls='--', lw=.4)
        ax1.axvspan(pH[0], pH[2], color='#f0810f', alpha=0.2)

        # find closest values
        iF_meas_min = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pH[0])
        iF_meas_max = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pH[2])

        # linear regression to pH measured
        arg_min = stats.linregress(x=pH_calib, y=iF_meas_min[2:])
        arg_max = stats.linregress(x=pH_calib, y=iF_meas_max[2:])
        y_min = arg_min[0] * pH[0] + arg_min[1]
        y_max = arg_max[0] * pH[2] + arg_max[1]
        ax1.axhline(y_min, color='k', ls='--', lw=.4)
        ax1.axhline(y_max, color='k', ls='--', lw=.4)
        ax1.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        ax1.legend(['pH = {:.2f} ± {:.2e}'.format(pH.mean(), pH.std())], frameon=True, framealpha=0.5, fancybox=True,
                   loc=0, fontsize=9)
        y_max = intF['max'].max()*1.05
        if intF['min'].min() < 1:
            if intF['min'].min() < 0:
                y_min = -np.abs(intF['min'].min())*0.95
            else:
                y_min = -5.
        else:
            y_min = intF['min'].min()*0.95
        ax1.set_ylim(y_min, y_max)

    # --------------------------------------------------------------------------------
        # O2 sensing - calibration points
        ax2.plot(tauP.index, tauP[1], color='#021c1e', lw=.75)
        ax2.plot(tauP[0], color='k', lw=.25, ls='--')
        ax2.plot(tauP[2], color='k', lw=.25, ls='--')
        ax2.fill_between(tauP.index, tauP[0], tauP[2], color='grey', alpha=0.2)

        # measurement points
        ax2.axvline(pO2_calc[0], color='k', lw=0.4, ls='--')
        ax2.axvline(pO2_calc[2], color='k', lw=0.4, ls='--')
        ax2.axvspan(pO2_calc[0], pO2_calc[2], color='#f0810f', alpha=0.2)

        # find closest value
        tauq_meas_min = fp.find_closest_value_(index=tauP.index, data=tauP[0].values, value=pO2_calc[0])
        tauq_meas_max = fp.find_closest_value_(index=tauP.index, data=tauP[2].values, value=pO2_calc[2])

        # linear regression to pO2 measured
        arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
        arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
        y_min = arg_min[0] * pO2_calc[0] + arg_min[1]
        y_max = arg_max[0] * pO2_calc[2] + arg_max[1]
        ax2.axhline(y_min, color='k', lw=0.4, ls='--')
        ax2.axhline(y_max, color='k', lw=0.4, ls='--')
        ax2.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        ax2.legend(['pO2 = {:.2f} ± {:.2f}'.format(pO2_calc.mean(), pO2_calc.std())], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)
        xmax = tauP.index[-1]*1.05
        ymax = tauP.values[0].max()*1.05
        ymin = tauP.values[-1].min()*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_results_pH_pO2_meas(self, pH, pH_calib, pO2_calc, pO2_calib, intF, tauP, f1, ax1, f2, ax2):

        # pH sensing - calibration points
        ax1.plot(intF[0].index, intF[0]['mean'], color='#336b87', lw=0.75)
        ax1.plot(intF[0].index, intF[0]['min'], color='k', ls='--', lw=0.5)
        ax1.plot(intF[0].index, intF[0]['max'], color='k', ls='--', lw=0.5)
        ax1.fill_between(intF[0].index, intF[0]['min'], intF[0]['max'], color='grey', alpha=0.15)

        ax1.plot(intF[len(pH)-1].index, intF[len(pH)-1]['mean'], color='#336b87', lw=0.75)
        ax1.plot(intF[len(pH)-1].index, intF[len(pH)-1]['min'], color='k', ls='-.', lw=0.5)
        ax1.plot(intF[len(pH)-1].index, intF[len(pH)-1]['max'], color='k', ls='-.', lw=0.5)
        ax1.fill_between(intF[len(pH)-1].index, intF[len(pH)-1]['min'], intF[len(pH)-1]['max'], color='grey', alpha=0.1)

        # 1st measurement point
        ax1.axvline(pH[0][0], color='k', ls='-', lw=.4)
        ax1.axvline(pH[0][2], color='k', ls='-', lw=.4)
        ax1.axvspan(pH[0][0], pH[0][2], color='#f0810f', alpha=0.4)

        # last measurement point
        ax1.axvline(pH[len(pH)-1][0], color='k', ls='-.', lw=.4)
        ax1.axvline(pH[len(pH)-1][2], color='k', ls='-.', lw=.4)
        ax1.axvspan(pH[len(pH)-1][0], pH[len(pH)-1][2], color='#f0810f', alpha=0.2)

        # find closest values to 1st point
        iF_meas_min = fp.find_closest_value_(index=intF[0].index, data=intF[0]['mean'].values, value=pH[0][0])
        iF_meas_max = fp.find_closest_value_(index=intF[0].index, data=intF[0]['mean'].values, value=pH[0][2])
        # find closest values to last point
        iF_meas_min2 = fp.find_closest_value_(index=intF[len(pH)-1].index, data=intF[len(pH)-1]['mean'].values,
                                              value=pH[len(pH)-1][0])
        iF_meas_max2 = fp.find_closest_value_(index=intF[len(pH)-1].index, data=intF[len(pH)-1]['mean'].values,
                                              value=pH[len(pH)-1][2])

        # linear regression to pH measured
        arg_min = stats.linregress(x=pH_calib, y=iF_meas_min[2:])
        arg_max = stats.linregress(x=pH_calib, y=iF_meas_max[2:])
        y_min = arg_min[0] * pH[0][0] + arg_min[1]
        y_max = arg_max[0] * pH[0][2] + arg_max[1]
        ax1.axhline(y_min, color='k', ls='--', lw=.4)
        ax1.axhline(y_max, color='k', ls='--', lw=.4)
        ax1.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

        arg_min2 = stats.linregress(x=pH_calib, y=iF_meas_min2[2:])
        arg_max2 = stats.linregress(x=pH_calib, y=iF_meas_max2[2:])
        y_min2 = arg_min2[0] * pH[len(pH)-1][0] + arg_min2[1]
        y_max2 = arg_max2[0] * pH[len(pH)-1][2] + arg_max2[1]
        ax1.axhline(y_min2, color='k', ls='-.', lw=.4)
        ax1.axhline(y_max2, color='k', ls='-.', lw=.4)
        ax1.axhspan(y_min2, y_max2, color='#f0810f', alpha=0.2)

        # legend
        ax1.legend(['pH$_{}$ = {:.2f} ± {:.2f}'.format('1', pH[0].mean(), pH[0].std()),
                    'pH$_{}$ = {:.2f} ± {:.2f}'.format(len(pH), pH[len(pH)-1].mean(), pH[len(pH)-1].std())],
                   frameon=True, framealpha=0.5, fancybox=True, loc=0, fontsize=9)

    # --------------------------------------------------------------------------------
        # O2 sensing - calibration points
        ax2.plot(tauP[0].index, tauP[0][1], color='#021c1e', lw=.75)
        ax2.plot(tauP[0][0], color='k', lw=.25, ls='--')
        ax2.plot(tauP[0][2], color='k', lw=.25, ls='--')
        ax2.fill_between(tauP[0].index, tauP[0][0], tauP[0][2], color='grey', alpha=0.4)

        ax2.plot(tauP[len(pH)-1].index, tauP[len(pH)-1][1], color='#021c1e', lw=.75)
        ax2.plot(tauP[len(pH)-1][0], color='k', lw=.25, ls='-.')
        ax2.plot(tauP[len(pH)-1][2], color='k', lw=.25, ls='-.')
        ax2.fill_between(tauP[len(pH)-1].index, tauP[len(pH)-1][0], tauP[len(pH)-1][2], color='grey', alpha=0.2)

        # 1st measurement point
        ax2.axvline(pO2_calc[0][0], color='k', lw=0.4, ls='--')
        ax2.axvline(pO2_calc[0][2], color='k', lw=0.4, ls='--')
        ax2.axvspan(pO2_calc[0][0], pO2_calc[0][2], color='#f0810f', alpha=0.4)
        # last measurement point
        ax2.axvline(pO2_calc[len(pH)-1][0], color='k', lw=0.4, ls='-.')
        ax2.axvline(pO2_calc[len(pH)-1][2], color='k', lw=0.4, ls='-.')
        ax2.axvspan(pO2_calc[len(pH)-1][0], pO2_calc[len(pH)-1][2], color='#f0810f', alpha=0.2)

        # find closest value
        tauq_meas_min = fp.find_closest_value_(index=tauP[0].index, data=tauP[0][0].values, value=pO2_calc[0][0])
        tauq_meas_max = fp.find_closest_value_(index=tauP[0].index, data=tauP[0][2].values, value=pO2_calc[0][2])
        tauq_meas_min2 = fp.find_closest_value_(index=tauP[len(pH)-1].index, data=tauP[len(pH)-1][0].values,
                                                value=pO2_calc[len(pH)-1][0])
        tauq_meas_max2 = fp.find_closest_value_(index=tauP[len(pH)-1].index, data=tauP[len(pH)-1][2].values,
                                                value=pO2_calc[len(pH)-1][2])

        # linear regression to pO2 measured
        arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
        arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
        y_min = arg_min[0] * pO2_calc[0][0] + arg_min[1]
        y_max = arg_max[0] * pO2_calc[0][2] + arg_max[1]
        ax2.axhline(y_min, color='k', lw=0.4, ls='--')
        ax2.axhline(y_max, color='k', lw=0.4, ls='--')
        ax2.axhspan(y_min, y_max, color='#f0810f', alpha=0.4)

        arg_min2 = stats.linregress(x=pO2_calib, y=tauq_meas_min2[2:])
        arg_max2 = stats.linregress(x=pO2_calib, y=tauq_meas_max2[2:])
        y_min2 = arg_min2[0] * pO2_calc[len(pH)-1][0] + arg_min2[1]
        y_max2 = arg_max2[0] * pO2_calc[len(pH)-1][2] + arg_max2[1]
        ax2.axhline(y_min2, color='k', lw=0.4, ls='-.')
        ax2.axhline(y_max2, color='k', lw=0.4, ls='-.')
        ax2.axhspan(y_min2, y_max2, color='#f0810f', alpha=0.2)

        # legend
        ax2.legend(['pO2$_{}$ = {:.2f} ± {:.2f}'.format('1', pO2_calc[0].mean(), pO2_calc[0].std()),
                    'pO2$_{}$ = {:.2f} ± {:.2f}'.format(len(pH), pO2_calc[len(pH)-1].mean(), pO2_calc[len(pH)-1].std())],
                   frameon=True, framealpha=0.5, fancybox=True, loc=0, fontsize=9)
        xmax = tauP[0].index[-1]*1.05
        ymax = tauP[0].values[0].max()*1.05
        ymin = tauP[0].values[-1].min()*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def open_conversion(self):
        self.load_tauP_intP_conv_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px;"
                                                      "border-color: #077487; border-style: solid; border-radius: 7;"
                                                      "padding: 5px; font-size: 10px; padding-left: 1px; padding-right:"
                                                      "5px; min-height: 10px; max-height: 18px;")

        path_gui_ = os.path.abspath("GUI_dualsensors.py")
        path_gui = path_gui_.split('GUI_dualsensors')[0]

        self.fname_comp = QFileDialog.getOpenFileName(self, "Select specific txt file for temperature compensation",
                                                      path_gui)[0]
        self.load_tauP_intP_conv_button.setStyleSheet("color: white; background-color: "
                                                      "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                      "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                      "border-style: solid; border-radius: 7; padding: 5px; "
                                                      "font-size: 10px; padding-left: 1px; padding-right: 5px; "
                                                      "min-height: 10px; max-height: 18px;")
        if not self.fname_comp:
            return
        # next function -> load file
        self.read_conversion(self.fname_comp)

    def read_conversion(self, fname_comp):
        self.conversion_edit.clear()
        try:
            self.conv_file = fname_comp
        except:
            conv_file_load_failed = QMessageBox()
            conv_file_load_failed.setIcon(QMessageBox.Information)
            conv_file_load_failed.setText('Invalid file for tauP -> intP conversion during pH/O2 dual-sensing!')
            conv_file_load_failed.setInformativeText('Choose another file from path...')
            conv_file_load_failed.setWindowTitle('Error!')
            conv_file_load_failed.buttonClicked.connect(self.open_conversion)
            conv_file_load_failed.exec_()
            return
        # ---------------------------------------------------------------------
        # write (part of the path) to text line
        parts = fname_comp.split('/')
        if 3 < len(parts) <= 6:
            self.conv_file_part = '/'.join(parts[3:])
        elif len(parts) > 6:
            self.conv_file_part = '/'.join(parts[5:])
        else:
            self.conv_file_part = '/'.join(parts)
        self.conversion_edit.insertPlainText(str(self.conv_file_part))

        # ---------------------------------------------------------------------
        # load txt file
        self.df_conv_tauP_intP = pd.read_csv(self.conv_file, sep='\t', decimal='.', index_col=0, header=None,
                                             encoding='latin-1')

# ---------------------------------------------------------------------------------------------------------------------
# dual sensor calculations 3rd tab
    def pH_oxygen_sensing(self):
        print('#--------------------------------------')
        print('pH / O2 dualsensing')
        self.run_tab3_button.setStyleSheet(
            "color: white; background-color: #2b5977; border-width: 1px; border-color: #077487; border-style: solid; "
            "border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px;"
            " max-height: 18px;")

        # status of progressbar
        self.progress_tab3.setValue(0)

        # clear everything
        self.message_tab3.clear()

        # calibration pH at 2 frequencies
        self.ax_pH_calib.cla()
        self.fig_pH_calib.clear()
        self.ax_pH_calib = self.fig_pH_calib.gca()
        self.ax_pH_calib.set_xlim(0, 15)
        self.ax_pH_calib.set_xlabel('pH', fontsize=9)
        self.ax_pH_calib.set_ylabel('Rel. intensity $I_F$ [%]', fontsize=9)
        self.ax_pH_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pH_calib.canvas.draw()

        # calibration pO2 sensing
        self.ax_pO2_calib.cla()
        self.fig_pO2_calib.clear()
        self.ax_pO2_calib = self.fig_pO2_calib.gca()
        self.ax_pO2_calib.set_xlim(0, 100)
        self.ax_pO2_calib.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2_calib.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
        self.ax_pO2_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pO2_calib.canvas.draw()

        # O2 sensing
        self.ax_pO2.cla()
        self.fig_pO2.clear()
        self.ax_pO2 = self.fig_pO2.gca()
        self.ax_pO2.set_xlim(0, 100)
        self.ax_pO2.set_ylim(0, 105)
        self.ax_pO2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
        self.ax_pO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pO2.canvas.draw()

        # pH sensing
        self.ax_pH.cla()
        self.fig_pH.clear()
        self.ax_pH = self.fig_pH.gca()
        self.ax_pH.set_xlim(0, 15)
        self.ax_pH.set_ylim(0, 105)
        self.ax_pH.set_xlabel('pH', fontsize=9)
        self.ax_pH.set_ylabel('Rel. intensity $I_F$ [%]', fontsize=9)
        self.ax_pH.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pH.canvas.draw()

        self.progress_tab3.setValue(5)
# ----------------------------------------------------------------------------------------------------------------
        # Entries for calibration points
        for i in range(self.tableCalibration.rowCount()):
            for j in [0, 3]:
                try:
                    self.tableCalibration.item(i, j).text()
                except:
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("2 calibration points for each parameter (pH and O2) are"
                                                               " required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

                if self.tableCalibration.item(i, j).text() == '':
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("2 calibration points for each parameter (pH and O2) are"
                                                               " required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

        # pH list
        self.pH_list = []
        self.pH_order = []
        self.pH_list, self.pH_order = self.extract_calibration_points(para_list=self.pH_list, para_order=self.pH_order,
                                                                      cols=0, calib_type='2point')

        # pO2 list
        self.pO2_list = []
        self.pO2_order = []
        self.pO2_list, self.pO2_order = self.extract_calibration_points(para_list=self.pO2_list, cols=3,
                                                                        para_order=self.pO2_order, calib_type='2point')

        # re-check if there are 2pH and 2pO2 values
        if len(self.pH_list) != len(self.pO2_list):
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("2 calibration points for each parameter (pH and O2) are"
                                                       " required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # combine pH and pO2 to obtain the input order
        calib_points = []
        for i in range(len(self.pH_order)):
            r = self.pH_order[i], self.pO2_order[i]
            calib_points.append(r)

        if len(list(set(calib_points))) < 4:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("4 different calibration points for the analytes (pH and "
                                                       "pO2) are required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # -------------------------------------------------------------
        # check tauP -> intP conversion file
        try:
            self.conv_file
            if self.conv_file is None:
                calibration_para_failed = QMessageBox()
                calibration_para_failed.setIcon(QMessageBox.Information)
                calibration_para_failed.setText("Error for calibration parameters!")
                calibration_para_failed.setInformativeText("File for conversion of tauP --> intP is required!")
                calibration_para_failed.setWindowTitle("Error!")
                calibration_para_failed.exec_()
                return
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("File for conversion of tauP --> intP is required!")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # -----------------------------------------------------------------------------------------------
        # Collect additional input parameters
        [self.f1, self.f2, self.error_assumed, self.int_fluoro_max, self.int_phosphor_max] = self.input_collecting()
        pKa = np.float64(self.pka_tab3_edit.text().replace(',', '.'))
        slope_pH = np.float64(self.slope_tab3_edit.text().replace(',', '.'))
        self.Ksv1 = np.float64(self.Ksv1_tab3_edit.text().replace(',', '.'))
        self.prop_ksv = np.float64(self.Ksv2_tab3_edit.text().replace(',', '.'))
        self.curv_O2 = np.float64(self.curv_O2_tab3_edit.text().replace(',', '.'))
        tau_phosphor_c0 = np.float64(self.lifetime_phos_dualsens_edit.text().replace(',', '.'))
        pO2_range = np.linspace(0, 300, num=int(300/0.5)+1)
        pH_range = np.linspace(0, 14, num=int(14/0.01)+1)
        self.conv_tau_int = pd.read_csv(self.conv_file, sep='\t', index_col=0)
        print('current tab', self.tabs.currentIndex()+1, ' - pH/O2 dualsensor')
        self.progress_tab3.setValue(10)

# -------------------------------------------------------------------------------------------------------------
# if simulation is True
# -------------------------------------------------------------------------------------------------------------
        if self.simulation_checkbox.checkState() == 2:
            print('simulation pH/O2')
            # clear dPhi from calibration/input table (pH in row0, pO2 in row3)
            self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                                   cols=[1, 2, 4, 5, 6])
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[1, 2, 4, 5, 6])

            # Measurement points pH
            try:
                self.tableINPUT.item(0, 0).text()
                if np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')) > 14. or \
                                np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input pH between 0 - 14.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("pH and O2 content have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            # Measurement points pO2
            try:
                self.tableINPUT.item(0, 3).text()
                if np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')) > 250. or \
                                np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input pO2 content between 0 - 250 hPa.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("pH and O2 content have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            self.pH_list.append(np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')))
            self.pO2_list.append(np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')))

            self.progress_tab3.setValue(30)

        # ------------------------------------------------------------------------------
            # Input parameter simulation
            [self.Phi_f1_deg, self.Phi_f2_deg, self.Phi_f1_deg_er, self.Phi_f2_deg_er, self.int_ratio,
             self.para_simulation] = pHox.simulate_phaseangle_pO2_pH(pO2_range=pO2_range, pO2_calib=self.pO2_list[:2],
                                                                     pk_a=pKa, ox_meas=self.pO2_list[2],  f1=self.f1,
                                                                     K_sv1=self.Ksv1, curv_O2=self.curv_O2, f2=self.f2,
                                                                     slope=slope_pH, prop_ksv=self.prop_ksv,
                                                                     pH_range=pH_range, tau_phos0=tau_phosphor_c0,
                                                                     pH_meas=self.pH_list[2], pH_calib=self.pH_list[:2],
                                                                     df_conv_tau_int=self.conv_tau_int,
                                                                     int_phosphor_c0=self.int_phosphor_max,
                                                                     int_fluoro_max=self.int_fluoro_max,
                                                                     er_phase=self.error_assumed, plotting=False,
                                                                     normalize_phosphor=False, fontsize_=13)
            self.progress_tab3.setValue(65)

            # ----------------------------------------------
            # write input parameter to tableCalibration
            for i in range(self.tableCalibration.rowCount()):
                if self.pH_order[i] == self.pH_list[0]:
                    if self.pO2_order[i] == self.pO2_list[0]:
                        int_r = self.int_ratio['fluoro0, phosphor0']
                        phi_f1 = self.Phi_f1_deg['fluoro0, phosphor0']
                        phi_f2 = self.Phi_f2_deg['fluoro0, phosphor0']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                        self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
                    else:
                        int_r = self.int_ratio['fluoro0, phosphor1']
                        phi_f1 = self.Phi_f1_deg['fluoro0, phosphor1']
                        phi_f2 = self.Phi_f2_deg['fluoro0, phosphor1']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                        self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
                else:
                    if self.pO2_order[i] == self.pO2_list[0]:
                        int_r = self.int_ratio['fluoro1, phosphor0']
                        phi_f1 = self.Phi_f1_deg['fluoro1, phosphor0']
                        phi_f2 = self.Phi_f2_deg['fluoro1, phosphor0']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                        self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
                    else:
                        int_r = self.int_ratio['fluoro1, phosphor1']
                        phi_f1 = self.Phi_f1_deg['fluoro1, phosphor1']
                        phi_f2 = self.Phi_f2_deg['fluoro1, phosphor1']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                        self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
            self.tableINPUT.setItem(0, 6, QTableWidgetItem('{:.3f}'.format(self.int_ratio['meas'])))

            if isinstance(self.Phi_f1_deg['meas'], np.float):
                p_meas_f1 = self.Phi_f1_deg['meas']
                p_meas_f2 = self.Phi_f2_deg['meas']
            elif isinstance(self.Phi_f1_deg['meas'], pd.DataFrame):
                p_meas_f1 = self.Phi_f1_deg['meas'].values[0][0]
                p_meas_f2 = self.Phi_f2_deg['meas'].values[0][0]
            else:
                p_meas_f1 = self.Phi_f1_deg['meas'].values[1]
                p_meas_f2 = self.Phi_f2_deg['meas'].values[1]
                if isinstance(p_meas_f1, np.float):
                    pass
                else:
                    p_meas_f1 = p_meas_f1[0]
                    p_meas_f2 = p_meas_f2[0]
            self.tableINPUT.setItem(0, 4, QTableWidgetItem('{:.2f}'.format(p_meas_f1)))
            self.tableINPUT.setItem(0, 5, QTableWidgetItem('{:.2f}'.format(p_meas_f2)))

            self.progress_tab3.setValue(75)

        # ------------------------------------------------------------------------------
            # Plotting calibration
            self.pH_calib = self.pH_list[:-1]
            self.pO2_calib = self.pO2_list[:-1]

            self.plot_calibration_pH_pO2(pH_calib=self.pH_calib, pO2_calib=self.pO2_calib, pk_a=pKa, Ksv1=self.Ksv1,
                                         Ksv2=self.prop_ksv*self.Ksv1, int_f=self.para_simulation['intF'],
                                         tau_quot=self.para_simulation['tau_quot'], f1=self.fig_pH_calib,
                                         ax1=self.ax_pH_calib, f2=self.fig_pO2_calib, ax2=self.ax_pO2_calib)
            self.progress_tab3.setValue(80)

        # ------------------------------------------------------------------------------
            # Dual sensing according to simulated input parameter
            self.Phi_f1_meas_er = [self.Phi_f1_deg['meas'] - self.error_assumed, self.Phi_f1_deg['meas'],
                                   self.Phi_f1_deg['meas'] + self.error_assumed]
            self.Phi_f2_meas_er = [self.Phi_f2_deg['meas'] - self.error_assumed, self.Phi_f2_deg['meas'],
                                   self.Phi_f2_deg['meas'] + self.error_assumed]

            [self.pO2_calc, self.tau_quot, self.tauP, self.intP, self.pH_calc, self.intF_meas, para_TSM,
             self.ampl_total_f1, self.ampl_total_f2, ax_pO2,
             ax_pH] = pHox.pH_oxygen_dualsensor(phi_f1_deg=self.Phi_f1_deg, phi_f2_deg=self.Phi_f2_deg,
                                                phi_f1_meas=self.Phi_f1_deg['meas'],
                                                phi_f2_meas=self.Phi_f2_deg.loc['meas'],
                                                error_phaseangle=self.error_assumed, pO2_range=pO2_range,
                                                pO2_calib=self.pO2_calib, curv_O2=self.curv_O2, prop_ksv=self.prop_ksv,
                                                df_conv_tau_int=self.conv_tau_int,
                                                intP_c0=self.int_phosphor_max, pH_range=pH_range, v50=pKa,
                                                pH_calib=self.pH_calib, slope_pH=slope_pH, f1=self.f1, f2=self.f2,
                                                plotting=False, type_='moderate', method_='std')
            self.progress_tab3.setValue(90)

        # ---------------------------------------
            # Plotting results
            if self.tauP.loc[0, 1] < 1:
                tauP_mikros = self.tauP*1E6
            else:
                tauP_mikros = self.tauP

            if np.isnan(self.pH_calc).any() == True:
                self.plot_results_pH_pO2(pH=self.pH_calc_corr, pH_calib=self.pH_calib, pO2_calc=self.pO2_calc,
                                         pO2_calib=self.pO2_calib, tauP=tauP_mikros, intF=self.intF_meas, f1=self.fig_pH,
                                         ax1=self.ax_pH, f2=self.fig_pO2, ax2=self.ax_pO2)
            else:
                self.plot_results_pH_pO2(pH=self.pH_calc, pH_calib=self.pH_calib, pO2_calc=self.pO2_calc,
                                         pO2_calib=self.pO2_calib, tauP=tauP_mikros, intF=self.intF_meas, f1=self.fig_pH,
                                         ax1=self.ax_pH, f2=self.fig_pO2, ax2=self.ax_pO2)

        # --------------------------------------------------
            # Output - total amplitude reported in calibration table
            if self.int_ratio_checkbox.isChecked() is True:
                for i in range(self.tableCalibration.rowCount()):
                    if self.pH_order[i] == self.pH_list[0]:
                        if self.pO2_order[i] == self.pO2_list[0]:
                            int_r = self.int_ratio['fluoro0, phosphor0']
                        else:
                            int_r = self.int_ratio['fluoro0, phosphor1']
                    else:
                        if self.pO2_order[i] == self.pO2_list[0]:
                            int_r = self.int_ratio['fluoro1, phosphor0']
                        else:
                            int_r = self.int_ratio['fluoro1, phosphor1']
                    self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
                self.tableINPUT.setItem(0, 6, QTableWidgetItem('{:.3f}'.format(self.int_ratio['meas'])))
            else:
                self.int_ratio = None
                for i in range(self.tableCalibration.rowCount()):
                    if self.pH_order[i] == self.pH_list[0]:
                        if self.pO2_order[i] == self.pO2_list[0]:
                            total_ampl1 = self.ampl_total_f1['fluoro0, phosphor0']
                            total_ampl2 = self.ampl_total_f2['fluoro0, phosphor0']
                        else:
                            total_ampl1 = self.ampl_total_f1['fluoro0, phosphor1']
                            total_ampl2 = self.ampl_total_f2['fluoro0, phosphor1']
                    else:
                        if self.pO2_order[i] == self.pO2_list[0]:
                            total_ampl1 = self.ampl_total_f1['fluoro1, phosphor0']
                            total_ampl2 = self.ampl_total_f2['fluoro1, phosphor0']
                        else:
                            total_ampl1 = self.ampl_total_f1['fluoro1, phosphor1']
                            total_ampl2 = self.ampl_total_f2['fluoro1, phosphor1']

                    self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.2f}'.format(total_ampl1[1])))
                    self.tableCalibration.setItem(i, 7, QTableWidgetItem('{:.2f}'.format(total_ampl2[1])))

                self.tableINPUT.setItem(0, 6, QTableWidgetItem('{:.2f}'.format(self.ampl_total_f1['meas'][1])))
                self.tableINPUT.setItem(0, 7, QTableWidgetItem('{:.2f}'.format(self.ampl_total_f2['meas'][1])))

        # --------------------------------------------------
            # Results reported in message box
            self.message_tab3.append('Calculated pO2 = {:.2f} hPa (mean) - interval '
                                     '{:.2f} - {:.2f} hPa'.format(self.pO2_calc.mean(), self.pO2_calc[0],
                                                                  self.pO2_calc[2]))
            if np.isnan(self.pH_calc).any() == True:
                self.message_tab3.append('Calculated pH = {:.2f} (median) - interval {:.2f} -'
                                         ' {:.2f}'.format(self.pH_calc_corr[1], self.pH_calc_corr[0],
                                                          self.pH_calc_corr[2]))
            else:
                self.message_tab3.append('Calculated pH = {:.2f} (mean) - interval {:.2f} -'
                                         ' {:.2f}'.format(self.pH_calc.mean(), self.pH_calc[0], self.pH_calc[2]))

            # simulation completed
            self.progress_tab3.setValue(100)

# -------------------------------------------------------------------------------------------------------------
# if simulation is False -> measurement evaluation
# -------------------------------------------------------------------------------------------------------------
        else:
            self.message_tab3.setText('Measurement evaluation')

            # clear pH/pO2 from input table and I_ratio
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[0, 1, 2, 3])

            if self.int_ratio_checkbox.isChecked():
                cols_required = [4, 5, 6]
            else:
                cols_required = [4, 5, 6, 7]
            # superimposed phase angles and intensity ratio required
            for i in range(self.tableCalibration.rowCount()):
                for j in cols_required:
                    try:
                        self.tableCalibration.item(i, j).text()
                    except:
                        calibration_para_failed = QMessageBox()
                        calibration_para_failed.setIcon(QMessageBox.Information)
                        calibration_para_failed.setText("Insufficient calibration parameters!")
                        calibration_para_failed.setInformativeText("For measurement evaluation the superimposed phase "
                                                                   "angles and either the intensity ratios or the total "
                                                                   "amplitudes at two modulation frequencies are "
                                                                   "required.")
                        calibration_para_failed.setWindowTitle("Error!")
                        calibration_para_failed.exec_()
                        return

                if self.tableCalibration.item(i, j).text() == '':
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("For measurement evaluation the superimposed phase "
                                                               "angles and either the intensity ratios or the total "
                                                               "amplitudes at two modulation frequencies are "
                                                               "required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

            # measurement point
            for j in cols_required:
                try:
                    self.tableINPUT.item(0, j).text()
                except:
                    measurement_failed = QMessageBox()
                    measurement_failed.setIcon(QMessageBox.Information)
                    measurement_failed.setText("Insufficient measurement parameters!")
                    measurement_failed.setInformativeText("The superimposed phase angles and the intensity ratio are"
                                                          " required for the measurement evaluation.")
                    measurement_failed.setWindowTitle("Error!")
                    measurement_failed.exec_()
                    return

            self.progress_tab3.setValue(15)

            # -------------------------------------------------------------------------------------
            # all phase angles including error
            self.phi_meas_f1 = []
            self.phi_meas_f2 = []
            self.Phi_meas_f1_er = {}
            self.Phi_meas_f2_er = {}
            self.ampl_total_f1_meas = []
            self.ampl_total_f2_meas = []
            self.int_ratio_meas = []

            # check the number of measurement points
            list_meas = []
            for i in range(self.tableINPUT.rowCount()):
                it = self.tableINPUT.item(i, 4)
                if it and it.text():
                    list_meas.append(it.text())

            for i in range(len(list_meas)):
                # phase angles for both modulation frequencies including assumed error
                dphi1 = np.float64(self.tableINPUT.item(i, 4).text().replace(',', '.'))
                dphi2 = np.float64(self.tableINPUT.item(i, 5).text().replace(',', '.'))
                self.phi_meas_f1.append(dphi1)
                self.phi_meas_f2.append(dphi2)
                self.Phi_meas_f1_er['meas {}'.format(i)] = [dphi1 - self.error_assumed, dphi1,
                                                            dphi1 + self.error_assumed]
                self.Phi_meas_f2_er['meas {}'.format(i)] = [dphi2 - self.error_assumed, dphi2,
                                                            dphi2 + self.error_assumed]

            # intensity ratio or amplitudes
            for i in range(len(list_meas)):
                if self.int_ratio_checkbox.isChecked() is True:
                    self.int_ratio_meas.append(np.float64(self.tableINPUT.item(i, 6).text().replace(',', '.')))
                else:
                    self.ampl_total_f1_meas.append(np.float64(self.tableINPUT.item(i, 6).text().replace(',', '.')))
                    self.ampl_total_f2_meas.append(np.float64(self.tableINPUT.item(i, 7).text().replace(',', '.')))

            # extract measured phase angles and intensity ratio with calibration points
            f_ind = ['']*len(calib_points)
            ampl_ind = ['']*len(calib_points)
            for i in range(len(calib_points)):
                if calib_points[i][0] == self.pH_list[0]:
                    f_ind[i] = f_ind[i] + 'fluoro0, '
                    ampl_ind[i] = ampl_ind[i] + 'fluoro0, '
                else:
                    f_ind[i] = f_ind[i] + 'fluoro1, '
                    ampl_ind[i] = ampl_ind[i] + 'fluoro1, '
                if calib_points[i][1] == self.pO2_list[0]:
                    f_ind[i] = f_ind[i] + 'phosphor0'
                    ampl_ind[i] = ampl_ind[i] + 'phosphor0'
                else:
                    f_ind[i] = f_ind[i] + 'phosphor1'
                    ampl_ind[i] = ampl_ind[i] + 'phosphor1'

            list_phi_f1 = []
            list_phi_f2 = []
            list_int_ratio = []
            list_ampl1 = []
            list_ampl2 = []

            for i in range(self.tableCalibration.rowCount()):
                list_phi_f1.append(np.float64(self.tableCalibration.item(i, 4).text().replace(',', '.')))
                list_phi_f2.append(np.float64(self.tableCalibration.item(i, 5).text().replace(',', '.')))
                if self.int_ratio_checkbox.isChecked() is True:
                    list_int_ratio.append(np.float64(self.tableCalibration.item(i, 6).text().replace(',', '.')))
                else:
                    list_ampl1.append(np.float64(self.tableCalibration.item(i, 6).text().replace(',', '.')))
                    list_ampl2.append(np.float64(self.tableCalibration.item(i, 7).text().replace(',', '.')))

            ls_meas_index = []
            for num in range(len(list_meas)):
                ls_meas_index.append('meas {}'.format(num))
            ampl_ind = ampl_ind + ls_meas_index
            if self.int_ratio_checkbox.isChecked() is True:
                list_int_ratio = list_int_ratio + self.int_ratio_meas
                # list_int_ratio.append(self.int_ratio_meas)
                self.int_ratio = pd.Series(list_int_ratio, index=ampl_ind)
                self.ampl_total_f1 = None
                self.ampl_total_f2 = None
            else:
                list_ampl1 = list_ampl1 + self.ampl_total_f1_meas
                list_ampl2 = list_ampl2 + self.ampl_total_f2_meas
                self.ampl_total_f1 = pd.Series(list_ampl1, index=ampl_ind)
                self.ampl_total_f2 = pd.Series(list_ampl2, index=ampl_ind)
                self.int_ratio = None

            dPhi_f1 = pd.Series(list_phi_f1, index=f_ind)
            dPhi_f2 = pd.Series(list_phi_f2, index=f_ind)

            # ----------------------------------------------
            # Include error in measured phase angle
            keys = self.Phi_f1_deg.keys().tolist()
            dPhi_f1_er = pd.Series({key: None for key in keys})
            dPhi_f2_er = pd.Series({key: None for key in keys})
            for i in self.Phi_f1_deg.keys():
                dPhi_f1_er[i] = ([self.Phi_f1_deg[i] - self.error_assumed, self.Phi_f1_deg[i], self.Phi_f1_deg[i] +
                                  self.error_assumed])
                dPhi_f2_er[i] = ([self.Phi_f2_deg[i] - self.error_assumed, self.Phi_f2_deg[i], self.Phi_f2_deg[i] +
                                  self.error_assumed])

            # all phase angles including error
            self.dPhi_meas_f1_er = pd.concat([dPhi_f1_er, pd.Series(self.Phi_meas_f1_er)], axis=0)
            self.dPhi_meas_f2_er = pd.concat([dPhi_f2_er, pd.Series(self.Phi_meas_f2_er)], axis=0)

            self.progress_tab3.setValue(30)

        # ---------------------------------------------------------------------------------------------------------
            # Dualsensing pH and pO2
            [self.pO2_calc, self.tau_quot, self.tauP, self.intP, self.pH_calc, self.intF_meas, self.para_TSM, ax_pO2,
            ax_pH] = pHox.pH_oxygen_dualsensor_meas(phi_f1_deg=dPhi_f1, phi_f2_deg=dPhi_f2, method_='std',
                                                     phi_f1_meas=self.phi_meas_f1,
                                                     phi_f2_meas=self.phi_meas_f2,
                                                     error_phaseangle=self.error_assumed, pO2_range=pO2_range,
                                                     pO2_calib=self.pO2_list[:2], int_ratio=self.int_ratio,
                                                     curv_O2=self.curv_O2, ampl_total_f1=self.ampl_total_f1,
                                                     prop_ksv=self.prop_ksv, ampl_total_f2=self.ampl_total_f2,
                                                     df_conv_tau_int=self.conv_tau_int, plotting=False, f2=self.f2,
                                                     intP_c0=self.int_phosphor_max, pH_range=pH_range, f1=self.f1,
                                                     pH_calib=self.pH_list[:2], v50=pKa, slope_pH=slope_pH)

            self.progress_tab3.setValue(65)

        # ----------------------------------------------------------------------
            # Plotting calibration - according to first measurement point
            self.plot_calibration_pH_pO2_measurement(pH_calib=self.pH_list, pO2_calib=self.pO2_list, pk_a=pKa,
                                                     K_sv=self.para_TSM[0]['Ksv_fit1'], f1=self.fig_pH_calib,
                                                     int_f_c0=self.intF_meas[0], ax1=self.ax_pH_calib,
                                                     tau_quot=self.tau_quot[0], f2=self.fig_pO2_calib,
                                                     ax2=self.ax_pO2_calib)
            self.progress_tab3.setValue(75)

        # ---------------------------------------
            # Plotting results
            tauP_mikros = {}
            for i in range(len(self.tauP)):
                if self.tauP[i].loc[0, 1] < 1.:
                    tauP_mikros[i] = self.tauP[i] * 1E6
                else:
                    tauP_mikros[i] = self.tauP[i]

            # first and last measurement point
            self.plot_results_pH_pO2_meas(pH=self.pH_calc, pH_calib=self.pH_list, pO2_calc=self.pO2_calc,
                                          tauP=tauP_mikros, intF=self.intF_meas, pO2_calib=self.pO2_list,
                                          f1=self.fig_pH, ax1=self.ax_pH, f2=self.fig_pO2, ax2=self.ax_pO2)
            self.progress_tab3.setValue(85)

        # --------------------------------------------------
            # Results reported in message box
            for i in range(len(list_meas)):
                self.message_tab3.append('# Point - {}'.format(i + 1))
                self.message_tab3.append('\t Calculated pO2 = {:.2f} hPa (median) - interval '
                                         '{:.2f} - {:.2f} hPa'.format(np.array(self.pO2_calc[i]).mean(),
                                                                      self.pO2_calc[i][0], self.pO2_calc[i][2]))
                self.message_tab3.append('\t Calculated pH = {:.2f} (median) - interval '
                                         '{:.2f} - {:.2f}'.format(np.array(self.pH_calc[i]).mean(), self.pH_calc[i][0],
                                                                  self.pH_calc[i][2]))

            self.progress_tab3.setValue(100)

        print('pH / O2 dualsensing finished')
        self.run_tab3_button.setStyleSheet(
            "color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #227286, stop: 1 "
            "#54bad4); border-width: 1px; border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
            "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")
        print('#-------------------------------------------------------------------')


# ---------------------------------------------------------------------------------------------------------------------
# pH / T sensing
# ---------------------------------------------------------------------------------------------------------------------
    def open_compensation(self):
        self.load_temp_comp_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px;"
                                                 " border-color: #077487; border-style: solid; border-radius: 7; "
                                                 " padding: 5px; font-size: 10px; padding-left: 1px; padding-right:"
                                                 " 5px; min-height: 10px; max-height: 18px;")

        path_gui_ = os.path.abspath("GUI_dualsensors.py")
        path_gui = path_gui_.split('GUI_dualsensors')[0]

        self.fname_comp = QFileDialog.getOpenFileName(self, "Select specific txt file for temperature compensation",
                                                      path_gui)[0]

        self.load_temp_comp_button.setStyleSheet("color: white; background-color:"
                                                 "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                 "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                 "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px;"
                                                 "padding-left: 1px; padding-right: 5px; min-height: 10px;"
                                                 "max-height: 18px;")
        if not self.fname_comp:
            return
        # next function -> load file
        self.read_compensation(self.fname_comp)

    def read_compensation(self, fname_comp):
        self.compensation_edit.clear()
        try:
            self.comp_file = fname_comp
        except:
            comp_file_load_failed = QMessageBox()
            comp_file_load_failed.setIcon(QMessageBox.Information)
            comp_file_load_failed.setText('Invalid file for temperature compensation!')
            comp_file_load_failed.setInformativeText('Choose another file from path...')
            comp_file_load_failed.setWindowTitle('Error!')
            comp_file_load_failed.buttonClicked.connect(self.open_compensation)
            comp_file_load_failed.exec_()
            return
        # ---------------------------------------------------------------------
        # write (part of the path) to text line
        parts = fname_comp.split('/')
        if 3 < len(parts) <= 6:
            self.comp_file_part = '/'.join(parts[3:])
        elif len(parts) > 6:
            self.comp_file_part = '/'.join(parts[5:])
        else:
            self.comp_file_part = '/'.join(parts)
        self.compensation_edit.insertPlainText(str(self.comp_file_part))

        # ---------------------------------------------------------------------
        # load txt file
        self.df_temp_comp = pd.read_csv(self.comp_file, sep='\t', decimal='.', index_col=0, encoding='latin-1')

    def simulation_to_evaluation(self):
        if self.simulation_checkbox.isChecked() is True:
            self.measurement_checkbox.setCheckState(False)
        elif self.simulation_checkbox.isChecked() is False:
            self.measurement_checkbox.setCheckState(True)

    def evaluation_to_simulation(self):
        if self.measurement_checkbox.isChecked() is True:
            self.simulation_checkbox.setCheckState(False)
            if np.int(self.tabs.currentIndex()+1) == 6:
                self.total_amplitude_checkbox.setCheckState(True)
                self.intensity_to_amplitude()
        elif self.measurement_checkbox.isChecked() is False:
            self.simulation_checkbox.setCheckState(True)

    def calib_ksv_to_tauP(self):
        if self.calib_ksv_checkbox.isChecked() is True:
            self.calib_tauP_checkbox.setCheckState(False)
        elif self.calib_ksv_checkbox.isChecked() is False:
            self.calib_tauP_checkbox.setCheckState(True)

    def calib_tauP_to_ksv(self):
        if self.calib_tauP_checkbox.isChecked() is True:
            self.calib_ksv_checkbox.setCheckState(False)
        if self.calib_tauP_checkbox.isChecked() is False:
            self.calib_ksv_checkbox.setCheckState(True)

    def amplitude_to_intensity(self):
        if self.int_ratio_checkbox.isChecked() is True:
            self.total_amplitude_checkbox.setCheckState(False)

            columnPosition = self.tableCalibration.columnCount()
            if columnPosition == 7:
                pass
            else:
                self.tableCalibration.removeColumn(columnPosition-1)
                self.tableINPUT.removeColumn(columnPosition-1)

            self.tableINPUT.setItem(0, 6, QTableWidgetItem(''))
            for i in range(self.tableCalibration.rowCount()):
                self.tableCalibration.setItem(i, 6, QTableWidgetItem(''))

            self.tableCalibration.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                             "dPhi(f2) [°];  I-ratio").split(";"))
            self.tableINPUT.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                       "dPhi(f2) [°];  I-ratio").split(";"))

        elif self.int_ratio_checkbox.isChecked() is False:
            self.total_amplitude_checkbox.setCheckState(True)

            columnPosition = self.tableCalibration.columnCount()
            if columnPosition < 8:
                self.tableCalibration.insertColumn(columnPosition)
                self.tableINPUT.insertColumn(columnPosition)
            else:
                pass

            self.tableINPUT.setItem(0, 6, QTableWidgetItem(''))
            self.tableINPUT.setItem(0, 7, QTableWidgetItem(''))
            for i in range(self.tableCalibration.rowCount()):
                self.tableCalibration.setItem(i, 6, QTableWidgetItem(''))
                self.tableCalibration.setItem(i, 7, QTableWidgetItem(''))

            self.tableCalibration.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                             "dPhi(f2) [°]; A(f1) [mV]; A(f2) [mV]").split(";"))
            self.tableINPUT.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                       "dPhi(f2) [°]; A(f1) [mV]; A(f2) [mV]").split(";"))
        self.tableCalibration.resizeColumnToContents(0)
        self.tableINPUT.resizeColumnToContents(0)

    def intensity_to_amplitude(self):
        if self.total_amplitude_checkbox.isChecked() is True:
            self.int_ratio_checkbox.setCheckState(False)

            columnPosition = self.tableCalibration.columnCount()
            if columnPosition < 8:
                self.tableCalibration.insertColumn(columnPosition)
                self.tableINPUT.insertColumn(columnPosition)
            else:
                pass

            self.tableINPUT.setItem(0, 6, QTableWidgetItem(''))
            self.tableINPUT.setItem(0, 7, QTableWidgetItem(''))
            for i in range(self.tableCalibration.rowCount()):
                self.tableCalibration.setItem(i, 6, QTableWidgetItem(''))
                self.tableCalibration.setItem(i, 7, QTableWidgetItem(''))

            self.tableCalibration.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                             " dPhi(f2) [°]; A(f1) [mV]; A(f2) [mV]").split(";"))
            self.tableINPUT.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; dPhi(f1) [°]; "
                                                       " dPhi(f2) [°]; A(f1) [mV]; A(f2) [mV]").split(";"))
        elif self.total_amplitude_checkbox.isChecked() is False:
            self.int_ratio_checkbox.setCheckState(True)
            columnPosition = self.tableCalibration.columnCount()
            if columnPosition == 7:
                pass
            else:
                self.tableCalibration.removeColumn(columnPosition-1)
                self.tableINPUT.removeColumn(columnPosition-1)

            self.tableINPUT.setItem(0, 6, QTableWidgetItem(''))
            for i in range(self.tableCalibration.rowCount()):
                self.tableCalibration.setItem(i, 6, QTableWidgetItem(''))

            self.tableCalibration.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; "
                                                             "dPhi(f1) [°]; dPhi(f2) [°]; I-ratio").split(";"))
            self.tableINPUT.setHorizontalHeaderLabels(("pH; Temp [°C]; pCO2 [hPa]; pO2 [hPa]; I-ratio; dPhi(f1) [°]; "
                                                       "dPhi(f2) [°]").split(";"))

        self.tableCalibration.resizeColumnToContents(0)
        self.tableINPUT.resizeColumnToContents(0)

    def simulation_methods_optimistic(self):
        if self.optimistic_checkbox.isChecked() is True:
            self.moderate_checkbox.setCheckState(False)
            self.pessimistic_checkbox.setCheckState(False)

    def simulation_methods_moderate(self):
        if self.moderate_checkbox.isChecked() is True:
            self.optimistic_checkbox.setCheckState(False)
            self.pessimistic_checkbox.setCheckState(False)

    def simulation_methods_pessimistic(self):
        if self.pessimistic_checkbox.isChecked() is True:
            self.moderate_checkbox.setCheckState(False)
            self.optimistic_checkbox.setCheckState(False)

    def input_collecting(self):
        f1 = np.float64(self.frequency1fix_edit.text().replace(',', '.'))
        f2 = np.float64(self.frequency2fix_edit.text().replace(',', '.'))
        error = np.float64(self.error_assumed_meas_edit.text().replace(',', '.'))
        int_fluoro_max = 100. # %
        int_ratio_dualsens = np.float64(self.int_ratio_dualsens_edit.text().replace(',', '.'))
        int_phosphor_max = int_fluoro_max / int_ratio_dualsens

        return f1, f2, error, int_fluoro_max, int_phosphor_max

    def temperature_scanning(self):
        slope_min = (self.res_T['tauP'][0][1] - self.res_T['tauP'][0][0]) / (self.temp_list[1] - self.temp_list[0])
        slope_av = (self.res_T['tauP'][1][1] - self.res_T['tauP'][1][0]) / (self.temp_list[1] - self.temp_list[0])
        slope_max = (self.res_T['tauP'][2][1] - self.res_T['tauP'][2][0]) / (self.temp_list[1] - self.temp_list[0])

        intercept_min = self.res_T['tauP'][0][1] - slope_min*(self.temp_list[1] + conv_temp)
        intercept_av = self.res_T['tauP'][1][1] - slope_av*(self.temp_list[1] + conv_temp)
        intercept_max = self.res_T['tauP'][2][1] - slope_max*(self.temp_list[1] + conv_temp)

        y_min = []
        y_av = []
        y_max = []
        for i in self.temp_range_K:
            y_min.append(i*slope_min + intercept_min)
            y_av.append(i*slope_av + intercept_av)
            y_max.append(i*slope_max + intercept_max)
        temp_scan = pd.Series({'min': y_min, 'mean': y_av, 'max': y_max})

        return temp_scan

    def temp_compensation(self, file, T_meas, temp_range, fit_k=False, fit_pka=True, fit_bottom=True,
                          fit_top=True):

        # sensor feature
        ddf = pd.read_csv(file, encoding='latin-1', sep='\t')#, usecols=[1, 2, 3, 4, 5])

        x = ddf['T [°C]'].values
        slope_k, intercept_k, r_value_k, p_value_k, std_err_k = stats.linregress(x=x, y=ddf['slope'])
        slope_pka, intercept_pka, r_value_pka, p_value_pka, std_err_pka = stats.linregress(x=x, y=ddf['V50'])
        slope_top, intercept_top, r_value_top, p_value_top, std_err_top = stats.linregress(x=x, y=ddf['Top'])
        [slope_bottom, intercept_bottom, r_value_bottom, p_value_bottom,
         std_err_bottom] = stats.linregress(x=x, y=ddf['Bottom'])

        parameter_fit = pd.Series({'slope, slope': slope_k, 'top, slope': slope_top, 'bottom, slope': slope_bottom,
                                   'pka, slope': slope_pka, 'slope, intercept': intercept_k,
                                   'top, intercept': intercept_top, 'bottom, intercept': intercept_bottom,
                                   'pka, intercept': intercept_pka})

        # determination of sensor characteristics at measurement temperature
        if np.isnan(slope_k) is True or fit_k is False:
            slope_k = 0.
            fit_k = False
            k = ddf['slope'].mean()
        else:
            k = slope_k*T_meas + intercept_k
        if np.isnan(slope_pka) is True or fit_pka is False:
            slope_pka = 0.
            fit_pka = False
            pka = ddf['V50'].mean()
        else:
            pka = slope_pka*T_meas + intercept_pka
        if np.isnan(slope_top) is True or fit_top is False:
            slope_top = 0.
            fit_top = False
            top = ddf['Top'].mean()
        else:
            top = slope_top*T_meas + intercept_top
        if np.isnan(slope_bottom) is True or fit_bottom is False:
            slope_bottom = 0.
            fit_bottom = False
            bottom = ddf['Bottom'].mean()
        else:
            bottom = slope_bottom*T_meas + intercept_bottom
        para_meas = pd.Series({'slope': k, 'pka': pka, 'top': top, 'bottom': bottom})

        y_k = slope_k*temp_range + intercept_k
        y_pka = slope_pka*temp_range + intercept_pka
        y_top = slope_top*temp_range + intercept_top
        y_bottom = slope_bottom*temp_range + intercept_bottom

        fit_parameter = pd.Series({'slope': y_k, 'pka': y_pka, 'top': y_top, 'bottom': y_bottom})

        # ----------------------------------------------------
        # Fitting along frequency

        return ddf, fit_parameter, para_meas

    def plot_temp_compensation(self, fig, ax, ddf, temp_range, T_meas, para_meas, fit_parameter, fit_k=True,
                               fit_pka=True, fit_bottom=True, fit_top=True):
        if fig is None:
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
        ax[0][0].plot(ddf['T [°C]'], ddf['slope'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['slope'], T_meas))

        if fit_k is True:
            ax[0][0].plot(temp_range, fit_parameter['slope'], lw=0.75, ls='-', color='k')
            ax[0][0].axhline(para_meas['slope'], color='crimson', lw=0.5, ls='--')
            ax[0][0].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[1][0].plot(ddf['T [°C]'], ddf['V50'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['pka'], T_meas))

        if fit_pka is True:
            ax[1][0].plot(temp_range, fit_parameter['pka'], lw=0.75, ls='-', color='k')
            ax[1][0].axhline(para_meas['pka'], color='crimson', lw=0.5, ls='--')
            ax[1][0].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[0][1].plot(ddf['T [°C]'], ddf['Bottom'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['bottom'], T_meas))

        if fit_bottom is True:
            ax[0][1].plot(temp_range, fit_parameter['bottom'], lw=0.75, ls='-', color='k')
            ax[0][1].axhline(para_meas['bottom'], color='crimson', lw=0.5, ls='--')
            ax[0][1].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[1][1].plot(ddf['T [°C]'], ddf['Top'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['top'], T_meas))

        if fit_top is True:
            ax[1][1].plot(temp_range, fit_parameter['top'], lw=0.75, ls='-', color='k')
            ax[1][1].axhline(para_meas['top'], color='crimson', lw=0.5, ls='--')
            ax[1][1].axvline(T_meas, color='crimson', lw=0.5, ls='--')

        ax[0][0].set_ylabel('slope', fontsize=9)
        ax[1][0].set_ylabel('pka', fontsize=9)
        ax[0][1].set_ylabel('bottom', fontsize=9)
        ax[1][1].set_ylabel('top', fontsize=9)
        ax[1][0].set_xlabel('Temperature T [°C]', fontsize=9)
        ax[1][1].set_xlabel('Temperature T [°C]', fontsize=9)
        ax[0][0].tick_params(axis='both', which='both', labelsize=7)
        ax[1][0].tick_params(axis='both', which='both', labelsize=7)
        ax[0][1].tick_params(axis='both', which='both', labelsize=7)
        ax[1][1].tick_params(axis='both', which='both', labelsize=7)

        ax[0][0].legend(fontsize=9)
        ax[1][0].legend(fontsize=9)
        ax[0][1].legend(fontsize=9)
        ax[1][1].legend(fontsize=9)

        ax[0][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[1][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[0][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[1][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)

        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98, wspace=.3, hspace=.2)
        fig.canvas.draw()

    def plot_temp_compensation_meas(self, fig, ax, ddf, T_meas, temp_range, fit_parameter, para_meas, fit_k=True,
                                    fit_pka=True, fit_bottom=True, fit_top=True):
        if fig is None:
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
        ax[0][0].plot(ddf['T [°C]'], ddf['slope'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['slope'], T_meas))

        if fit_k is True:
            ax[0][0].plot(temp_range, fit_parameter['slope'], lw=0.75, ls='-', color='k')

        ax[1][0].plot(ddf['T [°C]'], ddf['V50'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['pka'], T_meas))

        if fit_pka is True:
            ax[1][0].plot(temp_range, fit_parameter['pka'], lw=0.75, ls='-', color='k')

        ax[0][1].plot(ddf['T [°C]'], ddf['Bottom'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['bottom'], T_meas))

        if fit_bottom is True:
            ax[0][1].plot(temp_range, fit_parameter['bottom'], lw=0.75, ls='-', color='k')

        ax[1][1].plot(ddf['T [°C]'], ddf['Top'], marker='o', markersize=3, lw=0,
                      label='{:.3f} @ {:.2f}°C'.format(para_meas['top'], T_meas))

        if fit_top is True:
            ax[1][1].plot(temp_range, fit_parameter['top'], lw=0.75, ls='-', color='k')

        ax[0][0].set_ylabel('slope', fontsize=9)
        ax[1][0].set_ylabel('pka', fontsize=9)
        ax[0][1].set_ylabel('bottom', fontsize=9)
        ax[1][1].set_ylabel('top', fontsize=9)
        ax[1][0].set_xlabel('Temperature T [°C]', fontsize=9)
        ax[1][1].set_xlabel('Temperature T [°C]', fontsize=9)
        ax[0][0].tick_params(axis='both', which='both', labelsize=7)
        ax[1][0].tick_params(axis='both', which='both', labelsize=7)
        ax[0][1].tick_params(axis='both', which='both', labelsize=7)
        ax[1][1].tick_params(axis='both', which='both', labelsize=7)

        ax[0][0].legend(fontsize=9)
        ax[1][0].legend(fontsize=9)
        ax[0][1].legend(fontsize=9)
        ax[1][1].legend(fontsize=9)

        ax[0][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[1][0].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[0][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax[1][1].tick_params(axis='both', which='both', direction='in', top=True, right=True)

        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98, wspace=.3, hspace=.2)
        fig.canvas.draw()

    def plot_temperature(self, f, ax, temp_scan, res_T, temp_deg, temp_calib):
        ax.cla()
        # preparation of figure plot
        if ax is None:
            f = plt.figure()
            ax = f.gca()

        # mean value of phosphor lifetime
        ax.plot(temp_scan-conv_temp, res_T['mean'], color='navy', lw=0.75)
        ax.legend(['T = {:.2f} ± {:.2f} °C'.format(temp_deg[1], (temp_deg[2]-temp_deg[0])/2)], frameon=True,
                  framealpha=0.5, fancybox=True, loc=0, fontsize=9)

        # min - max range when measurement uncertainty is included
        ax.plot(temp_scan-conv_temp, res_T['min'], color='k', ls='--', lw=0.5)
        ax.plot(temp_scan-conv_temp, res_T['max'], color='k', ls='--', lw=0.5)
        ax.fill_between(temp_scan-conv_temp, res_T['min'], res_T['max'], color='grey', alpha=0.1)

        # region of measurement point - Temperature (x axis)
        ax.axvline(temp_deg[0], color='k', ls='--', lw=.5)
        ax.axvline(temp_deg[2], color='k', ls='--', lw=.5)
        ax.axvspan(temp_deg[0], temp_deg[2], alpha=0.2, color='#f0810f')

        # find closest y value (lifetime phosphor)
        res_T_min = fp.find_closest_value_(index=temp_scan-conv_temp, data=res_T['mean'], value=temp_deg[0])
        res_T_max = fp.find_closest_value_(index=temp_scan-conv_temp, data=res_T['mean'], value=temp_deg[2])

        # linear regression to tauP measured
        arg_min = stats.linregress(x=temp_calib, y=res_T_min[2:])
        arg_max = stats.linregress(x=temp_calib, y=res_T_max[2:])
        y_min = arg_min[0] * temp_deg[0] + arg_min[1]
        y_max = arg_max[0] * temp_deg[2] + arg_max[1]
        ax.axhline(y_min, color='k', ls='--', lw=.4)
        ax.axhline(y_max, color='k', ls='--', lw=.4)
        ax.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        ax.set_xlabel('Temperature [°C]', fontsize=9)
        ax.set_ylabel('$τ_P$ [µs]', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        f.canvas.draw()

    def plot_temperature_meas(self, f, ax, temp_scan, y_reg_ph1, tau_phos, temp_calc):
        ax.cla()
        # preparation of figure plot
        if ax is None:
            f = plt.figure()
            ax = f.gca()

        # T evaluation - linear egression tauP vs. T
        ax.plot(temp_scan-conv_temp, y_reg_ph1.mean(axis=1), color='navy', lw=1.)
        ax.plot(temp_scan-conv_temp, y_reg_ph1['min'], color='k', ls='--', lw=0.5)
        ax.plot(temp_scan-conv_temp, y_reg_ph1['max'], color='k', ls='--', lw=0.5)
        ax.fill_between(temp_scan-conv_temp, y_reg_ph1['min'], y_reg_ph1['max'], color='grey', alpha=0.2)

        if len(tau_phos) > 1:
            # last measurement point
            last = len(tau_phos) - 1
            ax.legend(['T$_{}$ = {:.2f} ± {:.2f} °C'.format('1', temp_calc[0][1],
                                                            (temp_calc[0][2] - temp_calc[0][0]) / 2),
                       'T$_{}$ = {:.2f} ± {:.2f} °C'.format(len(tau_phos), temp_calc[last][1],
                                                            (temp_calc[last][2] - temp_calc[last][0]) / 2)], loc=0,
                      frameon=True, framealpha=0.5, fancybox=True, fontsize=9)

        else:
            ax.legend(['T$_{}$ = {:.2f} ± {:.2f} °C'.format('1', temp_calc[0][1],
                                                            (temp_calc[0][2] - temp_calc[0][0]) / 2)], loc=0,
                      frameon=True, framealpha=0.5, fancybox=True, fontsize=9)

        ax.axvline(temp_calc[0][0], color='k', lw=0.5, ls='--')
        ax.axvline(temp_calc[0][2], color='k', lw=0.5, ls='--')
        ax.axvspan(temp_calc[0][0], temp_calc[0][2], color='#f0810f', alpha=0.4)

        ax.axhline(tau_phos[0][0], color='k', ls='--', lw=.5)
        ax.axhline(tau_phos[0][2], color='k', ls='--', lw=.5)
        ax.axhspan(tau_phos[0][0], tau_phos[0][2], color='#f0810f', alpha=0.4)

        if len(tau_phos) > 1:
            # last measurement point
            last = len(tau_phos) - 1
            ax.axvline(temp_calc[last][0], color='k', lw=0.5, ls='-.')
            ax.axvline(temp_calc[last][2], color='k', lw=0.5, ls='-.')
            ax.axvspan(temp_calc[last][0], temp_calc[last][2], color='#f0810f', alpha=0.2)

            ax.axhline(tau_phos[last][0], color='k', ls='-.', lw=.5)
            ax.axhline(tau_phos[last][2], color='k', ls='-.', lw=.5)
            ax.axhspan(tau_phos[last][0], tau_phos[last][2], color='#f0810f', alpha=0.2)

        # layout
        ax.set_xlabel('Temperature [°C]', fontsize=9)
        ax.set_ylabel('$τ_P$ [µs]', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        f.canvas.draw()

    def plot_pH_calib(self, f, ax, ax2, ph_range, pk_a, cot_dPhi, para_f, pH_list):
        ax.cla()
        ax2.cla()

        if ax is None:
            f, ax = plt.subplots(figsize=(20, 10))
        if ax2 is None:
            ax2 = ax.twinx()

        # plotting cot_dPhi and dPhi
        mean_cotPhi = np.mean(np.array([cot_dPhi['phosphor0'], cot_dPhi['phosphor1']]), axis=0)
        ax.plot(ph_range, mean_cotPhi, color='navy', lw=0.75, label='pK$_a$ = {}')
        ax.axvline(pH_list[0], color='k', lw=0.5, ls='--')
        ax.axvline(pH_list[1], color='k', lw=0.5, ls='--')

        ax.legend(['pK$_a$ = {:.2f}'.format(pk_a), 'pH$_1$ = {:.1f}'.format(pH_list[0]),
                   'pH$_2$ = {:.1f}'.format(pH_list[1])], frameon=True, framealpha=0.5, fancybox=True,
                  loc='upper center', bbox_to_anchor=(0.85, 0.8), fontsize=9)

        ax.plot(ph_range, cot_dPhi['phosphor0'], color='k', ls='--', lw=0.5)
        ax.plot(ph_range, cot_dPhi['phosphor1'], color='k', ls='--', lw=0.5)

        mean_Phi = np.mean(np.array([np.rad2deg(fp.arccot(cot_dPhi['phosphor0'])),
                                     np.rad2deg(fp.arccot(cot_dPhi['phosphor1']))]), axis=0)
        ax2.plot(ph_range, mean_Phi, color='forestgreen', lw=0.5)
        ax2.plot(ph_range, np.rad2deg(fp.arccot(cot_dPhi['phosphor0'])), color='k', ls='--', lw=0.5)
        ax2.plot(ph_range, np.rad2deg(fp.arccot(cot_dPhi['phosphor1'])), color='k', ls='--', lw=0.5)

        ax.fill_between(ph_range, cot_dPhi['phosphor0'], cot_dPhi['phosphor1'], color='grey', alpha=0.2)
        ax2.fill_between(ph_range, np.rad2deg(fp.arccot(cot_dPhi['phosphor0'])),
                         np.rad2deg(fp.arccot(cot_dPhi['phosphor1'])),
                         color='grey', alpha=0.2)

        ax.plot(pH_list[0], cot_dPhi['fluoro0, phosphor0'], marker='D', color='orange')
        ax.plot(pH_list[1], cot_dPhi['fluoro1, phosphor0'], marker='D', color='orange')
        ax.plot(pH_list[0], cot_dPhi['fluoro0, phosphor1'], marker='o', alpha=0.5, color='orange')
        ax.plot(pH_list[1], cot_dPhi['fluoro1, phosphor1'], marker='o', alpha=0.5, color='orange')

        ax.axhline(para_f['phosphor0']['bottom'], color='k', lw=0.5, ls='--')
        ax.axhline(para_f['phosphor1']['bottom'], color='k', lw=0.5, ls='--')

        ax.set_xlim(0, 15)
        ax.set_ylabel('cot(Φ)', color='navy', fontsize=9)
        ax2.set_ylabel('Φ [deg]', color='forestgreen', fontsize=9)
        ax.set_xlabel('pH', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        f.canvas.draw()

    def plot_pH_calib_meas(self, f, ax, ax2, ph_range, pk_a, cot_dPhi, para_f, pH_list):
        ax.cla()
        ax2.cla()

        if ax is None:
            f, ax = plt.subplots(figsize=(20, 10))
        if ax2 is None:
            ax2 = ax.twinx()

        # boltzmann sigmoid
        cot_dPhi_T0 = pHtemp.boltzmann_sigmoid(top=para_f['phosphor0']['top'], bottom=para_f['phosphor0']['bottom'],
                                               slope=para_f['phosphor0']['slope'],
                                               pka=para_f['phosphor0']['pka'], pH=ph_range)
        cot_dPhi_T1 = pHtemp.boltzmann_sigmoid(top=para_f['phosphor1']['top'], bottom=para_f['phosphor1']['bottom'],
                                               slope=para_f['phosphor1']['slope'],
                                               pka=para_f['phosphor1']['pka'], pH=ph_range)

        # plotting cot_dPhi and dPhi
        mean_cotPhi = np.mean(np.array([cot_dPhi_T0, cot_dPhi_T1]), axis=0)
        ax.plot(ph_range, mean_cotPhi, color='navy', lw=0.75, label='pK$_a$ = {}')
        ax.axvline(pH_list[0], color='k', ls='--', lw=.5)
        ax.axvline(pH_list[1], color='k', ls='--', lw=.5)

        ax.legend(['pK$_a$ = {:.2f}'.format(pk_a), 'pH$_1$ = {:.1f}'.format(pH_list[0]),
                   'pH$_2$ = {:.1f}'.format(pH_list[1])], frameon=True, framealpha=0.5, fancybox=True,
                  loc='upper center', bbox_to_anchor=(0.85, 0.8), fontsize=9)

        ax.plot(ph_range, cot_dPhi_T0, color='k', lw=0.5, ls='--')
        ax.plot(ph_range, cot_dPhi_T1, color='k', lw=0.5, ls='--')

        mean_Phi = np.mean(np.array([np.rad2deg(fp.arccot(cot_dPhi_T0)), np.rad2deg(fp.arccot(cot_dPhi_T1))]), axis=0)
        ax2.plot(ph_range, mean_Phi, color='forestgreen', lw=0.75)
        ax2.plot(ph_range, np.rad2deg(fp.arccot(cot_dPhi_T0)), color='k', lw=0.5, ls='--')
        ax2.plot(ph_range, np.rad2deg(fp.arccot(cot_dPhi_T1)), color='k', lw=0.5, ls='--')

        ax.fill_between(ph_range, cot_dPhi_T0, cot_dPhi_T1, color='grey', alpha=0.2)
        ax2.fill_between(ph_range, np.rad2deg(fp.arccot(cot_dPhi_T0)), np.rad2deg(fp.arccot(cot_dPhi_T1)),
                         color='grey', alpha=0.2)

        ax.plot(pH_list[0], cot_dPhi['fluoro0, phosphor0'][1], marker='D', color='orange')
        ax.plot(pH_list[1], cot_dPhi['fluoro1, phosphor0'][1], marker='D', color='orange')
        ax.plot(pH_list[0], cot_dPhi['fluoro0, phosphor1'][1], marker='o', alpha=0.5, color='orange')
        ax.plot(pH_list[1], cot_dPhi['fluoro1, phosphor1'][1], marker='o', alpha=0.5, color='orange')

        ax.axhline(para_f['phosphor0']['bottom'], color='k', lw=0.5, ls='--')
        ax.axhline(para_f['phosphor1']['bottom'], color='k', lw=0.5, ls='--')

        ax.set_xlim(0, 15)
        ax.set_ylabel('cot(Φ)', color='navy', fontsize=9)
        ax2.set_ylabel('Φ [deg]', color='forestgreen', fontsize=9)
        ax.set_xlabel('pH', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        f.canvas.draw()

    def plot_pH(self, f, ax, ax2, pH_scan, cot_dPhi, pH_calc, pH_mean, pH_std, pH_calib):
        ax.cla()
        ax2.cla()

        if ax is None:
            f, ax = plt.subplots(figsize=(20, 10))
        if ax2 is None:
            ax2 = ax.twinx()

        # preparation of figure plot - mean value
        ax.plot(pH_scan, cot_dPhi[1], color='navy', lw=1.)
        ax2.plot(pH_scan, np.rad2deg(fp.arccot(cot_dPhi[1])), color='forestgreen', lw=1.)
        ax.legend(['pH = {:.2f} ± {:.2f}'.format(pH_mean, pH_std)], frameon=True, framealpha=0.5, fancybox=True,
                  loc='upper center', bbox_to_anchor=(0.2, 0.25), fontsize=9)

        ax.plot(pH_scan, cot_dPhi[0], color='k', lw=.25, ls='--')
        ax.plot(pH_scan, cot_dPhi[2], color='k', lw=.25, ls='--')
        ax2.plot(pH_scan, np.rad2deg(fp.arccot(cot_dPhi[0])), color='k', lw=0.5, ls='--')
        ax2.plot(pH_scan, np.rad2deg(fp.arccot(cot_dPhi[2])), color='k', lw=0.5, ls='--')
        ax.fill_between(pH_scan, cot_dPhi[0], cot_dPhi[2], color='grey', alpha=0.2)
        ax2.fill_between(pH_scan, np.rad2deg(fp.arccot(cot_dPhi[0])), np.rad2deg(fp.arccot(cot_dPhi[2])), color='grey',
                         alpha=0.2)

        ax.axvline(pH_calc[0], color='k', ls='--', lw=.5)
        ax.axvline(pH_calc[2], color='k', ls='--', lw=.5)
        ax.axvspan(pH_calc[0], pH_calc[2], alpha=0.1, color='#f0810f')

        # find closest y value (lifetime phosphor)
        cot_dPhi_min = fp.find_closest_value_(index=pH_scan, data=cot_dPhi[0], value=pH_calc[0])
        cot_dPhi_max = fp.find_closest_value_(index=pH_scan, data=cot_dPhi[0], value=pH_calc[2])

        # linear regression to tauP measured
        arg_min = stats.linregress(x=pH_calib, y=cot_dPhi_min[2:])
        arg_max = stats.linregress(x=pH_calib, y=cot_dPhi_max[2:])
        y_min = arg_min[0] * pH_calc[0] + arg_min[1]
        y_max = arg_max[0] * pH_calc[2] + arg_max[1]
        ax.axhline(y_min, color='k', ls='--', lw=.4)
        ax.axhline(y_max, color='k', ls='--', lw=.4)
        ax.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        ax.set_xlabel('pH', fontsize=9)
        ax.set_ylabel('cot(Φ)', color='navy', fontsize=9)
        ax2.set_ylabel('Φ [deg]', color='forestgreen', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        f.canvas.draw()

    def plot_pH_meas(self, f, ax, ax1, pH_scan, cot_dPhi_min, cot_dPhi_max, dPhi_min, dPhi_max, Phi_meas, pH_calc,
                     clear=True):
        if clear is True:
            ax.cla()
            ax1.cla()

        if ax is None:
            f, ax = plt.subplots(figsize=(20, 10))
        if ax1 is None:
            ax1 = ax.twinx()

        # pH boltzmann sigmoid
        cot_dPhi_mean = pd.concat([pd.DataFrame(cot_dPhi_min[0]), pd.DataFrame(cot_dPhi_max[0])], axis=1).mean(axis=1)
        dPhi_mean = pd.concat([pd.DataFrame(dPhi_min[0]), pd.DataFrame(dPhi_max[0])], axis=1).mean(axis=1)

        # Boltzmann fit - mean values (cot(dPhi) and dPhi)
        ax.plot(pH_scan, cot_dPhi_mean, color='navy', lw=1.)
        ax1.plot(pH_scan, dPhi_mean, color='forestgreen', lw=1.)

        # measurement range (min - max values)
        ax.plot(pH_scan, cot_dPhi_min[0], color='k', lw=.5, ls='--')
        ax.plot(pH_scan, cot_dPhi_min[0], color='k', lw=.5, ls='--')
        ax.fill_between(pH_scan, cot_dPhi_min[0], cot_dPhi_max[0], color='grey', alpha=0.2)

        ax1.plot(pH_scan, np.rad2deg(fp.arccot(cot_dPhi_min[0])), color='k', lw=0.5, ls='--')
        ax1.plot(pH_scan, np.rad2deg(fp.arccot(cot_dPhi_max[0])), color='k', lw=0.5, ls='--')
        ax1.fill_between(pH_scan, np.rad2deg(fp.arccot(cot_dPhi_min[0])), np.rad2deg(fp.arccot(cot_dPhi_max[0])),
                         color='grey', alpha=0.2)

        if len(pH_calc) > 1:
            ax.legend(['pH #1 = {:.2f} ± {:.2f}'.format(pH_calc[0][1], (pH_calc[0][2]-pH_calc[0][0])/2),
                       'pH #{} = {:.2f} ± {: .2f}'.format(len(pH_calc[0]), pH_calc[0][1],
                                                          (pH_calc[0][2]-pH_calc[0][0])/2)], loc=0, frameon=True,
                      framealpha=0.5, fancybox=True, fontsize=9)
        else:
            ax.legend(['pH = {:.2f} ± {:.2f}'.format(pH_calc[0][1], (pH_calc[0][2] - pH_calc[0][0]) / 2)], loc=0,
                      frameon=True, framealpha=0.5, fancybox=True, fontsize=9)

        ax1.axvline(pH_calc[0][0], color='k', lw=0.5, ls='--')
        ax1.axvline(pH_calc[0][2], color='k', lw=0.5, ls='--')
        ax1.axvspan(pH_calc[0][0], pH_calc[0][2], color='#f0810f', alpha=0.2)

        ax1.axhline(Phi_meas[0][0], color='k', lw=0.5, ls='--')
        ax1.axhline(Phi_meas[0][2], color='k', lw=0.5, ls='--')
        ax1.axhspan(Phi_meas[0][0], Phi_meas[0][2], color='#f0810f', alpha=0.2)

        if len(pH_calc) > 1:
            # last measurement point
            last = len(pH_calc) - 1
            ax1.axvline(pH_calc[last][0], color='k', lw=0.5, ls='--')
            ax1.axvline(pH_calc[last][2], color='k', lw=0.5, ls='--')
            ax1.axvspan(pH_calc[last][0], pH_calc[last][2], color='#f0810f', alpha=0.2)

            ax1.axhline(Phi_meas[last][0], color='k', lw=0.5, ls='--')
            ax1.axhline(Phi_meas[last][2], color='k', lw=0.5, ls='--')
            ax1.axhspan(Phi_meas[last][0], Phi_meas[last][2], color='#f0810f', alpha=0.2)

        # layout
        ax.set_xlabel('pH', fontsize=13)
        ax1.set_ylabel('dPhi [°]', color='forestgreen', fontsize=9)
        ax.set_ylabel('cot(dPhi)', color='navy', fontsize=9)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        f.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        f.canvas.draw()

# ---------------------------------------------------------------------------------------------------------------------
# dual sensor calculations 4th tab
    def pH_temp_sensing(self):
        print('#--------------------------------------')
        self.run_tab4_button.setStyleSheet(
            "color: white; background-color: #2b5977; border-width: 1px; border-color: #077487; border-style: solid;"
            " border-radius: 7; padding: 5px; font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px;"
            " max-height: 18px;")
        print('pH / T dualsensing')

        # status of progressbar
        self.progress_tab4.setValue(0)

        # clear everything
        self.message_tab4.clear()

        # calibration pH at 2 frequencies
        self.ax_pH_calib2.cla()
        self.ax_pH_calib2_mir.cla()
        self.fig_pH_calib2.clear()
        self.ax_pH_calib2 = self.fig_pH_calib2.gca()
        self.ax_pH_calib2_mir = self.ax_pH_calib2.twinx()
        self.ax_pH_calib2.set_xlim(0, 15)
        self.ax_pH_calib2.set_xlabel('pH', fontsize=9)
        self.ax_pH_calib2.set_ylabel('cot(Φ)', fontsize=9)
        self.ax_pH_calib2_mir.set_ylabel('Φ [deg]', fontsize=9)
        self.ax_pH_calib2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH_calib2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        self.fig_pH_calib2.canvas.draw()

        # calibration temp compensation
        self.ax_temp_calib[0][0].cla()
        self.ax_temp_calib[1][0].cla()
        self.ax_temp_calib[0][1].cla()
        self.ax_temp_calib[1][1].cla()
        self.fig_temp_calib.clear()
        self.ax_temp_calib = self.fig_temp_calib.subplots(nrows=2, ncols=2, sharex=True)
        self.ax_temp_calib[0][0].set_xlim(0, 50)
        self.ax_temp_calib[1][0].set_xlim(0, 50)
        self.ax_temp_calib[0][1].set_xlim(0, 50)
        self.ax_temp_calib[1][1].set_xlim(0, 50)
        self.ax_temp_calib[1][0].set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp_calib[1][1].set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp_calib[0][0].set_ylabel('slope', fontsize=9)
        self.ax_temp_calib[0][1].set_ylabel('bottom', fontsize=9)
        self.ax_temp_calib[1][0].set_ylabel('pka', fontsize=9)
        self.ax_temp_calib[1][1].set_ylabel('top', fontsize=9)
        self.ax_temp_calib[0][0].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True,
                                             right=True)
        self.ax_temp_calib[1][0].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True,
                                             right=True)
        self.ax_temp_calib[0][1].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True,
                                             right=True)
        self.ax_temp_calib[1][1].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True,
                                             right=True)
        self.fig_temp_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_temp_calib.canvas.draw()

        # temperature sensing
        self.ax_temp.cla()
        self.fig_temp.clear()
        self.ax_temp = self.fig_temp.gca()
        self.ax_temp.set_xlim(0, 50)
        self.ax_temp.set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_temp.set_ylabel('$τ_P$ [µs]', fontsize=9)
        self.ax_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_temp.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_temp.canvas.draw()

        # pH sensing
        self.ax_pH2.cla()
        self.ax_pH2_mir.cla()
        self.fig_pH2.clear()
        self.ax_pH2 = self.fig_pH2.gca()
        self.ax_pH2_mir = self.ax_pH2.twinx()
        self.ax_pH2.set_xlim(0, 15)
        self.ax_pH2.set_xlabel('pH', fontsize=9)
        self.ax_pH2.set_ylabel('cot(Φ)', fontsize=9)
        self.ax_pH2_mir.set_ylabel('Φ [deg]', fontsize=9)
        self.ax_pH2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pH2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
        self.fig_pH2.canvas.draw()

    # -----------------------------------------------
        if self.tab4_fit_bottom.isChecked() is False:
            fit_parameter_failed = QMessageBox()
            fit_parameter_failed.setIcon(QMessageBox.Information)
            fit_parameter_failed.setText("Fitting the bottom value is mandatory!")
            fit_parameter_failed.setInformativeText("At least the bottom value has to be fitted...")
            fit_parameter_failed.setWindowTitle("Error!")
            fit_parameter_failed.exec_()
            self.tab4_fit_bottom.toggle()

        # status of progressbar
        self.progress_tab4.setValue(5)

    # ----------------------------------------------------------------------------------------------------------------
        # check input parameter of calibration
        for i in range(self.tableCalibration.rowCount()):
            for j in [0, 1]:
                try:
                    self.tableCalibration.item(i, j).text()
                except:
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("2 calibration points for each parameter "
                                                               "(pH and T) are required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

            if self.tableCalibration.item(i, j).text() == '':
                calibration_para_failed = QMessageBox()
                calibration_para_failed.setIcon(QMessageBox.Information)
                calibration_para_failed.setText("Insufficient calibration parameters!")
                calibration_para_failed.setInformativeText("2 calibration points for each parameter "
                                                           "(pH and T) are required.")
                calibration_para_failed.setWindowTitle("Error!")
                calibration_para_failed.exec_()
                return

        # pH list
        self.pH_list = []
        self.pH_order = []
        self.pH_list, self.pH_order = self.extract_calibration_points(para_list=self.pH_list, para_order=self.pH_order,
                                                                      cols=0, calib_type='2point')

        # T list
        self.temp_list = []
        self.temp_order = []
        self.temp_list, self.temp_order = self.extract_calibration_points(para_list=self.temp_list, calib_type='2point',
                                                                          para_order=self.temp_order, cols=1)

        # combine pH and pO2 to obtain the input order
        calib_points = []
        for i in range(len(self.pH_order)):
            r = self.pH_order[i], self.temp_order[i]
            calib_points.append(r)

        if len(list(set(calib_points))) < 4:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("4 different calibration points for the analytes (pH and "
                                                       "T) are required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # re-check if there are 2pH and 2T values
        if len(self.pH_list) != len(self.temp_list):
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("2 calibration points for each parameter (pH and T) are"
                                                       " required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # status of progressbar
        self.progress_tab4.setValue(7.5)

        # -------------------------------------------------------------
        # check temperature compensation file
        try:
            self.comp_file
            if self.comp_file is None:
                calibration_para_failed = QMessageBox()
                calibration_para_failed.setIcon(QMessageBox.Information)
                calibration_para_failed.setText("Error for calibration parameters!")
                calibration_para_failed.setInformativeText("File for temperature compensation is required!")
                calibration_para_failed.setWindowTitle("Error!")
                calibration_para_failed.exec_()
                return
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("File for temperature compensation is required!")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # -----------------------------------------------------------------------------------------------
        # Collect additional input parameters
        [self.f1, self.f2, self.error_assumed, self.int_fluoro_max, self.int_phosphor_max] = self.input_collecting()
        self.ph_range = np.linspace(start=0, stop=14, num=int((14-0)/0.1+1))
        self.temp_range_K = np.linspace(start=-50+conv_temp, stop=100+conv_temp, num=int((100--50)/0.02+1))
        self.temp_range_deg = np.linspace(start=-50, stop=100, num=int((100--50)/0.02+1))
        print('current tab', self.tabs.currentIndex()+1, ' - pH/T dualsensor')

        # status of progressbar
        self.progress_tab4.setValue(10)

# -------------------------------------------------------------------------------------------------------------
# if simulation is True
# -------------------------------------------------------------------------------------------------------------
        if self.simulation_checkbox.isChecked() is True:
            # amplitude ratio not useful
            if self.int_ratio_checkbox.isChecked() is True:
                pass
            else:
                self.int_ratio_checkbox.setCheckState(True)
                self.amplitude_to_intensity()

            # clear dPhi from calibration/input table (pH in row0, T in row1)
            self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                                   cols=[2, 3, 4, 5, 6])
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[2, 3, 4, 5, 6])

            # Measurement points pH
            try:
                self.tableINPUT.item(0, 0).text()
                if np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')) > 14. or \
                                np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input pH between 0 - 14.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("pH and T content have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            # Measurement points T
            try:
                self.tableINPUT.item(0, 1).text()
                if np.float64(self.tableINPUT.item(0, 1).text().replace(',', '.')) > 50. or \
                                np.float64(self.tableINPUT.item(0, 1).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input temperature between 0 - 50°C.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("pH and T have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            self.pH_list.append(np.float64(self.tableINPUT.item(0, 0).text().replace(',', '.')))
            self.temp_list.append(np.float64(self.tableINPUT.item(0, 1).text().replace(',', '.')))

            # status of progressbar
            self.progress_tab4.setValue(15)

        # ------------------------------------------------------------------------------------------------------------
            # Input parameter simulation
            [self.Phi_f1_deg_er, Phi_f1_deg, self.Phi_f2_deg_er, Phi_f2_deg, cotPhi, self.parameters, para_fit,
             para_f] = pHtemp.input_simulation_pH_T(file=self.comp_file, T0=self.temp_list[0], T1=self.temp_list[1],
                                                    T_meas=self.temp_list[2], f1=self.f1, f2=self.f2,
                                                    er=self.error_assumed, temp_range=self.temp_range_deg,
                                                    ph_range=self.ph_range, ph0=self.pH_list[0], ph1=self.pH_list[1],
                                                    ph_meas=self.pH_list[2], fit_slope=self.tab4_fit_slope.isChecked(),
                                                    fit_bottom=self.tab4_fit_bottom.isChecked(), plotting_=False,
                                                    fit_top=self.tab4_fit_top.isChecked(), plot_T_comp_f1=False,
                                                    fit_pka=self.tab4_fit_v50.isChecked(), plot_T_comp_f2=False)

            # status of progressbar
            self.progress_tab4.setValue(30)

            # conversion self.Phi_f1_deg
            self.Phi_f1_deg = pd.Series({'fluoro0, phosphor0': Phi_f1_deg['calib, deg'].loc[self.pH_list[0], self.temp_list[0]],
                                         'fluoro0, phosphor1': Phi_f1_deg['calib, deg'].loc[self.pH_list[0], self.temp_list[1]],
                                         'fluoro1, phosphor0': Phi_f1_deg['calib, deg'].loc[self.pH_list[1], self.temp_list[0]],
                                         'fluoro1, phosphor1': Phi_f1_deg['calib, deg'].loc[self.pH_list[1], self.temp_list[1]],
                                         'meas': Phi_f1_deg['meas, deg']})
            self.Phi_f2_deg = pd.Series({'fluoro0, phosphor0': Phi_f2_deg['calib, deg'].loc[self.pH_list[0], self.temp_list[0]],
                                         'fluoro0, phosphor1': Phi_f2_deg['calib, deg'].loc[self.pH_list[0], self.temp_list[1]],
                                         'fluoro1, phosphor0': Phi_f2_deg['calib, deg'].loc[self.pH_list[1], self.temp_list[0]],
                                         'fluoro1, phosphor1': Phi_f2_deg['calib, deg'].loc[self.pH_list[1], self.temp_list[1]],
                                         'meas': Phi_f2_deg['meas, deg']})
            # status of progressbar
            self.progress_tab4.setValue(35)

        # ------------------------------------------------------
            # write superimposed phase angles to table
            for i in range(self.tableCalibration.rowCount()):
                if self.pH_order[i] == self.pH_list[0]:
                    if self.temp_order[i] == self.temp_list[0]:
                        phi_f1 = self.Phi_f1_deg_er['fluoro0, phosphor0'][1]
                        phi_f2 = self.Phi_f2_deg_er['fluoro0, phosphor0'][1]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                    else:
                        phi_f1 = self.Phi_f1_deg_er['fluoro0, phosphor1'][1]
                        phi_f2 = self.Phi_f2_deg_er['fluoro0, phosphor1'][1]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                else:
                    if self.temp_order[i] == self.temp_list[0]:
                        phi_f1 = self.Phi_f1_deg_er['fluoro1, phosphor0'][1]
                        phi_f2 = self.Phi_f2_deg_er['fluoro1, phosphor0'][1]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
                    else:
                        phi_f1 = self.Phi_f1_deg_er['fluoro1, phosphor1'][1]
                        phi_f2 = self.Phi_f2_deg_er['fluoro1, phosphor1'][1]
                        self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                        self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))
            self.tableINPUT.setItem(0, 4, QTableWidgetItem('{:.2f}'.format(self.Phi_f1_deg_er['meas'][1])))
            self.tableINPUT.setItem(0, 5, QTableWidgetItem('{:.2f}'.format(self.Phi_f2_deg_er['meas'][1])))

            # status of progressbar
            self.progress_tab4.setValue(50)

        # ------------------------------------------------------------------------------------------------------------
            # Fit for plotting - temperature calibration
            [ddf, self.fit_parameter, para_meas] = self.temp_compensation(file=self.comp_file, T_meas=self.temp_list[2],
                                                                          temp_range=self.temp_range_deg,
                                                                          fit_k=self.tab4_fit_slope.isChecked(),
                                                                          fit_pka=self.tab4_fit_v50.isChecked(),
                                                                          fit_bottom=self.tab4_fit_bottom.isChecked(),
                                                                          fit_top=self.tab4_fit_top.isChecked())

            self.plot_temp_compensation(fig=self.fig_temp_calib, ax=self.ax_temp_calib,
                                        ddf=self.parameters['T compensation'], temp_range=self.temp_range_deg,
                                        T_meas=self.temp_list[2], para_meas=para_meas, fit_parameter=self.fit_parameter,
                                        fit_k=self.tab4_fit_slope.isChecked(), fit_pka=self.tab4_fit_v50.isChecked(),
                                        fit_bottom=self.tab4_fit_bottom.isChecked(),
                                        fit_top=self.tab4_fit_top.isChecked())

            # status of progressbar
            self.progress_tab4.setValue(60)

            #  Temperature compensation
            self.para_fit_f1 = pHtemp.linregression_temp_compensation_frequency(x=self.temp_list[:-1],
                                                                                para_f=para_f, f='f1')
            self.para_fit_f2 = pHtemp.linregression_temp_compensation_frequency(x=self.temp_list[:-1],
                                                                                para_f=para_f, f='f2')

            # ------------------------------------------------------
            # pH calibration
            self.plot_pH_calib(f=self.fig_pH_calib2, ax=self.ax_pH_calib2, ax2=self.ax_pH_calib2_mir,
                               ph_range=self.ph_range, pk_a=self.parameters['T compensation']['V50'].mean(),
                               cot_dPhi=cotPhi['f1'], para_f=para_f['f1'], pH_list=self.pH_list)

        # ------------------------------------------------------
            #  Update message box
            self.message_tab4.setText('Simulation - phaseangle at {:.2f}°C and pH {:.2}'.format(self.temp_list[2],
                                                                                                self.pH_list[2]))
            # status of progressbar
            self.progress_tab4.setValue(70)

        # ------------------------------------------------------------------------------------------------------------
            # Temperature sensing
            [self.temp_calc, para_temp,
             self.res_T] = pHtemp.temperature_sensing(Phi_f1_deg_er=self.Phi_f1_deg_er, fontsize_=13,
                                                      Phi_f2_deg_er=self.Phi_f2_deg_er, T_calib=self.temp_list[:-1],
                                                      f1=self.f1, f2=self.f2, option='moderate', plotting=False)
            self.temp_calc_std = np.abs(self.temp_calc[2] - self.temp_calc[0])/2
            self.message_tab4.append('Calculated T: {:.2f} ± {:.2f}°C'.format(self.temp_calc[1], self.temp_calc_std))

        # ------------------------------------------------------
            # pH sensing
            pH_f1, std_f1, df_ph_screen_f1 = pHtemp.pH_sensing(para_fit=self.para_fit_f1, temp=self.temp_list[:-1],
                                                               Phi_f1_deg_er=self.Phi_f1_deg_er, ph_range=self.ph_range,
                                                               T_range_K=self.temp_range_K, T_calc_deg=self.temp_calc,
                                                               f=self.f1, plotting=False, re_check=False)
            pH_f2, std_f2, df_ph_screen_f2 = pHtemp.pH_sensing(para_fit=self.para_fit_f2, temp=self.temp_list[:-1],
                                                               Phi_f1_deg_er=self.Phi_f2_deg_er, ph_range=self.ph_range,
                                                               T_range_K=self.temp_range_K, T_calc_deg=self.temp_calc,
                                                               f=self.f2, plotting=False, re_check=False)

            # remove nan from list
            pH_f1_cleaned = [x for x in pH_f1 if (math.isnan(x) is False)]
            pH_f2_cleaned = [x for x in pH_f2 if (math.isnan(x) is False)]
            self.pH_f1_mean = sum(pH_f1_cleaned) / float(len(pH_f1_cleaned))
            self.pH_f2_mean = sum(pH_f2_cleaned) / float(len(pH_f2_cleaned))
            self.pH_calc = (self.pH_f1_mean + self.pH_f2_mean)/2
            pH_std_ = pH_f1_cleaned + pH_f2_cleaned
            self.pH_calc_std = (statistics.stdev(pH_std_))
            self.message_tab4.append('Calculated pH: {:.2f} ± {:.2f} at T = {:.2f}°C'.format(self.pH_calc,
                                                                                             self.pH_calc_std,
                                                                                             self.temp_calc[1]))
            # status of progressbar
            self.progress_tab4.setValue(80)

        # ---------------------------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------------------------
            # Temperature sensing
            temp_scan = self.temperature_scanning()

            self.plot_temperature(f=self.fig_temp, ax=self.ax_temp, temp_scan=self.temp_range_K, res_T=temp_scan,
                                  temp_deg=self.temp_calc, temp_calib=self.temp_list[:-1])

            # status of progressbar
            self.progress_tab4.setValue(90)

            # ------------------------------------------------------
            # pH sensing
            self.plot_pH(f=self.fig_pH2, ax=self.ax_pH2, ax2=self.ax_pH2_mir, pH_scan=self.ph_range,
                         cot_dPhi=df_ph_screen_f1, pH_calc=pH_f1, pH_mean=self.pH_calc, pH_std=self.pH_calc_std,
                         pH_calib=self.pH_list[:-1])

            # status of progressbar
            self.progress_tab4.setValue(100)

# -------------------------------------------------------------------------------------------------------------
# if simulation is False - measurement evaluation
# -------------------------------------------------------------------------------------------------------------
        else:
            self.message_tab4.setText('Measurement evaluation')

            # clear everything form calibration /input table which is not required
            self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                                   cols=[2, 3, 6])
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[0, 1, 2, 3, 6])

            # check input parameter of calibration - dPhi required
            for i in range(self.tableCalibration.rowCount()):
                for j in [4, 5]:
                    try:
                        self.tableCalibration.item(i, j).text()
                    except:
                        calibration_para_failed = QMessageBox()
                        calibration_para_failed.setIcon(QMessageBox.Information)
                        calibration_para_failed.setText("Insufficient calibration parameters!")
                        calibration_para_failed.setInformativeText("2 calibration points for each parameter "
                                                                   "(pH and T) and corresponding dPhi are required.")
                        calibration_para_failed.setWindowTitle("Error!")
                        calibration_para_failed.exec_()
                        return

            # dPhi(f1)
            self.dPhi_list_f1 = []
            self.dPhi_order_f1 = []
            self.dPhi_list_f1, self.dPhi_order_f1 = self.extract_calibration_points(para_list=self.dPhi_list_f1,
                                                                                    para_order=self.dPhi_order_f1,
                                                                                    cols=4, calib_type='2point')

            # dPhi(f2)
            self.dPhi_list_f2 = []
            self.dPhi_order_f2 = []
            self.dPhi_list_f2, self.dPhi_order_f2 = self.extract_calibration_points(para_list=self.dPhi_list_f2,
                                                                                    para_order=self.dPhi_order_f2,
                                                                                    cols=5, calib_type='2point')

            # Measurement points dPhi(f1) and dPhi(f2)
            try:
                self.tableINPUT.item(0, 4).text()
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("dPhi(f1) has to be defined for the measurement evaluation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            try:
                self.tableINPUT.item(0, 5).text()
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("dPhi(f2) has to be defined for the measurement evaluation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            # combine superimposed phase angles for calibration and measurement
            phi_f1_pH0_T0 = self.extract_dPhi_calibration(pH_soll=self.pH_list[0], temp_soll=self.temp_list[0], row=4)
            phi_f1_pH0_T1 = self.extract_dPhi_calibration(pH_soll=self.pH_list[0], temp_soll=self.temp_list[1], row=4)
            phi_f1_pH1_T0 = self.extract_dPhi_calibration(pH_soll=self.pH_list[1], temp_soll=self.temp_list[0], row=4)
            phi_f1_pH1_T1 = self.extract_dPhi_calibration(pH_soll=self.pH_list[1], temp_soll=self.temp_list[1], row=4)

            phi_f2_pH0_T0 = self.extract_dPhi_calibration(pH_soll=self.pH_list[0], temp_soll=self.temp_list[0], row=5)
            phi_f2_pH0_T1 = self.extract_dPhi_calibration(pH_soll=self.pH_list[0], temp_soll=self.temp_list[1], row=5)
            phi_f2_pH1_T0 = self.extract_dPhi_calibration(pH_soll=self.pH_list[1], temp_soll=self.temp_list[0], row=5)
            phi_f2_pH1_T1 = self.extract_dPhi_calibration(pH_soll=self.pH_list[1], temp_soll=self.temp_list[1], row=5)

            # check the number of measurement points
            list_meas = []
            for i in range(self.tableINPUT.rowCount()):
                it = self.tableINPUT.item(i, 4)
                if it and it.text():
                    list_meas.append(it.text())

            phi_f1_meas = {}
            phi_f2_meas = {}
            for i in range(len(list_meas)):
                phi_f1_meas['meas {}'.format(i)] = np.float64(self.tableINPUT.item(i, 4).text().replace(',', '.'))
                phi_f2_meas['meas {}'.format(i)] = np.float64(self.tableINPUT.item(i, 5).text().replace(',', '.'))

            Phi_f1_deg = pd.Series({'fluoro0, phosphor0': phi_f1_pH0_T0, 'fluoro0, phosphor1': phi_f1_pH0_T1,
                                    'fluoro1, phosphor0': phi_f1_pH1_T0, 'fluoro1, phosphor1': phi_f1_pH1_T1})
            Phi_f2_deg = pd.Series({'fluoro0, phosphor0': phi_f2_pH0_T0, 'fluoro0, phosphor1': phi_f2_pH0_T1,
                                    'fluoro1, phosphor0': phi_f2_pH1_T0, 'fluoro1, phosphor1': phi_f2_pH1_T1})

            self.Phi_f1_deg = pd.concat([Phi_f1_deg, pd.Series(phi_f1_meas)], axis=0)
            self.Phi_f2_deg = pd.concat([Phi_f2_deg, pd.Series(phi_f2_meas)], axis=0)

            # status of progressbar
            self.progress_tab4.setValue(15)

            # -----------------------------------------------------------------------------------------------------
            # included error
            Phi_f1_deg_er = {}
            Phi_f2_deg_er = {}
            cotPhi_f1_er = {}
            cotPhi_f2_er = {}
            for ind in self.Phi_f1_deg.index:
                Phi_f1_deg_er[ind] = [self.Phi_f1_deg[ind] - self.error_assumed, self.Phi_f1_deg[ind],
                                      self.Phi_f1_deg[ind] + self.error_assumed]
                Phi_f2_deg_er[ind] = [self.Phi_f2_deg[ind] - self.error_assumed, self.Phi_f2_deg[ind],
                                      self.Phi_f2_deg[ind] + self.error_assumed]
            self.Phi_f1_deg_er = pd.Series(Phi_f1_deg_er)
            self.Phi_f2_deg_er = pd.Series(Phi_f2_deg_er)

            for ind in self.Phi_f1_deg.index:
                cotPhi_f1_er[ind] = fp.cot(np.deg2rad(self.Phi_f1_deg_er[ind]))
                cotPhi_f2_er[ind] = fp.cot(np.deg2rad(self.Phi_f2_deg_er[ind]))
            self.cotPhi_f1_er = pd.Series(cotPhi_f1_er)
            self.cotPhi_f2_er = pd.Series(cotPhi_f2_er)

            # status of progressbar
            self.progress_tab4.setValue(20)

        # ------------------------------------------------------------------------------------------------------------
            # Fit for plotting - temperature calibration
            [ddf, fit_para_T0_f1,
             self.para_fit_T0] = self.temp_compensation(file=self.comp_file, T_meas=self.temp_list[0],
                                                        temp_range=self.temp_range_deg,
                                                        fit_k=self.tab4_fit_slope.isChecked(),
                                                        fit_pka=self.tab4_fit_v50.isChecked(),
                                                        fit_bottom=self.tab4_fit_bottom.isChecked(),
                                                        fit_top=self.tab4_fit_top.isChecked())

            [ddf, fit_para_T1_f1,
             self.para_fit_T1] = self.temp_compensation(file=self.comp_file, T_meas=self.temp_list[1],
                                                        temp_range=self.temp_range_deg,
                                                        fit_k=self.tab4_fit_slope.isChecked(),
                                                        fit_pka=self.tab4_fit_v50.isChecked(),
                                                        fit_bottom=self.tab4_fit_bottom.isChecked(),
                                                        fit_top=self.tab4_fit_top.isChecked())
            para_f1 = pd.Series({'phosphor0': self.para_fit_T0, 'phosphor1': self.para_fit_T1})

            # status of progressbar
            self.progress_tab4.setValue(35)

            # --------------------------------------------------------------------------------------------------------
            # dual sensing
            [self.temp_calc, self.pH_calc,
             self.para] = pHtemp.pH_T_dualsensing_meas(file=self.comp_file, Phi_f1_deg_er=self.Phi_f1_deg_er,
                                                       Phi_f2_deg_er=self.Phi_f2_deg_er, f1=self.f1, f2=self.f2,
                                                       T_calib=self.temp_list, temp_range_deg=self.temp_range_deg,
                                                       plotting=False, fontsize_=9)
            # status of progressbar
            self.progress_tab4.setValue(55)

            # ---------------------------------------------------------------------------------------------------------
            # Calibration plots - plot for the first and the last measurement point
            self.plot_temp_compensation_meas(fig=self.fig_temp_calib, ax=self.ax_temp_calib, ddf=ddf,
                                             para_meas=self.para['parameter_pH_f1'][0], T_meas=self.temp_calc[0][1],
                                             fit_parameter=fit_para_T0_f1, temp_range=self.temp_range_deg,
                                             fit_k=self.tab4_fit_slope.isChecked(),
                                             fit_pka=self.tab4_fit_v50.isChecked(),
                                             fit_top=self.tab4_fit_top.isChecked(),
                                             fit_bottom=self.tab4_fit_bottom.isChecked())
            if len(list_meas) > 1:
                last = len(list_meas) -1
                self.plot_temp_compensation_meas(fig=self.fig_temp_calib, ax=self.ax_temp_calib, ddf=ddf,
                                                 para_meas=self.para['parameter_pH_f1'][last],
                                                 T_meas=self.temp_calc[last][1], fit_parameter=fit_para_T0_f1,
                                                 temp_range=self.temp_range_deg,
                                                 fit_k=self.tab4_fit_slope.isChecked(),
                                                 fit_pka=self.tab4_fit_v50.isChecked(),
                                                 fit_top=self.tab4_fit_top.isChecked(),
                                                 fit_bottom=self.tab4_fit_bottom.isChecked())

            # pH calibration
            self.plot_pH_calib_meas(f=self.fig_pH_calib2, ax=self.ax_pH_calib2, ax2=self.ax_pH_calib2_mir,
                                    ph_range=self.ph_range, pk_a=ddf['V50'].mean(), cot_dPhi=self.cotPhi_f1_er,
                                    para_f=para_f1, pH_list=self.pH_list)

            # status of progressbar
            self.progress_tab4.setValue(65)

            # ---------------------------------------------------------------------------------------------------------
            # Update message box
            for i in range(len(list_meas)):
                self.temp_calc_std = np.abs(self.temp_calc[i][2] - self.temp_calc[i][0]) / 2
                self.pH_calc_std = np.abs(self.pH_calc[i][2] - self.pH_calc[i][0]) / 2
                self.message_tab4.append('# Point - {}'.format(i+1))
                self.message_tab4.append('\t Calculated T: {:.2f} ± {:.2f}°C'.format(self.temp_calc[i][1], self.temp_calc_std))
                self.message_tab4.append('\t Calculated pH: {:.2f} ± {:.2f}'.format(self.pH_calc[i][1], self.pH_calc_std))

            # status of progressbar
            self.progress_tab4.setValue(70)

        # ---------------------------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------------------------
            # Temperature sensing
            self.plot_temperature_meas(f=self.fig_temp, ax=self.ax_temp, temp_scan=self.temp_range_K,
                                       y_reg_ph1=self.para['linear regression T'], tau_phos=self.para['tauP_calc'],
                                       temp_calc=self.temp_calc)

            # status of progressbar
            self.progress_tab4.setValue(80)

            # ------------------------------------------------------
            # pH sensing
            ls_m = []
            for c in self.Phi_f1_deg_er.index:
                if 'meas' in c:
                    ls_m.append(c)
            self.plot_pH_meas(f=self.fig_pH2, ax=self.ax_pH2, ax1=self.ax_pH2_mir, pH_scan=self.ph_range,
                              cot_dPhi_min=self.para['cotPhi_min'], cot_dPhi_max=self.para['cotPhi_max'],
                              dPhi_min=self.para['dPhi_min'], dPhi_max=self.para['dPhi_max'],
                              Phi_meas=self.Phi_f1_deg_er[ls_m], pH_calc=self.pH_calc)

            # status of progressbar
            self.progress_tab4.setValue(95)

        # ---------------------------------------------------------------------------
            # output preparation
            for i in range(len(list_meas)):
                self.pH_list.append(self.pH_calc[i][1])
                self.temp_list.append(self.temp_calc[i][1])

            # status of progressbar
            self.progress_tab4.setValue(100)

        print('pH / T dualsensing finished')
        self.run_tab4_button.setStyleSheet(
            "color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #227286, stop: 1 "
            "#54bad4); border-width: 1px; border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
            "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")
        print('#-------------------------------------------------------------------')


# ---------------------------------------------------------------------------------------------------------------------
# (C)O2 sensing
# ---------------------------------------------------------------------------------------------------------------------
    def plot_calibration_CO2_O2(self, pCO2_calib, pO2_calib, KsvO2, KsvCO2, Ksv2_O2, Ksv2_CO2, int_f, tau_quot, f1,
                                ax1, f2, ax2):
        # preparation of figure plot
        if ax1 is None:
            f1 = plt.figure()
            ax1 = f1.gca()
        if ax2 is None:
            f2 = plt.figure()
            ax2 = f2.gca()

        # ---------------------------------------------------------------------------------
        # pCO2 sensing
        ax1.plot(int_f.index, int_f, lw=0.75, color='#07575B')

        if isinstance(pCO2_calib, np.float):
            ax1.axvline(pCO2_calib, color='k', lw=0.5, ls='--')
            ax1.legend(['Ksv1 = {:.2e}, Ksv2 = {:.2e}'.format(KsvCO2, Ksv2_CO2),
                        'pCO$_{}$ = {:.2e} hPa'.format('2 - 1', pCO2_calib)], frameon=True, framealpha=0.5,
                       fancybox=True, loc=0, fontsize=8)
        else:
            ax1.axvline(pCO2_calib[0], color='k', lw=0.5, ls='--')
            ax1.axvline(pCO2_calib[1], color='k', lw=0.5, ls='--')

            ax1.legend(['Ksv1 = {:.2e}, Ksv2 = {:.2e}'.format(KsvCO2, Ksv2_CO2),
                        'pCO$_{}$$_{}$ = {:.2e} hPa'.format('2', '1', pCO2_calib[0]),
                        'pCO$_{}$$_{}$ = {:.2e} hPa'.format('2', '2', pCO2_calib[1])], frameon=True, framealpha=0.5,
                       fancybox=True, loc=0, fontsize=8)

        # ---------------------------------------------------------------------------------
        # O2 sensing
        ax2.plot(tau_quot.index, 1/tau_quot, color='#6fb98f', lw=1.) #004445
        ax2.axvline(pO2_calib[0], color='black', lw=0.5, ls='--')
        ax2.axvline(pO2_calib[1], color='black', lw=0.5, ls='--')

        xmax = tau_quot.index[-1]*1.05
        ymax = 1/tau_quot.values[-1][0]*1.05
        ymin = 1/tau_quot.values[0][0]*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        ax2.legend(['Ksv1 = {:.2e}, Ksv2 = {:.2e}'.format(KsvO2, Ksv2_O2),
                    'pO$_{}$$_{}$ = {:.2e} hPa'.format('2', '1', pO2_calib[0]),
                    'pO$_{}$$_{}$ = {:.2e} hPa'.format('2', '2', pO2_calib[1])], frameon=True, framealpha=0.5, fancybox=True,
                   loc=0, fontsize=8)

        ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_calibration_CO2_O2_measurement(self, pCO2_calib, pO2_calib, KsvO2, KsvCO2, int_f, tau_quot, f1, ax1, f2,
                                            ax2):
        # preparation of figure plot
        if ax1 is None:
            f1 = plt.figure()
            ax1 = f1.gca()
        if ax2 is None:
            f2 = plt.figure()
            ax2 = f2.gca()

        # ---------------------------------------------------------------------------------
        # pCO2 sensing
        p1, = ax1.plot(int_f.index, int_f['mean'], color='navy', lw=0.5, label='{:.1f}hPa'.format(pO2_calib[0]))
        ax1.plot(int_f.index, int_f['min'], color='k', ls='--', lw=0.5)
        ax1.plot(int_f.index, int_f['max'], color='k', ls='--', lw=0.5)

        ax1.legend(['Ksv = {:.2f}'.format(KsvCO2), 'pCO$_{}$$_{}$ = {:.2e} hPa'.format('2', '1', pCO2_calib[0]),
                    'pCO$_{}$$_{}$ = {:.2e} hPa'.format('2', '2', pCO2_calib[1])], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)

        ax1.fill_between(int_f.index, int_f['min'], int_f['max'], color='grey', alpha=0.15)
        ax1.axvline(pCO2_calib[0], color='black', lw=0.5, ls='--')
        ax1.axvline(pCO2_calib[1], color='black', lw=0.5, ls='--')
        # ax1.set_xlim(0, 100)
        ax1.set_ylim(0, int_f.loc[0, 'max']*1.05)

        # ---------------------------------------------------------------------------------
        # pO2 sensing
        if len(tau_quot.columns) > 2:
            ax2.plot(tau_quot.index, 1/tau_quot[1], color='#2a3132')
            ax2.plot(tau_quot.index, 1/tau_quot[0], color='k', ls='--', lw=0.5)
            ax2.plot(tau_quot.index, 1/tau_quot[2], color='k', ls='--', lw=0.5)
        else:
            ax2.plot(tau_quot.index, 1/tau_quot[0], color='#2a3132')

        ax2.legend(['Ksv = {:.2f}'.format(KsvO2), 'pO$_{}$$_{}$ = {:.2e} hPa'.format('2', '1', pO2_calib[0]),
                    'pO$_{}$$_{}$ = {:.2e} hPa'.format('2', '2', pO2_calib[1])], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)

        if len(tau_quot.columns) > 2:
            ax2.fill_between(tau_quot.index, 1/tau_quot[0], 1/tau_quot[2], color='grey', alpha=0.2)
            ax2.axvline(pO2_calib[0], color='black', lw=0.5, ls='--')
            ax2.axvline(pO2_calib[1], color='black', lw=0.5, ls='--')
        else:
            pass

        xmax = tau_quot.index[-1]*1.05
        xmin = tau_quot.index[0]*0.95
        if tau_quot.index[0] == 0.0:
            xmin = -5.

        ymax = 1/tau_quot.values[-1][0]*1.05
        ymin = 1/tau_quot.values[0][0]*0.95
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)

        ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_results_CO2_O2(self, pCO2, pCO2_calib, pO2_calc, pO2_calib, tauP, intF, f1, ax1, f2, ax2):

        # pCO2 sensing - calibration points
        ax1.plot(intF.index, intF['mean'], color='navy', lw=0.75)
        ax1.plot(intF.index, intF['min'], color='k', ls='--', lw=0.5)
        ax1.plot(intF.index, intF['max'], color='k', ls='--', lw=0.5)
        ax1.fill_between(intF.index, intF['min'], intF['max'], color='grey', alpha=0.15)

        # measurement points
        ax1.axvline(pCO2[0], color='k', ls='--', lw=.4)
        ax1.axvline(pCO2[2], color='k', ls='--', lw=.4)
        ax1.axvspan(pCO2[0], pCO2[2], color='#f0810f', alpha=0.2)

        # find closest values
        iF_meas_min = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2[0])
        iF_meas_max = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2[2])

        # linear regression to pH measured
        arg_min = stats.linregress(x=pCO2_calib, y=iF_meas_min[2:])
        arg_max = stats.linregress(x=pCO2_calib, y=iF_meas_max[2:])
        y_min = arg_min[0] * pCO2[0] + arg_min[1]
        y_max = arg_max[0] * pCO2[2] + arg_max[1]
        ax1.axhline(y_min, color='k', ls='--', lw=.4)
        ax1.axhline(y_max, color='k', ls='--', lw=.4)
        ax1.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        if len(pCO2) == 3:
            pCO2_mean = pCO2[1]
        else:
            pCO2_mean = pCO2.mean()
        ax1.legend(['pCO2 = {:.2f} ± {:.2e} hPa'.format(pCO2_mean, pCO2.std())], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)
        y_max = intF['max'].max()*1.05
        if intF['min'].min() < 1:
            if intF['min'].min() < 0:
                y_min = -np.abs(intF['min'].min())*0.95
            else:
                y_min = -5.
        else:
            y_min = intF['min'].min()*0.95
        ax1.set_ylim(y_min, y_max)

    # --------------------------------------------------------------------------------
        # O2 sensing - calibration points
        ax2.plot(tauP.index, tauP[1], color='#021c1e', lw=.75)
        ax2.plot(tauP[0], color='k', lw=.25, ls='--')
        ax2.plot(tauP[2], color='k', lw=.25, ls='--')
        ax2.fill_between(tauP.index, tauP[0], tauP[2], color='grey', alpha=0.2)

        # measurement points
        ax2.axvline(pO2_calc[0], color='k', lw=0.4, ls='--')
        ax2.axvline(pO2_calc[2], color='k', lw=0.4, ls='--')
        ax2.axvspan(pO2_calc[0], pO2_calc[2], color='#f0810f', alpha=0.2)

        # find closest value
        tauq_meas_min = fp.find_closest_value_(index=tauP.index, data=tauP[0].values, value=pO2_calc[0])
        tauq_meas_max = fp.find_closest_value_(index=tauP.index, data=tauP[2].values, value=pO2_calc[2])

        # linear regression to pO2 measured
        arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
        arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
        y_min = arg_min[0] * pO2_calc[0] + arg_min[1]
        y_max = arg_max[0] * pO2_calc[2] + arg_max[1]
        ax2.axhline(y_min, color='k', lw=0.4, ls='--')
        ax2.axhline(y_max, color='k', lw=0.4, ls='--')
        ax2.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        if len(pO2_calc) == 3:
            pO2_mean = pO2_calc[1]
        else:
            pO2_mean = pO2_calc.mean()
        ax2.legend(['pO2 = {:.2f} ± {:.2f} hPa'.format(pO2_mean, pO2_calc.std())], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)
        xmax = tauP.index[-1]*1.05
        ymax = tauP.values[0].max()*1.05
        ymin = tauP.values[-1].min()*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

    def plot_results_CO2_O2_meas(self, pCO2, pCO2_calib, pO2_calc, pO2_calib, intF, tauP, f1, ax1, f2, ax2):
        # pCO2 sensing - calibration points
        if np.any(np.isnan(intF)) == False:
            ax1.plot(intF.index, intF['mean'], color='#336b87', lw=0.75)
            ax1.plot(intF.index, intF['min'], color='k', ls='--', lw=0.5)
            ax1.plot(intF.index, intF['max'], color='k', ls='--', lw=0.5)
            ax1.fill_between(intF.index, intF['min'], intF['max'], color='grey', alpha=0.15)
        else:
            ax1.plot(intF.index, intF[intF.columns[1]], color='k', ls='--', lw=0.5)
            ax1.plot(intF.index, intF[intF.columns[-1]], color='k', ls='--', lw=0.5)
            ax1.fill_between(intF.index, intF[intF.columns[1]], intF[intF.columns[-1]], color='grey', alpha=0.15)

        # measurement point
        if len(pCO2) == 3:
            ax1.axvline(pCO2[0], color='k', ls='--', lw=.4)
            ax1.axvline(pCO2[2], color='k', ls='--', lw=.4)
            ax1.axvspan(pCO2[0], pCO2[2], color='#f0810f', alpha=0.2)
        else:
            ax1.axvline(pCO2[0], color='k', ls='--', lw=.4)
            ax1.axvline(pCO2[1], color='k', ls='--', lw=.4)
            ax1.axvspan(pCO2[0], pCO2[1], color='#f0810f', alpha=0.2)

        # find closest values
        iF_meas_min = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2[0])
        iF_meas_max = fp.find_closest_value_(index=intF.index, data=intF['mean'].values, value=pCO2[-1])

        # linear regression to pCO2 measured
        arg_min = stats.linregress(x=pCO2_calib, y=iF_meas_min[2:])
        arg_max = stats.linregress(x=pCO2_calib, y=iF_meas_max[2:])
        y_min = arg_min[0] * pCO2[0] + arg_min[1]
        y_max = arg_max[0] * pCO2[-1] + arg_max[1]
        ax1.axhline(y_min, color='k', ls='--', lw=.4)
        ax1.axhline(y_max, color='k', ls='--', lw=.4)
        ax1.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        if len(pCO2) == 3:
            pCO2_mean = pCO2[1]
        else:
            pCO2_mean = pCO2.mean()
        ax1.legend(['pCO2 = {:.2f} ± {:.2e} hPa'.format(pCO2_mean, pCO2.std())], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)

        y_max = intF['max'].max()*1.05
        if np.isnan(intF['min'].min()) == True:
            y_min = 0.
        else:
            if intF['min'].min() < 1:
                if intF['min'].min() < 0:
                    y_min = -np.abs(intF['min'].min())*0.95
                else:
                    y_min = -5.
            else:
                y_min = intF['min'].min()*0.95
        ax1.set_ylim(y_min, y_max)

    # --------------------------------------------------------------------------------
        # O2 sensing - calibration points
        if np.any(np.isnan(tauP)) == False:
            ax2.plot(tauP.index, tauP[1], color='#021c1e', lw=.75)
            ax2.plot(tauP[0], color='k', lw=.25, ls='--')
            ax2.plot(tauP[2], color='k', lw=.25, ls='--')
            ax2.fill_between(tauP.index, tauP[0], tauP[2], color='grey', alpha=0.2)
        else:
            ax2.plot(tauP[1], color='k', lw=.25, ls='--')
            ax2.plot(tauP[-1], color='k', lw=.25, ls='--')
            ax2.fill_between(tauP.index, tauP[1], tauP[-1], color='grey', alpha=0.2)

        # measurement points
        if len(pO2_calc) == 3:
            ax2.axvline(pO2_calc[0], color='k', lw=0.4, ls='--')
            ax2.axvline(pO2_calc[2], color='k', lw=0.4, ls='--')
            ax2.axvspan(pO2_calc[0], pO2_calc[2], color='#f0810f', alpha=0.2)
        else:
            ax2.axvline(pO2_calc[0], color='k', lw=0.4, ls='--')
            ax2.axvline(pO2_calc[-1], color='k', lw=0.4, ls='--')
            ax2.axvspan(pO2_calc[0], pO2_calc[-1], color='#f0810f', alpha=0.2)

        # find closest value
        tauq_meas_min = fp.find_closest_value_(index=tauP.index, data=tauP[0].values, value=pO2_calc[0])
        tauq_meas_max = fp.find_closest_value_(index=tauP.index, data=tauP[2].values, value=pO2_calc[2])

        # linear regression to pO2 measured
        arg_min = stats.linregress(x=pO2_calib, y=tauq_meas_min[2:])
        arg_max = stats.linregress(x=pO2_calib, y=tauq_meas_max[2:])
        y_min = arg_min[0] * pO2_calc[0] + arg_min[1]
        y_max = arg_max[0] * pO2_calc[-1] + arg_max[1]
        ax2.axhline(y_min, color='k', lw=0.4, ls='--')
        ax2.axhline(y_max, color='k', lw=0.4, ls='--')
        ax2.axhspan(y_min, y_max, color='#f0810f', alpha=0.2)

        # legend
        ax2.legend(['pO2 = {:.2f} ± {:.2f} hPa'.format(pO2_calc.mean(), pO2_calc.std())], frameon=True, framealpha=0.5,
                   fancybox=True, loc=0, fontsize=9)
        xmax = tauP.index[-1]*1.05
        ymax = tauP.values[0].max()*1.05
        ymin = tauP.values[-1].min()*0.95
        ax2.set_xlim(-10., xmax)
        ax2.set_ylim(ymin, ymax)

        f1.tight_layout()
        f2.tight_layout()
        f1.canvas.draw()
        f2.canvas.draw()

# ---------------------------------------------------------------------------------------------------------------------
# dual sensor calculations 5th tab
    def CO2_O2_sensing(self):
        print('#--------------------------------------')
        self.run_tab5_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px; "
                                           "border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
                                           "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; "
                                           "max-height: 18px;")
        print('(C)O2 dualsensing')

        # set progress status
        self.progress_tab5.setValue(0)

        # clear everything
        self.message_tab5.clear()

        # calibration pCO2 at 2 frequencies
        self.ax_CO2calib.cla()
        self.fig_CO2calib.clear()
        self.ax_CO2calib = self.fig_CO2calib.gca()
        self.ax_CO2calib.set_xlim(0, 100)
        self.ax_CO2calib.set_ylim(0, 100)
        self.ax_CO2calib.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
        self.ax_CO2calib.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
        self.ax_CO2calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_CO2calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_CO2calib.canvas.draw()

        # pH sensing
        self.ax_CO2.cla()
        self.fig_CO2.clear()
        self.ax_CO2 = self.fig_CO2.gca()
        self.ax_CO2.set_xlim(0, 100)
        self.ax_CO2.set_ylim(0, 100)
        self.ax_CO2.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
        self.ax_CO2.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
        self.ax_CO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_CO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_CO2.canvas.draw()

        # calibration pO2 sensing
        self.ax_pO2calib_2.cla()
        self.fig_pO2calib2.clear()
        self.ax_pO2calib_2 = self.fig_pO2calib2.gca()
        self.ax_pO2calib_2.set_xlim(0, 100)
        self.ax_pO2calib_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2calib_2.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
        self.ax_pO2calib_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2calib2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pO2calib2.canvas.draw()

        # O2 sensing
        self.ax_pO2_2.cla()
        self.fig_pO2_2.clear()
        self.ax_pO2_2 = self.fig_pO2_2.gca()
        self.ax_pO2_2.set_xlim(0, 100)
        self.ax_pO2_2.set_ylim(0, 105)
        self.ax_pO2_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2_2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
        self.ax_pO2_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2_2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pO2_2.canvas.draw()

# ----------------------------------------------------------------------------------------------------------------
        # Entries for calibration points
        for i in range(self.tableCalibration.rowCount()-2):
            for j in [2, 3]:
                try:
                    self.tableCalibration.item(i, j).text()
                except:
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("2 calibration points for CO2 and O2 are required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

                if self.tableCalibration.item(i, j).text() == '':
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("2 calibration points for CO2 and O2 are required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

        # pCO2 list
        self.pCO2_list = []
        pCO2_order = []
        self.pCO2_list, pCO2_order = self.extract_calibration_points(para_list=self.pCO2_list, para_order=pCO2_order,
                                                                     cols=2, calib_type='1point')

        # pO2 list
        self.pO2_list = []
        pO2_order = []
        self.pO2_list, pO2_order = self.extract_calibration_points(para_list=self.pO2_list, para_order=pO2_order,
                                                                   cols=3, calib_type='1point')

        # re-check if there are 2pCO2 and 2pO2 values
        if len(self.pCO2_list) != len(self.pO2_list):
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("2 calibration points for each parameter (pCO2 and O2) are"
                                                       " required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # combine pCO2 and pO2 to obtain the input order
        calib_points = []
        for i in range(len(pCO2_order)):
            r = pCO2_order[i], pO2_order[i]
            calib_points.append(r)

        if len(list(set(calib_points))) < 2:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("2 different calibration points for the analytes (pCO2 and "
                                                       "pO2) are required.")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # set progress status
        self.progress_tab5.setValue(5)

        # -------------------------------------------------------------
        # check tauP -> intP conversion file (for pO2 sensing)
        try:
            self.conv_file
            if self.conv_file is None:
                calibration_para_failed = QMessageBox()
                calibration_para_failed.setIcon(QMessageBox.Information)
                calibration_para_failed.setText("Error for calibration parameters!")
                calibration_para_failed.setInformativeText("File for conversion of tauP --> intP is required!")
                calibration_para_failed.setWindowTitle("Error!")
                calibration_para_failed.exec_()
                return
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("File for conversion of tauP --> intP is required!")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # -----------------------------------------------------------------------------------------------
        # Collect additional input parameters
        [self.f1, self.f2, self.error_assumed, self.int_fluoro_max, self.int_phosphor_max] = self.input_collecting()
        self.Ksv1_O2 = np.float64(self.Ksv1_O2_tab5_edit.text().replace(',', '.'))
        self.prop_ksv_O2 = np.float64(self.Ksv2_O2_tab5_edit.text().replace(',', '.'))
        self.curv_O2 = np.float64(self.curv_O2_tab5_edit.text().replace(',', '.'))
        self.Ksv1_CO2 = np.float64(self.Ksv1_CO2_tab5_edit.text().replace(',', '.'))
        self.prop_ksv_CO2 = np.float64(self.Ksv2_CO2_tab5_edit.text().replace(',', '.'))
        self.curv_CO2 = np.float64(self.curv_CO2_tab5_edit.text().replace(',', '.'))
        tau_phosphor_c0 = np.float64(self.lifetime_phos_dualsens_edit.text().replace(',', '.'))
        pO2_range = np.linspace(0, 300, num=int(300/0.5)+1)
        pCO2_range = np.linspace(0, 100, num=int(100/0.01)+1)
        self.conv_tau_int = pd.read_csv(self.conv_file, sep='\t', index_col=0)
        print('current tab', self.tabs.currentIndex()+1, ' - (C)O2 dualsensor')

        # set progress status
        self.progress_tab5.setValue(10)

# -------------------------------------------------------------------------------------------------------------
# if simulation is True
# -------------------------------------------------------------------------------------------------------------
        if self.simulation_checkbox.checkState() == 2:
            # clear dPhi from calibration/input table (pCO2 in row2, pO2 in row3)
            self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                                   cols=[0, 1, 4, 5, 6])
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[0, 1, 4, 5, 6])

            # Measurement points pCO2
            try:
                self.tableINPUT.item(0, 2).text()
                if np.float64(self.tableINPUT.item(0, 2).text().replace(',', '.')) > 100. or \
                                np.float64(self.tableINPUT.item(0, 2).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input pCO2 between 0 - 100 hPa.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("CO2 and O2 content have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            # Measurement points pO2
            try:
                self.tableINPUT.item(0, 3).text()
                if np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')) > 250. or \
                                np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')) < 0.:
                    range_exceeded = QMessageBox()
                    range_exceeded.setIcon(QMessageBox.Information)
                    range_exceeded.setText("Measurement range exceeded!")
                    range_exceeded.setInformativeText("Choose input pO2 content between 0 - 250 hPa.")
                    range_exceeded.setWindowTitle("Error!")
                    range_exceeded.exec_()
                    return
            except:
                measurement_para_failed = QMessageBox()
                measurement_para_failed.setIcon(QMessageBox.Information)
                measurement_para_failed.setText("Insufficient measurement parameters!")
                measurement_para_failed.setInformativeText("CO2 and O2 content have to be defined for the simulation.")
                measurement_para_failed.setWindowTitle("Error!")
                measurement_para_failed.exec_()
                return

            self.pCO2_list.append(np.float64(self.tableINPUT.item(0, 2).text().replace(',', '.')))
            self.pO2_list.append(np.float64(self.tableINPUT.item(0, 3).text().replace(',', '.')))


            # set progress status
            self.progress_tab5.setValue(15)

        # ------------------------------------------------------------------------------
            # Input parameter simulation
            [self.Phi_f1_deg, self.Phi_f2_deg, self.Phi_f1_deg_er, self.Phi_f2_deg_er, self.int_ratio,
             self.para_simulation] = cox.simulate_phaseangle_pO2_pCO2(pO2_range=pO2_range, pO2_calib=self.pO2_list[:2],
                                                                      pO2_meas=self.pO2_list[2], curv_O2=self.curv_O2,
                                                                      prop_ksv_O2=self.prop_ksv_O2,
                                                                      K_sv1_O2=self.Ksv1_O2, tauP_c0=tau_phosphor_c0,
                                                                      intP_max=self.int_phosphor_max, f2=self.f2,
                                                                      intF_max=self.int_fluoro_max, f1=self.f1,
                                                                      df_conv_tau_int_O2=self.conv_tau_int,
                                                                      pCO2_range=pCO2_range, curv_CO2=self.curv_CO2,
                                                                      pCO2_calib=self.pCO2_list[:2], plotting=False,
                                                                      pCO2_meas=self.pCO2_list[2], decimal=4,
                                                                      prop_ksv_CO2=self.prop_ksv_CO2,
                                                                      normalize=False, K_sv1_CO2=self.Ksv1_CO2,
                                                                      fontsize_=13, er_phase=self.error_assumed)

            # set progress status
            self.progress_tab5.setValue(25)

            # ----------------------------------------------
            # write input parameter to tableCalibration
            for i in range(self.tableCalibration.rowCount()-2):
                if pCO2_order[i] == self.pCO2_list[0]:
                    if pO2_order[i] == self.pO2_list[0]:
                        phi_f1 = self.Phi_f1_deg['fluoro0, phosphor0']
                        phi_f2 = self.Phi_f2_deg['fluoro0, phosphor0']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                    else:
                        phi_f1 = self.Phi_f1_deg['fluoro0, phosphor1']
                        phi_f2 = self.Phi_f2_deg['fluoro0, phosphor1']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                else:
                    if pO2_order[i] == self.pO2_list[0]:
                        phi_f1 = self.Phi_f1_deg['fluoro1, phosphor0']
                        phi_f2 = self.Phi_f2_deg['fluoro1, phosphor0']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                    else:
                        phi_f1 = self.Phi_f1_deg['fluoro1, phosphor1']
                        phi_f2 = self.Phi_f2_deg['fluoro1, phosphor1']
                        if isinstance(phi_f1, np.float):
                            pass
                        else:
                            phi_f1 = phi_f1.values[0]
                            phi_f2 = phi_f2.values[0]
                            if isinstance(phi_f1, np.float):
                                pass
                            else:
                                phi_f1 = phi_f1[0]
                                phi_f2 = phi_f2[0]
                self.tableCalibration.setItem(i, 4, QTableWidgetItem('{:.2f}'.format(phi_f1)))
                self.tableCalibration.setItem(i, 5, QTableWidgetItem('{:.2f}'.format(phi_f2)))

            if isinstance(self.Phi_f1_deg['meas'], np.float):
                p_meas_f1 = self.Phi_f1_deg['meas']
                p_meas_f2 = self.Phi_f2_deg['meas']
            elif isinstance(self.Phi_f1_deg['meas'], pd.DataFrame):
                p_meas_f1 = self.Phi_f1_deg['meas'].values[0][0]
                p_meas_f2 = self.Phi_f2_deg['meas'].values[0][0]
            else:
                p_meas_f1 = self.Phi_f1_deg['meas'].values[1]
                p_meas_f2 = self.Phi_f2_deg['meas'].values[1]
                if isinstance(p_meas_f1, np.float):
                    pass
                else:
                    p_meas_f1 = p_meas_f1[0]
                    p_meas_f2 = p_meas_f2[0]

            self.tableINPUT.setItem(0, 4, QTableWidgetItem('{:.2f}'.format(p_meas_f1)))
            self.tableINPUT.setItem(0, 5, QTableWidgetItem('{:.2f}'.format(p_meas_f2)))

            # set progress status
            self.progress_tab5.setValue(30)

        # ------------------------------------------------------------------------------
            # Plotting calibration
            pCO2_calib = self.pCO2_list[:-1]
            pO2_calib = self.pO2_list[:-1]

            self.plot_calibration_CO2_O2(pCO2_calib=pCO2_calib, pO2_calib=pO2_calib, KsvO2=self.Ksv1_O2,
                                         KsvCO2=self.Ksv1_CO2, Ksv2_O2=self.prop_ksv_O2*self.Ksv1_O2,
                                         Ksv2_CO2=self.prop_ksv_CO2*self.Ksv1_CO2, int_f=self.para_simulation['intF'],
                                         tau_quot=self.para_simulation['tau_quot'], f1=self.fig_CO2calib,
                                         ax1=self.ax_CO2calib, f2=self.fig_pO2calib2, ax2=self.ax_pO2calib_2)

            # set progress status
            self.progress_tab5.setValue(45)

        # ------------------------------------------------------------------------------
            # Dual sensing according to simulated input parameter
            self.Phi_f1_meas_er = [self.Phi_f1_deg['meas'] - self.error_assumed, self.Phi_f1_deg['meas'],
                                   self.Phi_f1_deg['meas'] + self.error_assumed]
            self.Phi_f2_meas_er = [self.Phi_f2_deg['meas'] - self.error_assumed, self.Phi_f2_deg['meas'],
                                   self.Phi_f2_deg['meas'] + self.error_assumed]

            [self.pO2_calc, self.pCO2_calc, self.tau_quot, self.tauP, self.paraO2, self.intP, self.intF, self.paraCO2,
             ax_pO2, ax_pCO2, self.ampl_total_f1,
             self.ampl_total_f2] = cox.CO2_oxygen_dualsensor(phi_f1_deg=self.Phi_f1_deg, phi_f2_deg=self.Phi_f2_deg,
                                                             phi_f1_meas=self.Phi_f1_deg['meas'], pO2_calib=pO2_calib,
                                                             phi_f2_meas=self.Phi_f2_deg['meas'], pO2_range=pO2_range,
                                                             er_phase=self.error_assumed, f1=self.f1, f2=self.f2,
                                                             curv_O2=self.curv_O2, prop_ksv_O2=self.prop_ksv_O2,
                                                             method_='std', df_conv_tau_intP=self.df_conv_tauP_intP,
                                                             pCO2_range=pCO2_range, intP_max=self.int_phosphor_max,
                                                             pCO2_calib=pCO2_calib, plotting=True,
                                                             curv_CO2=self.curv_CO2, prop_ksv_CO2=self.prop_ksv_CO2,
                                                             intF_max=self.int_fluoro_max)

            # set progress status
            self.progress_tab5.setValue(75)

        # ---------------------------------------
            # Plotting results
            if self.tauP.loc[0, 1] < 1:
                tauP_mikros = self.tauP*1E6
            else:
                tauP_mikros = self.tauP

            self.plot_results_CO2_O2(pCO2=self.pCO2_calc, pCO2_calib=pCO2_calib, pO2_calc=self.pO2_calc,
                                     pO2_calib=pO2_calib, tauP=tauP_mikros, intF=self.intF, f1=self.fig_CO2,
                                     ax1=self.ax_CO2, f2=self.fig_pO2_2, ax2=self.ax_pO2_2)

            # set progress status
            self.progress_tab5.setValue(90)

        # --------------------------------------------------
            # Output - total amplitude reported in calibration table
            if self.int_ratio_checkbox.isChecked() is True:
                for i in range(self.tableCalibration.rowCount()-2):
                    if pCO2_order[i] == self.pCO2_list[0]:
                        if pO2_order[i] == self.pO2_list[0]:
                            int_r = self.int_ratio['fluoro0, phosphor0']
                        else:
                            int_r = self.int_ratio['fluoro0, phosphor1']
                    else:
                        if pO2_order[i] == self.pO2_list[0]:
                            int_r = self.int_ratio['fluoro1, phosphor0']
                        else:
                            int_r = self.int_ratio['fluoro1, phosphor1']
                    self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.3f}'.format(int_r)))
                self.tableINPUT.setItem(0, 6, QTableWidgetItem('{:.3f}'.format(self.int_ratio['meas'])))
            else:
                self.int_ratio = None
                for i in range(self.tableCalibration.rowCount()-2):
                    if pCO2_order[i] == self.pCO2_list[0]:
                        if pO2_order[i] == self.pO2_list[0]:
                            total_ampl1 = self.ampl_total_f1['fluoro0, phosphor0']
                            total_ampl2 = self.ampl_total_f2['fluoro0, phosphor0']
                        else:
                            total_ampl1 = self.ampl_total_f1['fluoro0, phosphor1']
                            total_ampl2 = self.ampl_total_f2['fluoro0, phosphor1']
                    else:
                        if pO2_order[i] == self.pO2_list[0]:
                            total_ampl1 = self.ampl_total_f1['fluoro1, phosphor0']
                            total_ampl2 = self.ampl_total_f2['fluoro1, phosphor0']
                        else:
                            total_ampl1 = self.ampl_total_f1['fluoro1, phosphor1']
                            total_ampl2 = self.ampl_total_f2['fluoro1, phosphor1']

                    self.tableCalibration.setItem(i, 6, QTableWidgetItem('{:.2f}'.format(total_ampl1[1])))
                    self.tableCalibration.setItem(i, 7, QTableWidgetItem('{:.2f}'.format(total_ampl2[1])))

                self.tableINPUT.setItem(0, 6, QTableWidgetItem('{:.2f}'.format(self.ampl_total_f1['meas'][1])))
                self.tableINPUT.setItem(0, 7, QTableWidgetItem('{:.2f}'.format(self.ampl_total_f2['meas'][1])))

            # Results reported in message box
            self.message_tab5.append('Calculated pO2 = {:.2f}± {:.2e} hPa'.format(self.pO2_calc[1],
                                                                                  self.pO2_calc.std()))
            self.message_tab5.append('Calculated pCO2 = {:.2f}  ± {:.2e} hPa'.format(self.pCO2_calc[1],
                                                                                     self.pCO2_calc.std()))
            # set progress status
            self.progress_tab5.setValue(100)

# -------------------------------------------------------------------------------------------------------------
# if simulation is False -> measurement evaluation
# -------------------------------------------------------------------------------------------------------------
        else:
            self.message_tab5.setText('Measurement evaluation')

            # clear p(C)O2 from input table and I_ratio
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[0, 1, 2, 3])

            if self.int_ratio_checkbox.isChecked():
                cols_required = [4, 5, 6]
            else:
                cols_required = [4, 5, 6, 7]

            # superimposed phase angles and intensity ratio required (2-Point-Calibration)
            for i in range(self.tableCalibration.rowCount()-2):
                for j in cols_required:
                    try:
                        self.tableCalibration.item(i, j).text()
                    except:
                        calibration_para_failed = QMessageBox()
                        calibration_para_failed.setIcon(QMessageBox.Information)
                        calibration_para_failed.setText("Insufficient calibration parameters!")
                        calibration_para_failed.setInformativeText("For measurement evaluation the superimposed phase "
                                                                   "angles and the intensity ratios are required.")
                        calibration_para_failed.setWindowTitle("Error!")
                        calibration_para_failed.exec_()
                        return

                if self.tableCalibration.item(i, j).text() == '':
                    calibration_para_failed = QMessageBox()
                    calibration_para_failed.setIcon(QMessageBox.Information)
                    calibration_para_failed.setText("Insufficient calibration parameters!")
                    calibration_para_failed.setInformativeText("For measurement evaluation the superimposed phase "
                                                               "angles and either the intensity ratios or the total "
                                                               "amplitudes at two modulation frequencies are required.")
                    calibration_para_failed.setWindowTitle("Error!")
                    calibration_para_failed.exec_()
                    return

            # measurement point
            for j in cols_required:
                try:
                    self.tableINPUT.item(0, j).text()
                except:
                    measurement_failed = QMessageBox()
                    measurement_failed.setIcon(QMessageBox.Information)
                    measurement_failed.setText("Insufficient measurement parameters!")
                    measurement_failed.setInformativeText("For measurement evaluation the superimposed phase angles "
                                                          "and either the intensity ratios or the total amplitudes at "
                                                          "two modulation frequencies are required.")
                    measurement_failed.setWindowTitle("Error!")
                    measurement_failed.exec_()
                    return

            # -------------------------------------------------------------------------------------
            # all phase angles including error
            self.phi_meas_f1 = []
            self.phi_meas_f2 = []
            self.Phi_meas_f1_er = {}
            self.Phi_meas_f2_er = {}
            self.ampl_total_f1_meas = []
            self.ampl_total_f2_meas = []
            self.int_ratio_meas = []

            # check the number of measurement points
            list_meas = []
            for i in range(self.tableINPUT.rowCount()):
                it = self.tableINPUT.item(i, 4)
                if it and it.text():
                    list_meas.append(it.text())

            for i in range(len(list_meas)):
                # phase angles for both modulation frequencies including assumed error
                dphi1 = np.float64(self.tableINPUT.item(i, 4).text().replace(',', '.'))
                dphi2 = np.float64(self.tableINPUT.item(i, 5).text().replace(',', '.'))
                self.phi_meas_f1.append(dphi1)
                self.phi_meas_f2.append(dphi2)
                self.Phi_meas_f1_er['meas {}'.format(i)] = [dphi1 - self.error_assumed, dphi1,
                                                            dphi1 + self.error_assumed]
                self.Phi_meas_f2_er['meas {}'.format(i)] = [dphi2 - self.error_assumed, dphi2,
                                                            dphi2 + self.error_assumed]

            for i in range(len(list_meas)):
                if self.int_ratio_checkbox.isChecked() is True:
                    self.int_ratio_meas.append(np.float64(self.tableINPUT.item(i, 6).text().replace(',', '.')))
                else:
                    self.ampl_total_f1_meas.append(np.float64(self.tableINPUT.item(i, 6).text().replace(',', '.')))
                    self.ampl_total_f2_meas.append(np.float64(self.tableINPUT.item(i, 7).text().replace(',', '.')))

            # extract measured phase angles and intensity ratio with calibration points
            f_ind = ['']*len(calib_points)
            ampl_ind = ['']*len(calib_points)
            for i in range(len(calib_points)):
                if calib_points[i][0] == self.pCO2_list[0]:
                    f_ind[i] = f_ind[i] + 'fluoro0, '
                    ampl_ind[i] = ampl_ind[i] + 'fluoro0, '
                else:
                    f_ind[i] = f_ind[i] + 'fluoro1, '
                    ampl_ind[i] = ampl_ind[i] + 'fluoro1, '
                if calib_points[i][1] == self.pO2_list[0]:
                    f_ind[i] = f_ind[i] + 'phosphor0'
                    ampl_ind[i] = ampl_ind[i] + 'phosphor0'
                else:
                    f_ind[i] = f_ind[i] + 'phosphor1'
                    ampl_ind[i] = ampl_ind[i] + 'phosphor1'

            list_phi_f1 = []
            list_phi_f2 = []
            list_int_ratio = []
            list_ampl1 = []
            list_ampl2 = []

            for i in range(self.tableCalibration.rowCount()-2):
                list_phi_f1.append(np.float64(self.tableCalibration.item(i, 4).text().replace(',', '.')))
                list_phi_f2.append(np.float64(self.tableCalibration.item(i, 5).text().replace(',', '.')))
                if self.int_ratio_checkbox.isChecked() is True:
                    list_int_ratio.append(np.float64(self.tableCalibration.item(i, 6).text().replace(',', '.')))
                else:
                    list_ampl1.append(np.float64(self.tableCalibration.item(i, 6).text().replace(',', '.')))
                    list_ampl2.append(np.float64(self.tableCalibration.item(i, 7).text().replace(',', '.')))

            ls_meas_index = []
            for num in range(len(list_meas)):
                ls_meas_index.append('meas {}'.format(num))
            ampl_ind = ampl_ind + ls_meas_index
            if self.int_ratio_checkbox.isChecked() is True:
                list_int_ratio = list_int_ratio + self.int_ratio_meas
                self.int_ratio = pd.Series(list_int_ratio, index=ampl_ind)
                self.ampl_total_f1 = None
                self.ampl_total_f2 = None
            else:
                # ampl_ind.append('meas')
                list_ampl1 = list_ampl1 + self.ampl_total_f1_meas
                list_ampl2 = list_ampl2 + self.ampl_total_f2_meas
                self.ampl_total_f1 = pd.Series(list_ampl1, index=ampl_ind)
                self.ampl_total_f2 = pd.Series(list_ampl2, index=ampl_ind)
                self.int_ratio = None

            self.Phi_f1_deg = pd.Series(list_phi_f1, index=f_ind)
            self.Phi_f2_deg = pd.Series(list_phi_f2, index=f_ind)

            # ----------------------------------------------
            # Include error in measured phase angle
            keys = self.Phi_f1_deg.keys().tolist()
            dPhi_f1_er = pd.Series({key: None for key in keys})
            dPhi_f2_er = pd.Series({key: None for key in keys})
            for i in self.Phi_f1_deg.keys():
                dPhi_f1_er[i] = ([self.Phi_f1_deg[i] - self.error_assumed, self.Phi_f1_deg[i], self.Phi_f1_deg[i] +
                                  self.error_assumed])
                dPhi_f2_er[i] = ([self.Phi_f2_deg[i] - self.error_assumed, self.Phi_f2_deg[i], self.Phi_f2_deg[i] +
                                  self.error_assumed])

            # all phase angles including error
            self.dPhi_meas_f1_er = pd.concat([dPhi_f1_er, pd.Series(self.Phi_meas_f1_er)], axis=0)
            self.dPhi_meas_f2_er = pd.concat([dPhi_f2_er, pd.Series(self.Phi_meas_f2_er)], axis=0)

            # set progress status
            self.complete_tab5 = 25
            self.progress_tab5.setValue(self.complete_tab5)

        # ---------------------------------------------------------------------------------------------------------
            # Ksv or Tau optimization
            if self.calib_tauP_checkbox.isChecked() is True:
                self.message_tab5.append('tauP optimization \n')
                tau_c0 = None
                ksv_O2 = self.Ksv1_O2
            elif self.calib_ksv_checkbox.isChecked() is True:
                self.message_tab5.append('ksv optimization \n')
                if tau_phosphor_c0 > 1:
                    tau_c0 = tau_phosphor_c0*1E-6
                else:
                    tau_c0 = tau_phosphor_c0
                ksv_O2 = None
            else:
                tau_c0 = None
                ksv_O2 = None

            self.pO2_calc = {}
            self.pCO2_calc = {}
            self.paraO2 = {}
            self.paraCO2 = {}
            self.tauP = {}
            self.tau_quot = {}
            self.intF = {}
            for i in range(len(list_meas)):
                # Dual sensing according to measured input parameter
                [pO2_calc, pCO2_calc, tau_quot, tauP, paraO2, intP, intF, paraCO2, ax_pO2,
                 ax_pCO2] = cox.CO2_O2_hybrid_2cp_meas(pO2_range=pO2_range, pO2_calib=self.pO2_list,
                                                       curv_O2=self.curv_O2, prop_ksv_O2=self.prop_ksv_O2,
                                                       phi_f1_c1=self.Phi_f1_deg, phi_f1_meas=self.phi_meas_f1[i],
                                                       phi_f2_c1=self.Phi_f2_deg, phi_f2_meas=self.phi_meas_f2[i],
                                                       er_phase=self.error_assumed, pCO2_range=pCO2_range,
                                                       df_conv_tau_intP=self.df_conv_tauP_intP, f1=self.f1, f2=self.f2,
                                                       intP_max=self.int_phosphor_max, pCO2_calib=self.pCO2_list,
                                                       intF_max=self.int_fluoro_max, int_ratio=self.int_ratio,
                                                       curv_CO2=self.curv_CO2, ampl_total_f2=self.ampl_total_f2,
                                                       ampl_total_f1=self.ampl_total_f1, prop_ksv_CO2=self.prop_ksv_CO2,
                                                       ksv_O2=ksv_O2, tau_c0=tau_c0, plotting=False, fontsize_=13,
                                                       calibration='1point')
                self.complete_tab5 += 0.05
                self.progress_tab5.setValue(self.complete_tab5)

                self.pO2_calc[i] = pO2_calc
                self.pCO2_calc[i] = pCO2_calc
                self.paraO2[i] = paraO2
                self.paraCO2[i] = paraCO2
                self.tauP[i] = tauP
                self.tau_quot[i] = tau_quot
                self.intF[i] = intF

            if isinstance(self.paraO2[0]['Ksv_fit1'], np.float):
                ksv_fit1 = self.paraO2[0]['Ksv_fit1']
                ksv_fit2 = self.paraO2[0]['Ksv_fit2']
            else:
                ksv_fit1 = self.paraO2[0]['Ksv_fit1'][1]
                ksv_fit2 = self.paraO2[0]['Ksv_fit2'][1]

            # return fitting parameter in message box
            self.message_tab5.append('Fitting parameter O2 - Ksv1: {:.2e} \t Ksv2: {:.2e} \t m: '
                                     '{:.1f}'.format(ksv_fit1, ksv_fit2, self.paraO2[0]['slope']))

            if isinstance(self.paraCO2[0]['Ksv_fit1'], np.float):
                ksv_fit1 = self.paraCO2[0]['Ksv_fit1']
                ksv_fit2 = self.paraCO2[0]['Ksv_fit2']
            else:
                ksv_fit1 = self.paraCO2[0]['Ksv_fit1'][1]
                ksv_fit2 = self.paraCO2[0]['Ksv_fit2'][1]
            self.message_tab5.append('Fitting parameter CO2 - Ksv1: {:.2e} \t Ksv2: {:.2e} \t m: {:.1f} '
                                     '\n'.format(ksv_fit1, ksv_fit2, self.paraCO2[0]['slope']))

            # set progress status
            self.progress_tab5.setValue(70)

        # ----------------------------------------------------------------------
            # Plotting calibration - according to first measurement point
            self.plot_calibration_CO2_O2_measurement(pCO2_calib=self.pCO2_list, pO2_calib=self.pO2_list,
                                                     KsvO2=self.Ksv1_O2, KsvCO2=self.Ksv1_CO2, f1=self.fig_CO2calib,
                                                     int_f=self.intF[0], ax1=self.ax_CO2calib, f2=self.fig_pO2calib2,
                                                     tau_quot=self.tau_quot[0], ax2=self.ax_pO2calib_2)

            # set progress status
            self.progress_tab5.setValue(85)
        # ---------------------------------------
            # Plotting results
            tauP_mikros = {}
            self.pO2_calc_corr = {}
            self.pCO2_calc_corr = {}
            for i in range(len(self.tauP)):
                if self.tauP[i].loc[0, 1] < 1.:
                    tauP_mikros[i] = self.tauP[i] * 1E6
                else:
                    tauP_mikros[i] = self.tauP[i]

                self.pCO2_calc_corr[i] = np.array([x for x in self.pCO2_calc[i] if np.isnan(x) == False])
                self.pO2_calc_corr[i] = np.array([x for x in self.pO2_calc[i] if np.isnan(x) == False])

            # first measurement point
            self.plot_results_CO2_O2_meas(pCO2=self.pCO2_calc_corr[0], pCO2_calib=self.pCO2_list, tauP=tauP_mikros[0],
                                          pO2_calc=self.pO2_calc_corr[0], intF=self.intF[0], pO2_calib=self.pO2_list,
                                          f1=self.fig_CO2, ax1=self.ax_CO2, f2=self.fig_pO2_2, ax2=self.ax_pO2_2)
            # last measurement point
            if len(list_meas) > 1:
                last = len(list_meas) - 1
                self.plot_results_CO2_O2_meas(pCO2=self.pCO2_calc_corr[last], pCO2_calib=self.pCO2_list,
                                              tauP=tauP_mikros[last], pO2_calc=self.pO2_calc_corr[last],
                                              intF=self.intF[last], pO2_calib=self.pO2_list, f1=self.fig_CO2,
                                              ax1=self.ax_CO2, f2=self.fig_pO2_2, ax2=self.ax_pO2_2)

                # set progress status
                self.progress_tab5.setValue(95)
        # --------------------------------------------------
            # Results reported in message box
            for i in range(len(list_meas)):
                self.message_tab5.append('# Point - {}'.format(i + 1))
                self.message_tab5.append('\t Calculated pO2 = {:.2f}± {:.2f} hPa'.format(self.pO2_calc_corr[i].mean(),
                                                                                         self.pO2_calc_corr[i].std()))

                self.message_tab5.append('\t Calculated pCO2 = {:.2f} ± {:.2e} hPa'.format(self.pCO2_calc_corr[i].mean(),
                                                                                           self.pCO2_calc_corr[i].std()))

        print('pCO2 / O2 dualsensing finished')
        # set progress status
        self.progress_tab5.setValue(100)

        self.run_tab5_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                           "stop: 0 #227286, stop: 1 #54bad4); border-width: 1px; border-color: #077487;"
                                           " border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                           "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")
        print('#-------------------------------------------------------------------')


# ---------------------------------------------------------------------------------------------------------------------
# O2 / T sensing
# ---------------------------------------------------------------------------------------------------------------------
    def open_calibration(self):
        self.load_O_T_calib_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px;"
                                                 " border-color: #077487; border-style: solid; border-radius: 7; "
                                                 "padding: 5px; font-size: 10px; padding-left: 1px; padding-right: 5px;"
                                                 " min-height: 10px; max-height: 18px;")

        path_gui_ = os.path.abspath("GUI_dualsensors.py")
        path_gui = path_gui_.split('GUI_dualsensors')[0]
        self.fname_calib = QFileDialog.getOpenFileName(self, "Select specific txt file for dual sensing calibration "
                                                             "T/O2", path_gui)[0]
        self.load_O_T_calib_button.setStyleSheet("color: white; background-color:"
                                                 "QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a9eb7, "
                                                 "stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                                 "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px;"
                                                 "padding-left: 1px; padding-right: 5px; min-height: 10px;"
                                                 "max-height: 18px;")

        if not self.fname_calib:
            return
        # next function -> load file
        self.read_calibration(self.fname_calib)

    def read_calibration(self, fname_calib):
        self.calib_edit.clear()
        try:
            self.calib_file = fname_calib
        except:
            conv_file_load_failed = QMessageBox()
            conv_file_load_failed.setIcon(QMessageBox.Information)
            conv_file_load_failed.setText('Invalid file for calibration of the dual-sensor T/O2!')
            conv_file_load_failed.setInformativeText('Choose another file from path...')
            conv_file_load_failed.setWindowTitle('Error!')
            conv_file_load_failed.buttonClicked.connect(self.open_calibration)
            conv_file_load_failed.exec_()
            return

        # ---------------------------------------------------------------------
        # write (part of the path) to text line
        parts = fname_calib.split('/')
        if 3 < len(parts) <= 6:
            self.calib_file_part = '/'.join(parts[3:])
        elif len(parts) > 6:
            self.calib_file_part = '/'.join(parts[5:])
        else:
            self.calib_file_part = '/'.join(parts)
        self.calib_edit.insertPlainText(str(self.calib_file_part))

        # ---------------------------------------------------------------------
        # load txt file
        self.df_calib = pd.read_csv(self.calib_file, sep='\t', decimal='.', index_col=0, header=None,
                                    encoding='latin-1')

# ---------------------------------------------------------------------------------------------------------------------
    # dual sensor calculations 6th tab
    def O2_temp_sensing(self):
        print('#--------------------------------------')
        self.run_tab6_button.setStyleSheet("color: white; background-color: #2b5977; border-width: 1px; "
                                           "border-color: #077487; border-style: solid; border-radius: 7; padding: 5px; "
                                           "font-size: 10px; padding-left: 1px; padding-right: 5px; min-height: 10px; "
                                           "max-height: 18px;")
        # status of progressbar
        self.progress_tab6.setValue(0)
        print('O2 / T dualsensing')

        # clear everything
        self.message_tab6.clear()

        # calibration
        # clear - CALIBRATION plane - lifetime as it is comparable with the measurement results)
        self.ax_tau_tab6.cla()
        self.fig_tau_tab6.clear()
        if self.plot_3d_checkbox.isChecked() is True:
            self.ax_tau_tab6 = self.fig_tau_tab6.gca(projection='3d')
            self.ax_tau_tab6.set_xlim(0, 50)
            self.ax_tau_tab6.set_ylim(0, 20)
            self.ax_tau_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6.set_zlabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_tau_tab6.canvas.draw()
        else:
            self.ax_tau_tab6_o2 = self.fig_tau_tab6.add_subplot(212)
            self.ax_tau_tab6_temp = self.fig_tau_tab6.add_subplot(211)
            self.ax_tau_tab6_o2.set_xlim(0, 20)
            self.ax_tau_tab6_temp.set_xlim(0, 50)
            self.ax_tau_tab6_o2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6_temp.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6_o2.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_temp.set_ylabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_tau_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_tau_tab6.canvas.draw()

        # clear - CALIBRATION plane - intensity ratio (I prompt fluorescence vs delayed fluorescence as it is comparable
        # with the measurement results)
        self.ax_int_tab6.cla()
        self.fig_int_tab6.clear()
        if self.plot_3d_checkbox.isChecked() is True:
            self.ax_int_tab6 = self.fig_int_tab6.gca(projection='3d')
            self.ax_int_tab6.set_xlim(0, 100)
            self.ax_int_tab6.set_ylim(0, 100)
            self.ax_int_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6.set_zlabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_int_tab6.canvas.draw()
        else:
            self.ax_int_tab6_o2 = self.fig_int_tab6.add_subplot(212)
            self.ax_int_tab6_temp = self.fig_int_tab6.add_subplot(211)
            self.ax_int_tab6_o2.set_xlim(0, 20)
            self.ax_int_tab6_temp.set_xlim(0, 50)
            self.ax_int_tab6_o2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6_temp.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6_o2.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_temp.set_ylabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6_o2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.ax_int_tab6_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_int_tab6.canvas.draw()

        # clear - intersection of planes
        self.ax_pO2T_tab6.cla()
        self.fig_pO2T_tab6.clear()
        self.ax_pO2T_tab6 = self.fig_pO2T_tab6.gca()
        self.ax_pO2T_tab6.set_xlim(0, 100)
        self.ax_pO2T_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
        self.ax_pO2T_tab6.set_xlabel('Temperature [°C]', fontsize=9)
        self.ax_pO2T_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.fig_pO2T_tab6.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
        self.fig_pO2T_tab6.canvas.draw()

        # status of progressbar
        self.progress_tab6.setValue(0.5)
    # --------------------------------------------------------------------------------------------------------
        # Check all entries for calibration
        # calibration file
        try:
            self.calib_file
            if self.calib_file is None:
                calibration_para_failed = QMessageBox()
                calibration_para_failed.setIcon(QMessageBox.Information)
                calibration_para_failed.setText("Error for calibration parameters!")
                calibration_para_failed.setInformativeText("Calibration file is required!")
                calibration_para_failed.setWindowTitle("Error!")
                calibration_para_failed.exec_()
                return
        except:
            calibration_para_failed = QMessageBox()
            calibration_para_failed.setIcon(QMessageBox.Information)
            calibration_para_failed.setText("Error for calibration parameters!")
            calibration_para_failed.setInformativeText("Calibration file is required!")
            calibration_para_failed.setWindowTitle("Error!")
            calibration_para_failed.exec_()
            return

        # Collect additional input parameters
        [self.f1, self.f2, self.error_assumed, self.int_fluoro_max, self.int_phosphor_max] = self.input_collecting()
        # pO2 range
        pO2_str = str(self.pO2_range_tab6_edit.text())
        ox_ls = [np.float64(i) for i in pO2_str.split(',')]
        self.pO2_range = np.linspace(start=ox_ls[0], stop=ox_ls[1], num=int((ox_ls[1] - ox_ls[0]) / ox_ls[2] + 1))

        # T range # start, stop, step
        temp_str = str(self.temp_range_tab6_edit.text())
        temp_ls = [np.float64(i) for i in temp_str.split(',')]
        if len(temp_ls) > 3:
            # re-combine the last two values
            temp_ls = [temp_ls[0], temp_ls[1], int(temp_ls[-2])+int(temp_ls[-1])/10]
        temp_ls = sorted(temp_ls[:2]) + [temp_ls[-1]]
        self.temp_range = np.linspace(start=temp_ls[0], stop=temp_ls[1], num=int((temp_ls[1]-temp_ls[0])/temp_ls[2]+1))

        # re-check if calibration file coincide with to_fit parameter
        calib_para = self.calib_file.split('/')[-1].split('_')[-2].replace('-', ' / ')
        self.message_tab6.setText('Intensity is fitted as I/I0')
        self.message_tab6.append('Lifetime is fitted as tauP / tau0')

        # progressbar
        # status of progressbar
        self.progress_tab6.setValue(10)

        # -------------------------------------------------------------------------------------------------------------
        # if simulation is True
        # -------------------------------------------------------------------------------------------------------------
        if self.simulation_checkbox.checkState() == 2:
            print('simulation O2/T')

            [tau_calib, tau_tau0_calib, int_calib, int_invert_calib, fig_raw, fig_calib, ax_tau_calib, ax_int_calib,
             temp_reg, pO2_reg,
             para_fit] = Tox.fitting_tsm_for_T_O2_(file=self.calib_file, temp_reg=self.temp_range, pO2_reg=self.pO2_range,
                                                  fitpara1='tau/tau0', fitpara2='I/I0', calib_range_temp=None,
                                                  calib_range_O2=None, plot_calib_data=False)
            self.progress_tab6.setValue(50)

            # measurement data (including their error)
            calib_data = Tox.load_calibration_data(p=self.calib_file)
            calib_range_temp = [calib_data['tauP'].columns[0], calib_data['tauP'].columns[-1]]
            calib_range_O2 = [calib_data['tauP'].index[0], calib_data['tauP'].index[-1]]
            tau_calib_s = calib_data['tauP'].loc[calib_range_O2[0]:calib_range_O2[1],
                          calib_range_temp[0]:calib_range_temp[1]] * 1e-3
            int_calib_s = calib_data['I-ratio'].loc[calib_range_O2[0]:calib_range_O2[1],
                          calib_range_temp[0]:calib_range_temp[1]] * 1e-3

            if 'error tau' in calib_data.keys():
                tau_calib_er = calib_data['error tau'].loc[calib_range_O2[0]:calib_range_O2[1],
                               calib_range_temp[0]:calib_range_temp[1]] * 1e-3
                int_calib_er = calib_data['error I-ratio'].loc[calib_range_O2[0]:calib_range_O2[1],
                               calib_range_temp[0]:calib_range_temp[1]]
            else:
                tau_calib_er = None
                int_calib_er = None

        # -----------------------------------------------------------------------------------------------------------
            # plotting 3D planes for lifetime and intensity ratio according to fitted parameters
            temp_X, pO2_Y = np.meshgrid(tau_calib.columns, tau_calib.index)

            if self.plot_3d_checkbox.isChecked() is True:
                self.ax_tau_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=tau_calib*1e3, rstride=1, cstride=1, color='navy',
                                              alpha=.9, linewidth=0., antialiased=True)
                self.ax_tau_tab6.tick_params(which='both', labelsize=8)
                self.ax_tau_tab6.set_ylim(0, pO2_Y.max()*1.05)
                self.ax_tau_tab6.set_xlim(temp_X.max()*1.05, 0)
            else:
                temp_example = [np.float64(t) for t in str(self.temp_example_tab6_edit.text()).split(',')]
                ox_example = [np.float64(o) for o in str(self.ox_example_tab6_edit.text()).split(',')]
                col_tau = ['navy', 'grey']
                ls_ = ['-.', '--', ':']

                # ---------------------------------------------------
                # Oxygen dependence for different temperatures
                for en, t_ in enumerate(temp_example):
                    if en < len(col_tau):
                        ls_ist = ls_[0]
                    elif en >= len(col_tau):
                        if en >= 2 * len(col_tau):
                            ls_ist = ls_[2]
                        else:
                            ls_ist = ls_[1]

                    # find closest value in list
                    t_close = min(tau_calib.columns, key=lambda x: abs(x - t_))
                    t_close_ = min(tau_calib_s.columns, key=lambda x: abs(x - t_))

                    # fitting curve based on measurement data
                    self.ax_tau_tab6_o2.plot(tau_calib[t_close] * 1e3, color=col_tau[en], alpha=.9, linewidth=1.25,
                                             ls=ls_ist, label='$T$ = {:.1f}°C'.format(t_close))

                    # extract tau values (at certain pO2) from calibration plane that correspond to measurement point
                    ls_ox = []
                    ls_tau_cal = []
                    for ox_ in tau_calib_s.loc[:self.pO2_range[-1]].index.tolist():
                        ox_cal = min(tau_calib.index, key=lambda x: abs(x - ox_))
                        ls_ox.append(ox_cal)
                        ls_tau_cal.append(tau_calib.loc[ox_cal, t_close]*1e3)

                    # error bars
                    yerr = 2 * tau_calib_er[t_close_] * 1e3
                    yerr2 = yerr.loc[:self.pO2_range[-1]].values

                    self.ax_tau_tab6_o2.errorbar(ls_ox, ls_tau_cal, yerr=yerr2, fmt='o', ms=2, color=col_tau[en])

                    self.progress_tab6.setValue(62.5)
                # ---------------------------------------------------
                # Temperature dependence for different oxygen values
                for en, o_ in enumerate(ox_example):
                    if en < len(col_tau):
                        ls_ist = ls_[0]
                    elif en >= len(col_tau):
                        if en >= 2 * len(col_tau):
                            ls_ist = ls_[2]
                        else:
                            ls_ist = ls_[1]

                    # find closest value in list
                    o_close = min(tau_calib.index, key=lambda x: abs(x - o_))
                    o_close_ = min(tau_calib_s.index, key=lambda x: abs(x - o_))

                    # fitting curve based on measurement data
                    self.ax_tau_tab6_temp.plot(tau_calib.loc[o_close, :]*1e3, color=col_tau[en], alpha=.9, linewidth=1.25,
                                               ls=ls_ist, label='$pO_2$ = {:.1f}hPa'.format(o_close))
                    # error bars
                    yerr = 2*tau_calib_er.loc[o_close_] *1e3
                    yerr2 = yerr.values[:-1]
                    # select temp measurement (closest to tau_calib columns)
                    t_new = []
                    for temp2 in tau_calib_s.columns[:-1]:
                        t_new.append(min(tau_calib.columns, key=lambda x: abs(x - temp2)))
                    self.ax_tau_tab6_temp.errorbar(tau_calib_s.columns[:-1], tau_calib.loc[o_close, t_new].values*1e3,
                                                   yerr=yerr2, fmt='o', ms=2, color=col_tau[en])

                # Layout
                self.ax_tau_tab6_o2.legend(fontsize=7)
                self.ax_tau_tab6_temp.legend(fontsize=7)
                self.ax_tau_tab6_o2.set_xlim(-0.2, tau_calib.index[-1]*1.05)
                self.ax_tau_tab6_o2.set_ylim(1.45, 5.75)
                self.ax_tau_tab6_temp.set_xlim(9, 34)
                self.ax_tau_tab6_temp.set_ylim(2.45, 5.75)
                self.ax_tau_tab6_o2.xaxis.set_major_locator(MultipleLocator(2))
                self.ax_tau_tab6_o2.xaxis.set_minor_locator(MultipleLocator(0.5))
                self.ax_tau_tab6_o2.yaxis.set_major_locator(MultipleLocator(1.))
                self.ax_tau_tab6_o2.yaxis.set_minor_locator(MultipleLocator(0.2))
                self.ax_tau_tab6_temp.xaxis.set_major_locator(MultipleLocator(5))
                self.ax_tau_tab6_temp.xaxis.set_minor_locator(MultipleLocator(2.5))
                self.ax_tau_tab6_temp.yaxis.set_major_locator(MultipleLocator(0.5))
                self.ax_tau_tab6_temp.yaxis.set_minor_locator(MultipleLocator(0.25))
                self.ax_tau_tab6_o2.tick_params(which='both', axis='both', direction='in', labelsize=8)
                self.ax_tau_tab6_temp.tick_params(which='both', axis='both', direction='in', labelsize=8)

            self.fig_tau_tab6.canvas.draw()
            self.progress_tab6.setValue(75)

            # ------------------------------------------------------------------------------------------------------
            # intensity ratio
            if self.plot_3d_checkbox.isChecked() is True:
                self.ax_int_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=1/int_calib, rstride=1, cstride=1, color='#37681c',
                                              linewidth=0., antialiased=True)
                self.ax_int_tab6.tick_params(which='both', labelsize=8, pad=1)
                self.ax_int_tab6.set_ylim(0, pO2_Y.max()*1.05)
                self.ax_int_tab6.set_xlim(temp_X.max() * 1.05, 0)
            else:
                temp_example = [np.float64(t) for t in str(self.temp_example_tab6_edit.text()).split(',')]
                ox_example = [np.float64(o) for o in str(self.ox_example_tab6_edit.text()).split(',')]
                col_int = ['#37681c', 'grey']
                ls_ = ['-.', '--', ':']

                # ---------------------------------------------------
                # Oxygen dependence for different temperatures
                for en, t_ in enumerate(temp_example):
                    if en < len(col_int):
                        ls_ist = ls_[0]
                    elif en >= len(col_int):
                        if en >= 2 * len(col_int):
                            ls_ist = ls_[2]
                        else:
                            ls_ist = ls_[1]

                    # find closest value in list
                    t_close = min(int_calib.columns, key=lambda x: abs(x - t_))
                    t_close_ = min(int_calib_s.columns, key=lambda x: abs(x - t_))

                    # fitting curve based on measurement data
                    self.ax_int_tab6_o2.plot(1/int_calib[t_close], color=col_int[en], alpha=.9, linewidth=1.25, ls=ls_ist,
                                             label='$T$ = {:.1f}°C'.format(t_close))

                    # extract tau values (at certain pO2) from calibration plane that correspond to measurement point
                    ls_ox = []
                    ls_int_cal = []
                    for ox_ in int_calib_s.loc[:self.pO2_range[-1]].index.tolist():
                        ox_cal = min(int_calib.index, key=lambda x: abs(x - ox_))
                        ls_ox.append(ox_cal)
                        ls_int_cal.append(1/int_calib.loc[ox_cal, t_close])

                    # error bars
                    yerr = int_calib_er[t_close_]
                    yerr2 = yerr.loc[:self.pO2_range[-1]].values

                    self.ax_int_tab6_o2.errorbar(ls_ox, ls_int_cal, yerr=yerr2, fmt='o', ms=2, color=col_int[en])
                self.progress_tab6.setValue(87.5)

                # ---------------------------------------------------
                for en, o_ in enumerate(ox_example):
                    if en < len(col_int):
                        ls_ist = ls_[0]
                    elif en >= len(col_int):
                        if en >= 2 * len(col_int):
                            ls_ist = ls_[2]
                        else:
                            ls_ist = ls_[1]

                    # find closest value in list
                    o_close = min(int_calib.index, key=lambda x: abs(x - o_))
                    o_close_ = min(int_calib_s.index, key=lambda x: abs(x - o_))

                    # fitting curve based on measurement data
                    self.ax_int_tab6_temp.plot(1/int_calib.loc[o_close, :], color=col_int[en], alpha=.9, linewidth=1.25,
                                               ls=ls_ist, label='$pO_2$ = {:.1f}hPa'.format(o_close))
                    # error bars
                    yerr = int_calib_er.loc[o_close_]
                    yerr2 = yerr.values[:-1]
                    # select temp measurement (closest to int_calib columns)
                    t_new = []
                    for temp2 in int_calib_s.columns[:-1]:
                        t_new.append(min(int_calib.columns, key=lambda x: abs(x - temp2)))
                    self.ax_int_tab6_temp.errorbar(int_calib_s.columns[:-1], 1/int_calib.loc[o_close, t_new].values,
                                                   yerr=yerr2, fmt='o', ms=2, color=col_int[en])

                self.ax_int_tab6_o2.legend(fontsize=7)
                self.ax_int_tab6_temp.legend(fontsize=7)
                self.ax_int_tab6_o2.set_xlim(-0.2, tau_calib.index[-1]*1.05)
                self.ax_int_tab6_temp.set_xlim(tau_calib.columns[0]*.85, tau_calib.columns[-1]*1.05)
                self.ax_int_tab6_o2.xaxis.set_major_locator(MultipleLocator(2))
                self.ax_int_tab6_o2.xaxis.set_minor_locator(MultipleLocator(0.5))
                self.ax_int_tab6_o2.yaxis.set_major_locator(MultipleLocator(0.1))
                self.ax_int_tab6_o2.yaxis.set_minor_locator(MultipleLocator(0.05))
                self.ax_int_tab6_temp.xaxis.set_major_locator(MultipleLocator(5))
                self.ax_int_tab6_temp.xaxis.set_minor_locator(MultipleLocator(2.5))
                self.ax_int_tab6_temp.yaxis.set_major_locator(MultipleLocator(0.1))
                self.ax_int_tab6_temp.yaxis.set_minor_locator(MultipleLocator(0.05))
                self.ax_int_tab6_o2.tick_params(which='both', axis='both', direction='in', labelsize=8)
                self.ax_int_tab6_temp.tick_params(which='both', axis='both', direction='in', labelsize=8)
            self.fig_int_tab6.canvas.draw()

            # simulation completed
            self.progress_tab6.setValue(100)

        # -------------------------------------------------------------------------------------------------------------
        # if measurement is True
        # -------------------------------------------------------------------------------------------------------------
        else:
            # check that 3D plot is activated
            if self.plot_3d_checkbox.isChecked() is False:
                self.plot_3d_checkbox.setCheckState(True)
                self.plot_2d_checkbox.setCheckState(False)
                self.plot_calibration_system()
                self.temp_example_tab6_edit.setDisabled(True)
                self.ox_example_tab6_edit.setDisabled(True)

            # progress
            self.progress_tab6.setValue(12.5)

            # clear everything form calibration /input table which is not required
            self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                                   cols=[0, 1, 2, 3, 4, 5, 6])
            self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(), cols=[0, 1, 2, 3])

            # measurement point when measurement is stated
            self.message_tab6.append('')
            self.message_tab6.append('--------------------------------------------')
            self.message_tab6.append('Measurement evaluation')

            # change input table --> amplitudes instead of I-ratios
            if self.int_ratio_checkbox.isChecked() is True:
                self.total_amplitude_checkbox.setCheckState(True)
                self.intensity_to_amplitude()

            # measurement point
            for j in [4, 5, 6, 7]:
                try:
                    self.tableINPUT.item(0, j).text()
                except:
                    measurement_failed = QMessageBox()
                    measurement_failed.setIcon(QMessageBox.Information)
                    measurement_failed.setText("Insufficient measurement parameters!")
                    measurement_failed.setInformativeText("For measurement evaluation the superimposed phase angles "
                                                          "and the total amplitudes at two modulation frequencies are "
                                                          "required at least for one point.")
                    measurement_failed.setWindowTitle("Error!")
                    measurement_failed.exec_()
                    return

            # all phase angles including error
            self.phi_meas_f1 = []
            self.phi_meas_f2 = []
            self.Phi_meas_f1_er = {}
            self.Phi_meas_f2_er = {}
            self.ampl_total_f1_meas = []
            self.ampl_total_f2_meas = []

            # check the number of measurement points
            list_meas = []
            for i in range(self.tableINPUT.rowCount()):
                it = self.tableINPUT.item(i, 4)
                if it and it.text():
                    list_meas.append(it.text())

            for i in range(len(list_meas)):
                # phase angles for both modulation frequencies including assumed error
                dphi1 = np.float64(self.tableINPUT.item(i, 4).text().replace(',', '.'))
                dphi2 = np.float64(self.tableINPUT.item(i, 5).text().replace(',', '.'))
                self.phi_meas_f1.append(dphi1)
                self.phi_meas_f2.append(dphi2)
                self.Phi_meas_f1_er[i] = [dphi1 - self.error_assumed, dphi1, dphi1 + self.error_assumed]
                self.Phi_meas_f2_er[i] = [dphi2 - self.error_assumed, dphi2, dphi2 + self.error_assumed]

                # amplitudes for both modulation frequencies
                self.ampl_total_f1_meas.append(np.float64(self.tableINPUT.item(i, 6).text().replace(',', '.')))
                self.ampl_total_f2_meas.append(np.float64(self.tableINPUT.item(i, 7).text().replace(',', '.')))

            dev = np.float(self.error_assumed_meas_edit.text().replace(',', '.'))

            # update progress
            self.progress_tab6.setValue(20)

            # ===============================================================================================
            # load calibration
            [tau_calib, tau_tau0_calib, int_calib, int_invert_calib, fig_raw, fig_calib, ax_tau_calib, ax_int_calib,
             temp_reg, pO2_reg,
             para_fit] = Tox.fitting_tsm_for_T_O2_(file=self.calib_file, temp_reg=self.temp_range,
                                                   pO2_reg=self.pO2_range,
                                                   fitpara1='tau/tau0', fitpara2='I/I0', calib_range_temp=None,
                                                   calib_range_O2=None, plot_calib_data=False)
            # update progress
            self.progress_tab6.setValue(60)

            # measurement evaluation
            ls_tauP_meas = []
            ls_int_meas = []
            self.ls_T_calc = []
            self.ls_pO2_calc_tab6 = []
            intersection = {}
            for i in range(len(self.phi_meas_f1)):
                # 1) calculate lifetime and intensity ratio from dphi and amplitudes
                [tauP_meas, Phi_f1_rad_er,
                 Phi_f2_rad_er] = fp.phi_to_lifetime_including_error(phi_f1=self.phi_meas_f1[i], f2=self.f2,
                                                                     phi_f2=self.phi_meas_f2[i], f1=self.f1,
                                                                     err_phaseangle=self.error_assumed, er=True)

                iratio_meas = fp.ampl_to_int_ratio(tauP=tauP_meas, dphi_f1=self.Phi_meas_f1_er[i], f1=self.f1,
                                                   dphi_f2=self.Phi_meas_f2_er[i], f2=self.f2)

                self.message_tab6.append('# Point - {}'.format(i+1))
                self.message_tab6.append('\t Lifetime {:.2f} ± {:.2f} ms'.format(tauP_meas[1]*1e3,
                                                                                 np.array(tauP_meas).std()*1e3))
                self.message_tab6.append('\t Intensity ratio {:.2f} ± {:.2f} ms'.format(iratio_meas[1],
                                                                                        np.array(iratio_meas).std()))
                self.message_tab6.append('')

                # update progress
                self.progress_tab6.setValue(70)

                # -----------------------------------------------------
                # 2) calculate T and pO2
                temp_reg_detailed = np.linspace(start=0, stop=100, num=int(100/0.1+1))
                [self.T_calc, self.pO2_calc_tab6,
                 intersec_] = Tox.measurement_evaluation_(temp_reg=temp_reg_detailed, tau_contour=tau_calib, f1=None,
                                                          ax_tau=None, int_contour=int_calib, tauP_meas=tauP_meas,
                                                          ax_int=None, fig=None, iratio_meas=iratio_meas, dphi_f1=None,
                                                          dphi_f2=None, error_meas=None, f2=None, plotting=False)


                # output to message box
                self.message_tab6.append('\t Determined temperature = {:.2f} ± {:.2f} °C'.format(self.T_calc[1],
                                                                                                 np.array(self.T_calc).std()))
                self.message_tab6.append('\t Determined pO2 = {:.2f} ± {:.2f} hPa'.format(self.pO2_calc_tab6[1],
                                                                                          np.array(self.pO2_calc_tab6).std()))

                # -----------------------------------------------------
                # storage of relevant results in lists or dictionaries
                ls_tauP_meas.append(tauP_meas)
                ls_int_meas.append(iratio_meas)
                self.ls_T_calc.append(self.T_calc)
                self.ls_pO2_calc_tab6.append(self.pO2_calc_tab6)
                intersection[i] = intersec_

                # update progress
                self.progress_tab6.setValue(80)

            # ===============================================================================================
            # 3) plotting
            # preparation - interpolating intersection curve for 3D plotting
            # when more than one measurement point has to be evaluated, select first and last one for visualization
            # first measurement point
            temp_X, pO2_Y = np.meshgrid(tau_calib.columns, tau_calib.index)
            self.ax_tau_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=tau_calib * 1e3, rstride=1, cstride=1, color='navy',
                                          alpha=.9, linewidth=0., antialiased=True)
            self.ax_tau_tab6.tick_params(which='both', labelsize=8)
            self.ax_tau_tab6.set_ylim(0, pO2_Y.max() * 1.05)
            self.ax_tau_tab6.set_xlim(temp_X.max() * 1.05, 0)

            self.ax_int_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=1 / int_calib, rstride=1, cstride=1, color='#37681c',
                                          linewidth=0., antialiased=True)
            self.ax_int_tab6.tick_params(which='both', labelsize=8, pad=1)
            self.ax_int_tab6.set_ylim(0, pO2_Y.max() * 1.05)
            self.ax_int_tab6.set_xlim(temp_X.max() * 1.05, 0)

            xdata1 = intersection[0]['line_tau'][1].T.index
            ydata_tau1 = intersection[0]['line_tau'][1].T.values
            ydata_int1 = intersection[0]['line_int'][1].T.values
            xnew = np.linspace(start=temp_X.min(), stop=temp_X.max(), num=int((temp_X.max() - temp_X.min()) / 0.1 + 1))

            tck_tau1 = interpolate.splrep(xdata1, ydata_tau1, s=10)
            y_tau1 = pd.DataFrame(interpolate.splev(xnew, tck_tau1, der=0), columns=['pO2'], index=xnew).T
            zdata_tau1 = [ls_tauP_meas[0][1] * 1e3] * len(xnew)

            tck_int1 = interpolate.splrep(xdata1, ydata_int1, s=10)
            y_int1 = pd.DataFrame(interpolate.splev(xnew, tck_int1, der=0), columns=['pO2'], index=xnew).T
            zdata_int1 = [1/ls_int_meas[0][1]] * len(xnew)

            if len(self.phi_meas_f1) > 1:
                last = len(self.phi_meas_f1) - 1
                # last measurement point
                xdata2 = intersection[last]['line_tau'][1].T.index
                ydata_tau2 = intersection[last]['line_tau'][1].T.values
                ydata_int2 = intersection[last]['line_int'][1].T.values

                tck_tau2 = interpolate.splrep(xdata2, ydata_tau2, s=10)
                y_tau2 = pd.DataFrame(interpolate.splev(xnew, tck_tau2, der=0), columns=['pO2'], index=xnew).T
                zdata_tau2 = [ls_tauP_meas[last][1] * 1e3] * len(xnew)

                tck_int2 = interpolate.splrep(xdata2, ydata_int2, s=10)
                y_int2 = pd.DataFrame(interpolate.splev(xnew, tck_int2, der=0), columns=['pO2'], index=xnew).T
                zdata_int2 = [ls_int_meas[last][1]] * len(xnew)

                # update progress
                self.progress_tab6.setValue(85)

            # -----------------------------------------------------
            # 3.1) planes of calculated lifetime and intensity ratio respectively
            # lifetime - first measurement point
            self.ax_tau_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=intersection[0]['df_tau'][1]*1e3, rstride=1, cstride=1,
                                          linewidth=0, color='lightgrey', antialiased=True)
            self.ax_tau_tab6.plot(xnew, y_tau1.loc['pO2', :].values, zdata_tau1, lw=2., color='darkorange')
            self.fig_tau_tab6.canvas.draw()

            # intensity ratio - first measurement point
            self.ax_int_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=1/intersection[0]['df_int'][1], rstride=1, cstride=1,
                                          color='grey', linewidth=0, antialiased=True)
            self.ax_int_tab6.plot(xnew, y_int1.loc['pO2', :].values, zdata_int1, lw=2., color='#ffbb00')
            self.fig_int_tab6.canvas.draw()

            if len(self.phi_meas_f1) > 1:
                last = len(self.phi_meas_f1) - 1
                # lifetime - last measurement point
                self.ax_tau_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=intersection[last]['df_tau'][1] * 1e3, rstride=1,
                                              cstride=1, linewidth=0, color='lightgrey', antialiased=True)
                self.ax_tau_tab6.plot(xnew, y_tau2.loc['pO2', :].values, zdata_tau2, lw=2., color='darkorange')
                self.fig_tau_tab6.canvas.draw()

                # intensity ratio - last measurement point
                self.ax_int_tab6.plot_surface(X=temp_X, Y=pO2_Y, Z=1/intersection[last]['df_int'][1], rstride=1, cstride=1,
                                              color='grey', linewidth=0, antialiased=True)
                self.ax_int_tab6.plot(xnew, y_int2.loc['pO2', :].values, 1/zdata_int2, lw=2., color='#ffbb00')
                self.fig_int_tab6.canvas.draw()

                # update progress
                self.progress_tab6.setValue(90)

            # -----------------------------------------------------
            # 3.2) line intersection
            # first measurement point
            x_tau1 = intersection[0]['point_tau'][1].columns
            x_int1 = intersection[0]['point_int'][1].columns
            self.ax_pO2T_tab6.plot(x_tau1, intersection[0]['point_tau'][1].loc['pO2'].values, lw=1., color='navy',
                                   label='tau intersec')
            self.ax_pO2T_tab6.plot(x_int1, intersection[0]['point_int'][1].loc['pO2'].values, lw=1., color='#37681c',
                                   label='Iratio intersec')

            self.ax_pO2T_tab6.legend(fontsize=9, frameon=True, fancybox=True, loc=0)
            self.ax_pO2T_tab6.fill_between(x_tau1, intersection[0]['point_tau'][0].loc['pO2'].values,
                                           intersection[0]['point_tau'][2].loc['pO2'].values, lw=1., color='slategrey',
                                           alpha=0.2)
            self.ax_pO2T_tab6.plot(x_tau1, intersection[0]['point_tau'][0].loc['pO2'].values, lw=.5, ls='--',
                                   color='navy', label='tau intersec')
            self.ax_pO2T_tab6.plot(x_tau1, intersection[0]['point_tau'][2].loc['pO2'].values, lw=.5, ls='--',
                                   color='navy', label='tau intersec')
            self.ax_pO2T_tab6.fill_between(x_int1, intersection[0]['point_int'][0].loc['pO2'].values,
                                           intersection[0]['point_int'][2].loc['pO2'].values, lw=1.,
                                           color='slategrey', alpha=0.2)

            self.ax_pO2T_tab6.axvline(x=self.ls_T_calc[0][1], ymin=0, ymax=1, color='crimson')
            self.ax_pO2T_tab6.axhline(y=self.ls_pO2_calc_tab6[0][1], xmin=0, xmax=1, color='crimson')
            self.ax_pO2T_tab6.set_xlim(0, self.temp_range.max()*1.05)
            self.ax_pO2T_tab6.set_ylim(0, self.pO2_range.max()*1.05)
            self.fig_pO2T_tab6.canvas.draw()

            # last measurement point
            if len(self.phi_meas_f1) > 1:
                last = len(self.phi_meas_f1) - 1
                x_tau2 = intersection[last]['point_tau'][1].columns
                x_int2 = intersection[last]['point_int'][1].columns
                self.ax_pO2T_tab6.plot(x_tau2, intersection[last]['point_tau'][1].loc['pO2'].values, lw=1., ls='--',
                                       color='navy', label='tau intersec')
                self.ax_pO2T_tab6.plot(x_int2, intersection[last]['point_int'][1].loc['pO2'].values, lw=1., ls='--',
                                       color='#37681c', label='Iratio intersec')

                self.ax_pO2T_tab6.fill_between(x_tau2, intersection[last]['point_tau'][0].loc['pO2'].values,
                                               intersection[last]['point_tau'][2].loc['pO2'].values, lw=1.,
                                               color='slategrey', alpha=0.2)
                self.ax_pO2T_tab6.plot(x_tau2, intersection[last]['point_tau'][0].loc['pO2'].values, lw=.5, ls='--',
                                       color='navy', label='tau intersec')
                self.ax_pO2T_tab6.plot(x_tau2, intersection[last]['point_tau'][2].loc['pO2'].values, lw=.5, ls='--',
                                       color='navy', label='tau intersec')
                self.ax_pO2T_tab6.fill_between(x_int2, intersection[last]['point_int'][0].loc['pO2'].values,
                                               intersection[last]['point_int'][2].loc['pO2'].values, lw=1.,
                                               color='slategrey', alpha=0.2)

                self.ax_pO2T_tab6.axvline(x=self.ls_T_calc[-1][1], ymin=0, ymax=1, color='crimson', ls='--')
                self.ax_pO2T_tab6.axhline(y=self.ls_pO2_calc_tab6[-1][1], xmin=0, xmax=1, color='crimson', ls='--')
                self.fig_pO2T_tab6.canvas.draw()

            # update progress
            self.progress_tab6.setValue(100)

        print('pO2 / T dualsensing finished')
        self.run_tab6_button.setStyleSheet("color: white; background-color: QLinearGradient(x1: 0, y1: 0, x2: 0, y2: 1, "
                                           "stop: 0 #0a9eb7, stop: 1 #044a57); border-width: 1px; border-color: #077487; "
                                           "border-style: solid; border-radius: 7; padding: 5px; font-size: 10px; "
                                           "padding-left: 1px; padding-right: 5px; min-height: 10px; max-height: 18px;")
        print('#-------------------------------------------------------------------')

# -------------------------------------------------------------------------------------------------------------
# Clear everything
# -------------------------------------------------------------------------------------------------------------
    def clear_calib(self):
        self.clear_table_parts(table=self.tableCalibration, rows=self.tableCalibration.rowCount(),
                               cols=self.tableCalibration.columnCount())

    def clear_input(self):
        # clear entries in input table
        self.clear_table_parts(table=self.tableINPUT, rows=self.tableINPUT.rowCount(),
                               cols=self.tableINPUT.columnCount())

    def clear_files(self):
        # clear files for pH/O2 sensing
        if self.tabs.currentIndex() == 2:
            self.conv_file = None
            self.conversion_edit.clear()
        # clear files for pH/T sensing
        elif self.tabs.currentIndex() == 3:
            self.comp_file = None
            self.compensation_edit.clear()
        # clear files for CO2/O2 sensing
        elif self.tabs.currentIndex() == 4:
            self.conv_file = None
            self.conversion_edit.clear()
        # clear files for T/O2 sensing
        elif self.tabs.currentIndex() == 5:
            self.calib_file = None
            self.calib_edit.clear()
        else:
            pass

    def clear_all(self):
        # 1st and 2nd tab - simulation
        if self.tabs.currentIndex() == 0 or self.tabs.currentIndex() == 1:
            # update figures - 1st tab
            self.message.clear()

            self.ax_phaseangle.cla()
            self.ax_phaseangle.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
            self.ax_phaseangle.set_ylabel('Superimposed phase angle Phi [°]', fontsize=9)
            self.ax_phaseangle.set_xlim(0, 20)
            self.ax_phaseangle.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_phaseangle.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_phaseangle.canvas.draw()

            x = np.float64(self.lifetime_phosphor_edit.text().replace(',', '.'))
            self.ax_lifetime.cla()
            self.ax_lifetime.set_xlabel('Modulation frequency f2 [Hz]', fontsize=9)
            self.ax_lifetime.set_ylabel('lifetime tau [µs]', fontsize=9)
            self.ax_lifetime.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
            self.ax_lifetime.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_lifetime.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_lifetime.canvas.draw()

            self.ax_lifetime_er.cla()
            self.ax_lifetime_er.set_xlim(0, 20)
            self.ax_lifetime_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
            self.ax_lifetime_er.set_ylabel('Rel. error rate [%]', fontsize=9)
            self.ax_lifetime_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_lifetime_err.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_lifetime_err.canvas.draw()

            # update figures - 2nd tab
            self.message_int.clear()

            self.ax_intensity.cla()
            self.ax_intensity.set_xlim(0, 20)
            self.ax_intensity.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
            self.ax_intensity.set_ylabel('Intensity ratio', fontsize=9)
            self.ax_intensity.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_intensity.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_intensity.canvas.draw()

            self.ax_intensity_abs.cla()
            self.ax_intensity_abs.set_xlim(x/1000 * 0.7, x/1000 * 1.3)
            self.ax_intensity_abs.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
            self.ax_intensity_abs.set_ylabel('Abs. deviation intensity ratio', fontsize=9)
            self.ax_intensity_abs.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_intensity_abs.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_intensity_abs.canvas.draw()

            self.ax_intensity_er.cla()
            self.ax_intensity_er.set_xlim(0, 20)
            self.ax_intensity_er.set_xlabel('Modulation frequency f2 [kHz]', fontsize=9)
            self.ax_intensity_er.set_ylabel('Rel. error rate [%]', fontsize=9)
            self.ax_intensity_er.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_intensity_er.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89)
            self.fig_intensity_er.canvas.draw()

    # -------------------------------------------------------------------------------------------------------------
        # 3rd tab - pH/pO2 dual sensor
        elif self.tabs.currentIndex() == 2:
            self.message_tab3.clear()

            # calibration pH at 2 frequencies
            self.ax_pH_calib.cla()
            self.fig_pH_calib.clear()
            self.ax_pH_calib = self.fig_pH_calib.gca()
            self.ax_pH_calib.set_xlim(0, 15)
            self.ax_pH_calib.set_xlabel('pH', fontsize=9)
            self.ax_pH_calib.set_ylabel('Rel. intensity $I_F$ [%]', fontsize=9)
            self.ax_pH_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pH_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pH_calib.canvas.draw()

            # calibration pO2 sensing
            self.ax_pO2_calib.cla()
            self.fig_pO2_calib.clear()
            self.ax_pO2_calib = self.fig_pO2_calib.gca()
            self.ax_pO2_calib.set_xlim(0, 100)
            self.ax_pO2_calib.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_pO2_calib.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
            self.ax_pO2_calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pO2_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pO2_calib.canvas.draw()

            # O2 sensing
            self.ax_pO2.cla()
            self.fig_pO2.clear()
            self.ax_pO2 = self.fig_pO2.gca()
            self.ax_pO2.set_xlim(0, 100)
            self.ax_pO2.set_ylim(0, 105)
            self.ax_pO2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_pO2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
            self.ax_pO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pO2.canvas.draw()

            # pH sensing
            self.ax_pH.cla()
            self.fig_pH.clear()
            self.ax_pH = self.fig_pH.gca()
            self.ax_pH.set_xlim(0, 15)
            self.ax_pH.set_ylim(0, 105)
            self.ax_pH.set_xlabel('pH', fontsize=9)
            self.ax_pH.set_ylabel('Rel. intensity $I_F$ [%]', fontsize=9)
            self.ax_pH.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pH.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pH.canvas.draw()

    # -------------------------------------------------------------------------------------------------------------
        # 4rd tab - pH / T dual sensor
        elif self.tabs.currentIndex() == 3:
            self.message_tab4.clear()

            # calibration pH at 2 frequencies
            self.ax_pH_calib2.cla()
            self.ax_pH_calib2_mir.cla()
            self.fig_pH_calib2.clear()
            self.ax_pH_calib2 = self.fig_pH_calib2.gca()
            self.ax_pH_calib2_mir = self.ax_pH_calib2.twinx()
            self.ax_pH_calib2.set_xlim(0, 15)
            self.ax_pH_calib2.set_xlabel('pH', fontsize=9)
            self.ax_pH_calib2.set_ylabel('cot(Φ)', fontsize=9)
            self.ax_pH_calib2_mir.set_ylabel('Φ [deg]', fontsize=9)
            self.ax_pH_calib2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pH_calib2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
            self.fig_pH_calib2.canvas.draw()

            # calibration temp compensation
            self.ax_temp_calib[0][0].cla()
            self.ax_temp_calib[1][0].cla()
            self.ax_temp_calib[0][1].cla()
            self.ax_temp_calib[1][1].cla()
            self.fig_temp_calib.clear()
            self.ax_temp_calib = self.fig_temp_calib.subplots(nrows=2, ncols=2, sharex=True)
            self.ax_temp_calib[0][0].set_xlim(0, 100)
            self.ax_temp_calib[1][0].set_xlim(0, 100)
            self.ax_temp_calib[0][1].set_xlim(0, 100)
            self.ax_temp_calib[1][1].set_xlim(0, 100)
            self.ax_temp_calib[1][0].set_xlabel('Temperature [°C]', fontsize=9)
            self.ax_temp_calib[1][1].set_xlabel('Temperature [°C]', fontsize=9)
            self.ax_temp_calib[0][0].set_ylabel('slope', fontsize=9)
            self.ax_temp_calib[0][1].set_ylabel('bottom', fontsize=9)
            self.ax_temp_calib[1][0].set_ylabel('pka', fontsize=9)
            self.ax_temp_calib[1][1].set_ylabel('top', fontsize=9)
            self.ax_temp_calib[0][0].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True, right=True)
            self.ax_temp_calib[1][0].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True, right=True)
            self.ax_temp_calib[0][1].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True, right=True)
            self.ax_temp_calib[1][1].tick_params(axis='both', which='both', labelsize=7, direction='in', top=True, right=True)
            self.fig_temp_calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_temp_calib.canvas.draw()

            # temperature sensing
            self.ax_temp.cla()
            self.fig_temp.clear()
            self.ax_temp = self.fig_temp.gca()
            self.ax_temp.set_xlim(0, 100)
            self.ax_temp.set_xlabel('Temperature [°C]', fontsize=9)
            self.ax_temp.set_ylabel('$τ_P$ [µs]', fontsize=9)
            self.ax_temp.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_temp.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_temp.canvas.draw()

            # pH sensing
            self.ax_pH2.cla()
            self.ax_pH2_mir.cla()
            self.fig_pH2.clear()
            self.ax_pH2 = self.fig_pH2.gca()
            self.ax_pH2_mir = self.ax_pH2.twinx()
            self.ax_pH2.set_xlim(0, 15)
            self.ax_pH2.set_xlabel('pH', fontsize=9)
            self.ax_pH2.set_ylabel('cot(Φ)', fontsize=9)
            self.ax_pH2_mir.set_ylabel('Φ [deg]', fontsize=9)
            self.ax_pH2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pH2.subplots_adjust(left=0.14, right=0.85, bottom=0.2, top=0.98)
            self.fig_pH2.canvas.draw()

    # -------------------------------------------------------------------------------------------------------------
        # 5th tab - CO2 / pO2 dual sensor
        elif self.tabs.currentIndex() == 4:

            self.message_tab5.clear()

            # calibration pCO2 at 2 frequencies
            self.ax_CO2calib.cla()
            self.fig_CO2calib.clear()
            self.ax_CO2calib = self.fig_CO2calib.gca()
            self.ax_CO2calib.set_xlim(0, 100)
            self.ax_CO2calib.set_ylim(0, 100)
            self.ax_CO2calib.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
            self.ax_CO2calib.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
            self.ax_CO2calib.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_CO2calib.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_CO2calib.canvas.draw()

            # pH sensing
            self.ax_CO2.cla()
            self.fig_CO2.clear()
            self.ax_CO2 = self.fig_CO2.gca()
            self.ax_CO2.set_xlim(0, 100)
            self.ax_CO2.set_ylim(0, 100)
            self.ax_CO2.set_xlabel('$pCO_2$ [hPa]', fontsize=9)
            self.ax_CO2.set_ylabel('Rel. Intensity I$_F$ [%]', fontsize=9)
            self.ax_CO2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_CO2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_CO2.canvas.draw()

            # calibration pO2 sensing
            self.ax_pO2calib_2.cla()
            self.fig_pO2calib2.clear()
            self.ax_pO2calib_2 = self.fig_pO2calib2.gca()
            self.ax_pO2calib_2.set_xlim(0, 100)
            self.ax_pO2calib_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_pO2calib_2.set_ylabel('$τ_0$ / $τ_P$', fontsize=9)
            self.ax_pO2calib_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pO2calib2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pO2calib2.canvas.draw()

            # O2 sensing
            self.ax_pO2_2.cla()
            self.fig_pO2_2.clear()
            self.ax_pO2_2 = self.fig_pO2_2.gca()
            self.ax_pO2_2.set_xlim(0, 100)
            self.ax_pO2_2.set_ylim(0, 105)
            self.ax_pO2_2.set_xlabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_pO2_2.set_ylabel('Lifetime $τ_P$ [µs]', fontsize=9)
            self.ax_pO2_2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pO2_2.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pO2_2.canvas.draw()

    # -------------------------------------------------------------------------------------------------------------
        # 6th tab - T/O2 dual sensor
        else:
            # clear everything
            self.message_tab6.clear()

            # calibration
            # clear - CALIBRATION plane - lifetime as it is comparable with the measurement results)
            self.ax_tau_tab6.cla()
            self.fig_tau_tab6.clear()

            self.ax_tau_tab6 = self.fig_tau_tab6.gca(projection='3d')
            self.ax_tau_tab6.set_xlim(0, 50)
            self.ax_tau_tab6.set_ylim(0, 20)
            self.ax_tau_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_tau_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_tau_tab6.set_zlabel('$τ$ [ms]', fontsize=9)
            self.ax_tau_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_tau_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_tau_tab6.canvas.draw()

            # clear - CALIBRATION plane - intensity ratio
            self.ax_int_tab6.cla()
            self.fig_int_tab6.clear()
            self.ax_int_tab6 = self.fig_int_tab6.gca(projection='3d')
            self.ax_int_tab6.set_xlim(0, 100)
            self.ax_int_tab6.set_ylim(0, 100)
            self.ax_int_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_int_tab6.set_xlabel('$Temperature$ [°C]', fontsize=9)
            self.ax_int_tab6.set_zlabel('$DF/PF$', fontsize=9)
            self.ax_int_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_int_tab6.subplots_adjust(left=0.14, right=0.9, bottom=0.15, top=0.98)
            self.fig_int_tab6.canvas.draw()

            # clear - intersection of planes
            self.ax_pO2T_tab6.cla()
            self.fig_pO2T_tab6.clear()
            self.ax_pO2T_tab6 = self.fig_pO2T_tab6.gca()
            self.ax_pO2T_tab6.set_xlim(0, 100)
            self.ax_pO2T_tab6.set_ylabel('$pO_2$ [hPa]', fontsize=9)
            self.ax_pO2T_tab6.set_xlabel('Temperature [°C]', fontsize=9)
            self.ax_pO2T_tab6.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            self.fig_pO2T_tab6.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.98)
            self.fig_pO2T_tab6.canvas.draw()

# -------------------------------------------------------------------------------------------------------------
# SAVING AND PRINTING
# -------------------------------------------------------------------------------------------------------------
    def save_report(self):
        # create folder structure
        folder = 'reports'
        subfolder_tab1_2 = 'reports/simulation_errorpropagation'
        subfolder_tab3 = 'reports/pH_O2_dualsensor'
        subfolder_tab4 = 'reports/pH_T_dualsensor'
        subfolder_tab5 = 'reports/CO2_O2_dualsensor'
        subfolder_tab6 = 'reports/O2_T_dualsensor'

        if not os.path.exists(folder):
            os.makedirs(folder)
        if self.tabs.currentIndex() == 0 or self.tabs.currentIndex() == 1:
            if not os.path.exists(subfolder_tab1_2):
                os.makedirs(subfolder_tab1_2)
        elif self.tabs.currentIndex() == 2:
            if not os.path.exists(subfolder_tab3):
                os.makedirs(subfolder_tab3)
        elif self.tabs.currentIndex() == 3:
            if not os.path.exists(subfolder_tab4):
                os.makedirs(subfolder_tab4)
        elif self.tabs.currentIndex() == 4:
            if not os.path.exists(subfolder_tab5):
                os.makedirs(subfolder_tab5)
        elif self.tabs.currentIndex() == 5:
            if not os.path.exists(subfolder_tab6):
                os.makedirs(subfolder_tab6)
        else:
            raise ValueError('Current tab has no folder linked for saving so far...')

        # ------------------------------------------------------------------------------
        # general settings and parameters
        today = datetime.datetime.today().isoformat()
        # simulation or measurement evaluation
        if self.simulation_checkbox.isChecked() is True:
            what = 'simulation'
        else:
            what = 'measurement'

        if self.tabs.currentIndex() != 0 and self.tabs.currentIndex() != 1:
            # parameters
            [self.f1, self.f2, self.error_assumed, self.int_fluoro_max, self.int_phosphor_max] = self.input_collecting()
        # --------------------------------------------------------------------------------------------------------
        # tab selective saving or results and report
        # ------------------------------------------------------------------------------------------------------------
            if self.tabs.currentIndex() == 2:
                # check if something was calculated
                try:
                    self.pH_calib[0]
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                # pH / O2 simulation / measurement
                ls_m = []
                for i in self.int_ratio.index:
                    if 'meas' in i:
                        ls_m.append(i)

                # pH / pO2 simulation / measurement
                slope_tab3 = np.float64(self.slope_tab3_edit.text().replace(',', '.'))
                pKa_tab3 = np.float64(self.pka_tab3_edit.text().replace(',', '.'))
                ksv1_tab3 = np.float64(self.Ksv1_tab3_edit.text().replace(',', '.'))
                ksv2_tab3 = np.float64(self.Ksv2_tab3_edit.text().replace(',', '.'))
                lifetime_tab3 = np.float64(self.lifetime_phos_dualsens_edit.text().replace(',', '.'))

                day_time = str(today[:10]) + '_' + str(today[11:13]) + str(today[14:16]) + str(today[17:19])
                saving_name = subfolder_tab3 + '/' + day_time + '_' + str(what) + '_pH-pO2.txt'

                # ---------------------------------------------------------------------------------------------------
                # preparation dataframe for output
                if self.simulation_checkbox.isChecked() is True:
                    total_width = len(self.pH_calib) + 1
                    l_calib = self.pH_calib
                    l_meas = [self.pH_calc]
                else:
                    total_width = len(self.pH_calib) + len(self.pH_calc)
                    l_calib = self.pH_calib
                    l_meas = [self.pH_calc[i][1] for i in range(len(self.pH_calc))]
                col_sim = ['calib {}'.format(i) for i in range(len(l_calib))] + ['meas {}'.format(i + 1)
                                                                                 for i in range(len(l_meas))]

                df_sim = pd.DataFrame(np.zeros(shape=(0, total_width)))
                df_sim.columns = col_sim
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['pH\O2'] = [self.pO2_calib[0], self.pO2_calib[1], self.pO2_calc[1].round(2)]
                else:
                    df_sim.loc['pH\O2'] = [self.pO2_calib[0], self.pO2_calib[1]] + \
                                         [self.pO2_calc[i][1].round(2) for i in range(len(ls_m))]

                # ---------------------------------------------------------------
                # phase angle calibration points
                df_sim.loc['dPhi1 {:.2f}'.format(self.pH_calib[0])] = [self.Phi_f1_deg['fluoro0, phosphor0'].round(3),
                                                                       self.Phi_f1_deg['fluoro0, phosphor1'].round(3)]+\
                                                                      ['--'] * len(ls_m)
                df_sim.loc['dPhi1 {:.2f}'.format(self.pH_calib[1])] = [self.Phi_f1_deg['fluoro1, phosphor0'].round(3),
                                                                       self.Phi_f1_deg['fluoro1, phosphor1'].round(3)]+\
                                                                      ['--'] * len(ls_m)
                df_sim.loc['dPhi2 {:.2f}'.format(self.pH_calib[0])] = [self.Phi_f2_deg['fluoro0, phosphor0'].round(3),
                                                                       self.Phi_f2_deg['fluoro0, phosphor1'].round(3)]+\
                                                                       ['--'] * len(ls_m)
                df_sim.loc['dPhi2 {:.2f}'.format(self.pH_calib[1])] = [self.Phi_f2_deg['fluoro1, phosphor0'].round(3),
                                                                       self.Phi_f2_deg['fluoro1, phosphor1'].round(3)]+\
                                                                      ['--'] * len(ls_m)
                # --------------------------------
                # phase angle measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['dPhi1 {:.2f}'.format(self.pH_calc[1])] = ['--', '--', self.Phi_f1_deg['meas'].round(3)]
                    df_sim.loc['dPhi2 {:.2f}'.format(self.pH_calc[1])] = ['--', '--', self.Phi_f2_deg['meas'].round(3)]
                else:
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi1 meas'] = ['--', '--'] + [self.phi_meas_f1[c].round(3)
                                                                   for c in range(len(self.phi_meas_f1))]
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi2 meas'] = ['--', '--'] + [self.phi_meas_f2[c].round(3)
                                                                   for c in range(len(self.phi_meas_f2))]
                # ---------------------------------------------------------------
                # intensity ratio calibration points
                df_sim.loc['I-ratio {:.2f}'.format(self.pH_calib[0])] = [self.int_ratio['fluoro0, phosphor0'].round(4),
                                                                         self.int_ratio['fluoro1, phosphor0'].round(4)]+\
                                                                         ['--']*len(ls_m)
                df_sim.loc['I-ratio {:.2f}'.format(self.pH_calib[1])] = [self.int_ratio['fluoro0, phosphor1'].round(4),
                                                                         self.int_ratio['fluoro1, phosphor1'].round(4)]+\
                                                                         ['--']*len(ls_m)
                # --------------------------------
                # intensity ratio measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['I-ratio {:.2f}'.format(self.pH_calc[1])] = ['--', '--', self.int_ratio['meas'].round(4)]
                else:
                    df_sim.loc['I-ratio meas'] = ['--']*len(self.pO2_list) + [self.int_ratio[ls_m[c]].round(3)
                                                                              for c in range(len(ls_m))]
                # ---------------------------------------------------------------
                # summarize measurement/simulation parameter
                df_para = pd.DataFrame(np.zeros(shape=(16, 2)),
                                       index=[' ', 'f1', 'f2', 'I_F(max)', 'I_P(max)', 'meas. uncertainty', 'pH calib',
                                              'pO2 calib [hPa]', 'pH - slope', 'pH - pKa', 'pO2 - tauP', 'pO2 - Ksv',
                                              'pO2 - m', 'pO2 - f', '----', ' '])
                df_para[0] = ['Input', self.f1, self.f2, self.int_fluoro_max, self.int_phosphor_max, self.error_assumed,
                              self.pH_calib[0], self.pO2_calib[0], slope_tab3, pKa_tab3, lifetime_tab3, ksv1_tab3,
                              ksv2_tab3, self.curv_O2_tab3_edit.text(), '----', 'median']
                df_para[1] = ['parameters', 'Hz', 'Hz', '%', '%', '°', self.pH_calib[1].round(2),
                              self.pO2_calib[1].round(2), '', '', 'µs', '', '', '', '----', 'range']

                # add measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_para.loc['pH #1'] = [self.pH_calc[1].round(2), str(self.pH_calc[0].round(2)) + '-' +
                                            str(self.pH_calc[2].round(2))]
                    df_para.loc['pO2 [hPa] #1'] = [self.pO2_calc[1].round(2), str(self.pO2_calc[0].round(2)) + '-' +
                                                   str(self.pO2_calc[2].round(2))]
                else:
                    for c in range(len(ls_m)):
                        df_para.loc['pH #{}'.format(c + 1)] = [self.pH_calc[c][1].round(2),
                                                               str(self.pH_calc[c][0].round(2)) + '-' +
                                                               str(self.pH_calc[c][2].round(2))]
                        df_para.loc['pO2 [hPa] #{}'.format(c + 1)] = [self.pO2_calc[c][1].round(2),
                                                                      str(self.pO2_calc[c][0].round(2)) + '-' +
                                                                      str(self.pO2_calc[c][2].round(2))]
                df_para.loc['----'] = ['----', '----']

                # ---------------------------------------------------------------
                # final combination of input parameter and results
                df_res = pd.DataFrame(pd.concat([df_para, df_sim]))

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, encoding="utf-8")

        # ------------------------------------------------------------------------------------------------------------
            elif self.tabs.currentIndex() == 3:
                # check if something was calculated
                try:
                    self.pH_list[0]
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                # pH / T simulation / measurement
                ls_m = []
                for i in self.Phi_f1_deg.index:
                    if 'meas' in i:
                        ls_m.append(i)

                day_time = str(today[:10]) + '_' + str(today[11:13]) + str(today[14:16]) + str(today[17:19])
                saving_name = subfolder_tab4 + '/' + day_time + '_' + str(what) + '_pH-T.txt'

                # combine / merge parameter
                if isinstance(self.pH_calc, pd.Series):
                    t_bottom = {}
                    t_top = {}
                    t_pka = {}
                    t_slope = {}
                    for c in range(len(ls_m)):
                        t_bottom[c] = self.para_fit_f1['bottom, slope'] * self.temp_calc[c][1] + \
                                      self.para_fit_f1['bottom, intercept']
                        t_top[c] = self.para_fit_f1['top, slope'] * self.temp_calc[c][1] + \
                                   self.para_fit_f1['top, intercept']
                        t_pka[c] = self.para_fit_f1['pka, slope'] * self.temp_calc[c][1] + \
                                   self.para_fit_f1['pka, intercept']
                        t_slope[c] = self.para_fit_f1['slope, slope'] * self.temp_calc[c][1] + \
                                     self.para_fit_f1['slope, intercept']
                else:
                    t_bottom = self.para_fit_f1['bottom, slope'] * self.temp_calc[1] + \
                               self.para_fit_f1['bottom, intercept']
                    t_top = self.para_fit_f1['top, slope'] * self.temp_calc[1] + self.para_fit_f1['top, intercept']
                    t_pka = self.para_fit_f1['pka, slope'] * self.temp_calc[1] + self.para_fit_f1['pka, intercept']
                    t_slope = self.para_fit_f1['slope, slope'] * self.temp_calc[1] + \
                              self.para_fit_f1['slope, intercept']

                # ---------------------------------------------------------------
                # preparation dataframe for output
                if self.simulation_checkbox.isChecked() is True:
                    total_width = len(self.pH_list[:2]) + 1
                    l_calib = self.pH_list[:2]
                    l_meas = [self.pH_calc]
                else:
                    total_width = len(self.pH_list[:2]) + len(self.pH_calc)
                    l_calib = self.pH_list[:2]
                    l_meas = [self.pH_calc[i][1] for i in range(len(self.pH_calc))]
                col_sim = ['calib {}'.format(i) for i in range(len(l_calib))] + ['meas {}'.format(i + 1)
                                                                                       for i in range(len(l_meas))]

                df_sim = pd.DataFrame(np.zeros(shape=(0, total_width)))
                df_sim.columns = col_sim
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['pH\T'] = [self.temp_list[0], self.temp_list[1], self.temp_list[2].round(2)]
                else:
                    df_sim.loc['pH\T'] = [self.temp_list[0], self.temp_list[1]] + \
                                         [self.temp_calc[i][1].round(2) for i in range(len(ls_m))]

                # ---------------------------------------------------------------
                # phase angle calibration points
                df_sim.loc['dPhi1 {:.2f}'.format(self.pH_list[0])] = [self.Phi_f1_deg['fluoro0, phosphor0'].round(3),
                                                                      self.Phi_f1_deg['fluoro0, phosphor1'].round(3)]+\
                                                                     ['--']*len(ls_m)
                df_sim.loc['dPhi1 {:.2f}'.format(self.pH_list[1])] = [self.Phi_f1_deg['fluoro1, phosphor0'].round(3),
                                                                      self.Phi_f1_deg['fluoro1, phosphor1'].round(3)]+\
                                                                     ['--']*len(ls_m)
                df_sim.loc['dPhi2 {:.2f}'.format(self.pH_list[0])] = [self.Phi_f2_deg['fluoro0, phosphor0'].round(3),
                                                                      self.Phi_f2_deg['fluoro0, phosphor1'].round(3)]+\
                                                                     ['--']*len(ls_m)
                df_sim.loc['dPhi2 {:.2f}'.format(self.pH_list[1])] = [self.Phi_f2_deg['fluoro1, phosphor0'].round(3),
                                                                      self.Phi_f2_deg['fluoro1, phosphor1'].round(3)]+\
                                                                     ['--']*len(ls_m)

                # --------------------------------
                # phase angle measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['dPhi1 {:.2f}'.format(self.pH_calc)] = ['--', '--', self.Phi_f1_deg['meas'].round(3)]
                    df_sim.loc['dPhi2 {:.2f}'.format(self.pH_calc)] = ['--', '--', self.Phi_f2_deg['meas'].round(3)]
                else:
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi1 meas'] = ['--', '--'] + [self.Phi_f1_deg[c].round(3) for c in ls_m]
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi2 meas'] = ['--', '--'] + [self.Phi_f2_deg[c].round(3) for c in ls_m]

                # ---------------------------------------------------------------
                # summarize measurement/simulation parameter
                df_para = pd.DataFrame(np.zeros(shape=(9, 2)),
                                       index=[' ', 'f1', 'f2', 'I_F(max)', 'I_P(max)', 'meas. uncertainty', 'pH calib',
                                              'T calib [°C]', '----'])
                df_para[0] = ['Input', self.f1, self.f2, self.int_fluoro_max, self.int_phosphor_max,
                              self.error_assumed, self.pH_list[0], self.temp_list[0], '----']
                df_para[1] = ['parameters', 'Hz', 'Hz', '%', '%', '°', self.pH_list[1], self.temp_list[1], '----']

                # --------------------------------
                # add parameter for temperature compensation
                if self.simulation_checkbox.isChecked() is True:
                    df_para.loc['pH-slope'] = [t_slope.round(3), '']
                    df_para.loc['pH-pKa'] = [t_pka.round(3), '']
                    df_para.loc['T - bottom'] = [t_bottom.round(3), '']
                    df_para.loc['T - top'] = [t_top.round(3), '']
                else:
                    for c in range(len(ls_m)):
                        df_para.loc['pH-slope #{}'.format(c+1)] = [t_slope[c].round(3), '']
                        df_para.loc['pH-pKa #{}'.format(c+1)] = [t_pka[c].round(3), '']
                        df_para.loc['T - bottom #{}'.format(c+1)] = [t_bottom[c].round(3), '']
                        df_para.loc['T - top #{}'.format(c+1)] = [t_top[c].round(3), '']

                # --------------------------------
                # add measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_para.loc['pH #1'] = [self.pH_calc.round(2), '{:.2f}'.format(self.pH_calc - self.pH_calc_std) +
                                            '-' + '{:.2f}'.format(self.pH_calc + self.pH_calc_std)]
                    df_para.loc['T[°C] #1'] = [self.temp_calc[1].round(2), str(self.temp_calc[0].round(2)) + '-' +
                                               str(self.temp_calc[2].round(2))]
                else:
                    for c in range(len(ls_m)):
                        df_para.loc['pH #{}'.format(c + 1)] = [self.pH_calc[c][1].round(2),
                                                               str(self.pH_calc[c][0].round(2)) + '-' +
                                                               str(self.pH_calc[c][2].round(2))]
                        df_para.loc['Temperature [°C] #{}'.format(c + 1)] = [self.temp_calc[c][1].round(2),
                                                                str(self.temp_calc[c][0].round(2)) + '-' +
                                                                str(self.temp_calc[c][2].round(2))]
                df_para.loc['----'] = ['----', '----']

                # ---------------------------------------------------------------
                # final combination of input parameter and results
                df_res = pd.DataFrame(pd.concat([df_para, df_sim]))

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, encoding="utf-8")

    # ------------------------------------------------------------------------------------------------------------
            elif self.tabs.currentIndex() == 4:
                # check if something was calculated
                try:
                    self.pCO2_list[0]
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                # (C)O2 simulation / measurement
                ksv1_CO2_tab5 = np.float64(self.Ksv1_CO2_tab5_edit.text().replace(',', '.'))
                m_CO2_tab5 = np.float64(self.Ksv2_CO2_tab5_edit.text().replace(',', '.'))
                f_CO2_tab5 = np.float64(self.curv_CO2_tab5_edit.text().replace(',', '.'))
                lifetime_tab5 = np.float64(self.lifetime_phos_dualsens_edit.text().replace(',', '.'))
                ksv1_O2_tab5 = np.float64(self.Ksv1_O2_tab5_edit.text().replace(',', '.'))
                m_O2_tab5 = np.float64(self.Ksv2_O2_tab5_edit.text().replace(',', '.'))
                f_O2_tab5 = np.float64(self.curv_O2_tab5_edit.text().replace(',', '.'))
                ls_m = []
                for i in self.int_ratio.index:
                    if 'meas' in i:
                        ls_m.append(i)

                day_time = str(today[:10]) + '_' + str(today[11:13]) + str(today[14:16]) + str(today[17:19])
                saving_name = subfolder_tab5 + '/' + day_time + '_' + str(what) + '_CO2-O2.txt'

                # ---------------------------------------------------------------
                # preparation dataframe for output
                if isinstance(self.pCO2_calc, dict):
                    total_width = len(self.pCO2_list) + len(self.pCO2_calc)
                    l_calib = self.pCO2_list
                    l_meas = [self.pO2_calc[i][1] for i in range(len(self.pO2_calc))]
                else:
                    total_width = len(self.pCO2_list[:-1]) + 1
                    l_calib = self.pCO2_list[:-1]
                    l_meas = [self.pO2_calc[1]]
                col_sim = ['calib {}'.format(i) for i in range(len(l_calib))] + ['meas {}'.format(i + 1)
                                                                                       for i in range(len(l_meas))]
                df_sim = pd.DataFrame(np.zeros(shape=(0, total_width)))
                df_sim.columns = col_sim
                if isinstance(self.pCO2_calc, dict):
                    df_sim.loc['pCO2\pO2 '] = [self.pO2_list[0], self.pO2_list[1]] + \
                                              [self.pO2_calc[i][1].round(2) for i in range(len(self.pO2_calc))]
                else:
                    df_sim.loc['pCO2\pO2 '] = [self.pO2_list[0], self.pO2_list[1], self.pO2_calc[1].round(2)]

                # ---------------------------------------------------------------
                # phase angle calibration points
                df_sim.loc['dPhi1 {:.2f}'.format(self.pCO2_list[0])] = ['--',
                                                                        self.Phi_f1_deg['fluoro0, phosphor1'].round(3)] + \
                                                                       ['--']*len(ls_m)
                df_sim.loc['dPhi1 {:.2f}'.format(self.pCO2_list[1])] = [self.Phi_f1_deg['fluoro1, phosphor0'].round(3),
                                                                        '--'] + ['--']*len(ls_m)

                df_sim.loc['dPhi2 {:.2f}'.format(self.pCO2_list[0])] = ['--',
                                                                        self.Phi_f2_deg['fluoro0, phosphor1'].round(3)] + \
                                                                       ['--']*len(ls_m)
                df_sim.loc['dPhi2 {:.2f}'.format(self.pCO2_list[1])] = [self.Phi_f2_deg['fluoro1, phosphor0'].round(3),
                                                                        '--'] + ['--']*len(ls_m)

                # --------------------------------
                # phase angle measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['dPhi1 {:.2f}'.format(self.pCO2_calc[1])] = ['--', '--',
                                                                            self.Phi_f1_deg['meas'].round(3)]
                    df_sim.loc['dPhi2 {:.2f}'.format(self.pCO2_calc[1])] = ['--', '--',
                                                                            self.Phi_f2_deg['meas'].round(3)]
                else:
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi1 meas'] = ['--', '--'] + [self.phi_meas_f1[c].round(3)
                                                                   for c in range(len(self.phi_meas_f1))]
                    for c in range(len(ls_m)):
                        df_sim.loc['dPhi2 meas'] = ['--', '--'] + [self.phi_meas_f2[c].round(3)
                                                                   for c in range(len(self.phi_meas_f2))]

                # ---------------------------------------------------------------
                # intensity ratio calibration points
                df_sim.loc['I-ratio {:.2f}'.format(self.pCO2_list[0])] = ['--',
                                                                          self.int_ratio['fluoro0, phosphor1'].round(3)]+ \
                                                                          ['--']*len(ls_m)
                df_sim.loc['I-ratio {:.2f}'.format(self.pCO2_list[1])] = [self.int_ratio['fluoro1, phosphor0'].round(3),
                                                                          '--'] + ['--']*len(ls_m)

                # --------------------------------
                # intensity ratio measurement point(s)
                if self.simulation_checkbox.isChecked() is True:
                    df_sim.loc['I-ratio {:.2f}'.format(self.pCO2_calc[1])] = ['--', '--', self.int_ratio['meas'].round(4)]
                else:
                    df_sim.loc['I-ratio meas'] = ['--']*len(self.pO2_list) + [self.int_ratio[ls_m[c]].round(3)
                                                                              for c in range(len(ls_m))]

                # ---------------------------------------------------------------
                # summarize measurement/simulation parameter
                df_para = pd.DataFrame(np.zeros(shape=(17, 2)),
                                       index=[' ', 'f1', 'f2', 'I_F(max)', 'I_P(max)', 'meas. uncertainty',
                                              'pCO2 calib [hPa]', 'pO2 calib [hPa]', 'pCO2 - Ksv1', 'pCO2 - m',
                                              'pCO2 - f', 'pO2 - tauP', 'pO2 - Ksv', 'pO2 - m', 'pO2 - f', '----', ' '])
                df_para[0] = ['Input', self.f1, self.f2, self.int_fluoro_max, self.int_phosphor_max,
                              self.error_assumed, self.pCO2_list[0], self.pO2_list[0], ksv1_CO2_tab5, m_CO2_tab5,
                              f_CO2_tab5, lifetime_tab5, ksv1_O2_tab5, m_O2_tab5, f_O2_tab5, '----', 'median']
                df_para[1] = ['parameters', 'Hz', 'Hz', '%', '%', '°', self.pCO2_list[1], self.pO2_list[1], '', '',
                              '', 'µs', '', '', '', '----', 'range']

                # add measurement point(s)
                if isinstance(self.pCO2_calc, dict):
                    for c in range(len(ls_m)):
                        df_para.loc['pCO2 #{}'.format(c + 1)] = [self.pCO2_calc[c][1].round(2),
                                                                 str(self.pCO2_calc[c][0].round(2)) + '-' +
                                                                 str(self.pCO2_calc[c][2].round(2))]
                        df_para.loc['pO2 #{}'.format(c + 1)] = [self.pO2_calc[c][1].round(2),
                                                                str(self.pO2_calc[c][0].round(2)) + '-' +
                                                                str(self.pO2_calc[c][2].round(2))]
                else:
                    df_para.loc['pCO2 #1'] = [self.pCO2_calc[1].round(2), str(self.pCO2_calc[0].round(2)) + '-' +
                                              str(self.pCO2_calc[2].round(2))]
                    df_para.loc['pO2 #1'] = [self.pO2_calc[1].round(2), str(self.pO2_calc[0].round(2)) + '-' +
                                             str(self.pO2_calc[2].round(2))]
                df_para.loc['----'] = ['----', '----']

                # ---------------------------------------------------------------
                # final combination of input parameter and results
                df_res = pd.DataFrame(pd.concat([df_para, df_sim]))

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, encoding="utf-8")

        # ------------------------------------------------------------------------------------------------------------
            elif self.tabs.currentIndex() == 5:
                # check if something was calculated
                try:
                    self.T_calc[1]
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("There is nothing to save for T/O2-simulation as it just presents "
                                                     "the calibration file... ")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                # saving name
                day_time = str(today[:10]) + '_' + str(today[11:13]) + str(today[14:16]) + str(today[17:19])
                saving_name = subfolder_tab6 + '/' + day_time + '_' + str(what) + '_T-' +\
                              str(np.float64(self.T_calc[1]).round(2)) + '_O2-' +\
                              str(np.float64(self.pO2_calc_tab6[1]).round(2)) + '.txt'

                if self.simulation_checkbox.isChecked() is False:
                    out = pd.Series({'calibration file': self.calib_file.split('/')[-1], 'f1 [Hz]': self.f1,
                                     'f2 [Hz]': self.f2, 'meas uncertainty [deg]': self.error_assumed,
                                     'dPhi(f1) [deg]': self.phi_meas_f1, 'dPhi(f2) [deg]': self.phi_meas_f2,
                                     'A(f1) [mV]': self.ampl_total_f1_meas, 'A(f2) [mV]': self.ampl_total_f2_meas,
                                     'Temperature_calc [°C]': [round(i, 2) for i in self.T_calc], 'pO2 calc [hPa]':
                                         [round(i, 2) for i in self.pO2_calc_tab6]})


                else:
                    out = pd.Series({'calibration file': self.calib_file.split('/')[-1], 'f1 [Hz]': self.f1,
                                     'f2 [Hz]': self.f2, 'meas uncertainty [deg]': self.error_assumed,
                                     'dPhi(f1) [deg]': self.phi_meas_f1, 'dPhi(f2) [deg]': self.phi_meas_f2,
                                     'A(f1) [mV]': self.ampl_total_f1_meas, 'A(f2) [mV]': self.ampl_total_f2_meas,
                                     'Temperature_calc [°C]': [round(i, 2) for i in self.T_calc], 'pO2 calc [hPa]':
                                         [round(i, 2) for i in self.pO2_calc_tab6]})

                # saving pd.Series to csv
                out.to_csv(saving_name, sep='\t', decimal='.', header=False, encoding="utf-8")

        # ------------------------------------------------------------------------------------------------------------
            else:
                self.message.append('New Hybridsensor defined. Create new file for report...')
                self.message_int.append('New Hybridsensor defined. Create new file for report...')
                self.message_tab3.append('New Hybridsensor defined. Create new file for report...')
                self.message_tab4.append('New Hybridsensor defined. Create new file for report...')
                self.message_tab5.append('New Hybridsensor defined. Create new file for report...')
                self.message_tab6.append('New Hybridsensor defined. Create new file for report...')
                saving_name = None

    # ------------------------------------------------------------------------------------------------------------
        else:
            if self.lifetime_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                t = 1
                I = 1
                try:
                    self.signals
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals, lifetime_phosphor=self.tau_phos1,
                                                      intensity_ratio=self.i_ratio1, error_assumed=self.error_prop, 
                                                      today=today, f=self.f1_prop)

                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime2_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                t = 2
                I = 1
                try:
                    self.signals2
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals2, lifetime_phosphor=self.tau_phos2,
                                                      intensity_ratio=self.i_ratio1, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime3_checkbox.isChecked() is True and self.intensity_checkbox.isChecked() is True:
                t = 3
                I = 1

                try:
                    self.signals3
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals3, lifetime_phosphor=self.tau_phos3,
                                                      intensity_ratio=self.i_ratio1, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")

            if self.lifetime_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                t = 1
                I = 2
                try:
                    self.signals4
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals4, lifetime_phosphor=self.tau_phos1,
                                                      intensity_ratio=self.i_ratio2, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime2_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                t = 2
                I = 2
                try:
                    self.signals5
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals5, lifetime_phosphor=self.tau_phos2,
                                                      intensity_ratio=self.i_ratio2, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime3_checkbox.isChecked() is True and self.intensity2_checkbox.isChecked() is True:
                t = 3
                I = 2
                try:
                    self.signals6
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals6, lifetime_phosphor=self.tau_phos3,
                                                      intensity_ratio=self.i_ratio2, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")

            if self.lifetime_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                t = 1
                I = 3
                try:
                    self.signals7
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals7, lifetime_phosphor=self.tau_phos1,
                                                      intensity_ratio=self.i_ratio3, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime2_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                t = 2
                I = 3
                try:
                    self.signals8
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals8, lifetime_phosphor=self.tau_phos2,
                                                      intensity_ratio=self.i_ratio3, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")
            if self.lifetime3_checkbox.isChecked() is True and self.intensity3_checkbox.isChecked() is True:
                t = 3
                I = 3
                try:
                    self.signals9
                except:
                    saving_failed = QMessageBox()
                    saving_failed.setIcon(QMessageBox.Information)
                    saving_failed.setText("Upps...")
                    saving_failed.setInformativeText("An error occurred during saving - unfortunately there is nothing"
                                                     " to save...")
                    saving_failed.setWindowTitle("Error!")
                    saving_failed.exec_()
                    return

                [df_res, filename] = self.report_save(t=t, I=I, signals=self.signals9, lifetime_phosphor=self.tau_phos3,
                                                      intensity_ratio=self.i_ratio3, error_assumed=self.error_prop,
                                                      today=today, f=self.f1_prop)
                saving_name = subfolder_tab1_2 + '/' + filename

                # saving dataframe to csv
                df_res.to_csv(saving_name, sep='\t', decimal='.', header=False, index=False, encoding="utf-8")

        print('# saving finished')
        print('#-------------------------------------------------------------------')

        # Infobox saving done
        saving_done = QMessageBox()
        saving_done.setIcon(QMessageBox.Information)
        saving_done.setText("Saving done")
        saving_done.setInformativeText("Report successfully saved under \n {}".format(saving_name))
        saving_done.setWindowTitle("Info!")
        saving_done.exec_()


#################################################################################################
if __name__ == '__main__':
    app = QApplication(sys.argv)
    g = Gui()
    app.exec_()
