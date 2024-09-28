import pandas as pd
import math
import pkg_resources
from enum import Enum
import sys
import random
import statistics
import plotly.graph_objects as go

CSV_NAME = 'current_gain_data_set.csv'
BOARDS_DATA_CSV_PATH = pkg_resources.resource_filename(__name__, CSV_NAME)

class BridgeType(Enum):
    MINEW = {
        "avg_beacon_current": 208,
        "beacon_tx_time": 1.5,
        "avg_rx_current": 17.5,
        "board_max_output_power": 26.9,
        "lower_bound_output_power_cfg": -9,
        "upper_bound_output_power_cfg": 3

    }
    FANSTEL = {
        "avg_beacon_current": 101,
        "beacon_tx_time": 1.52,
        "avg_rx_current": 42.5,
        "board_max_output_power": 27.3,
        "lower_bound_output_power_cfg": -8,
        "upper_bound_output_power_cfg": 9
    }
    KOAMTAC = {
        "avg_beacon_current": 129.3,
        "beacon_tx_time": 2.6,
        "avg_rx_current": 17.3,
        "board_max_output_power": 27,
        "lower_bound_output_power_cfg": -9,
        "upper_bound_output_power_cfg": 12
    }


class ChargingEstimatorSimulator: 
    def __init__(self, bridge_type, rxtx_period, sim_time, bat_cap, int_bat_level, dis_volt, charge_volt, sam_period, charging_func, const_charge_current, bat_level_meas_res, time_to_last, inf_time_mode, interrupt_charging):
        self.bridge_type = bridge_type
        self.rxtx_period_beacons = rxtx_period
        self.charging_func = charging_func
        self.sim_time = sim_time
        self.time_target_hr = time_to_last
        self.inf_time_mode = inf_time_mode
        self.interrupt_charging = interrupt_charging
        self.bat_cap_wh = bat_cap
        self.int_bat_level = int_bat_level
        self.board_voltage = dis_volt
        self.charging_voltage = charge_volt
        self.sampling_period = sam_period
        self.bat_level_meas_resolution = bat_level_meas_res
        self.constant_charging_current = const_charge_current
        self.bat_level = int_bat_level
        self.bat_level_est = int_bat_level
        self.elapsed_time_minutes = 0
        self.avg_current_opt = 0
        self.avg_beacon_current = None
        self.beacon_tx_time = None
        self.avg_rx_current = None
        self.board_max_output_power = None
        self.lower_bound_output_power_cfg = None
        self.upper_bound_output_power_cfg = None
        self.charging_func_type = None
        self.data_df = None
        self.charging_power_est = 0
        self.charging_power_list = []
        self.charging_power_list_est = []
        self.charging_current = []
        self.time_axis_est = []
        self.time_axis_charging_power = []
        self.time_axis_filter = []
        self.time_axis_filter_est = []
        self.bat_level_list = []
        self.charging_power_avg_list = []
        self.filtered_charging_power_list_est = []
        self.filtered_charging_power_est = 0
        self.time_window_est_filter = (1440*7) # 1440 one day. last val: 1440*4
        self.prev_measured_bat_level = self.int_bat_level
        self.charging_power = 0
        self.prev_charging_power_meas_time_est = 0
        self.est_cp_time_dict = {}
        self.window_state_flag = 0
        self.pulse_start_time_calculated_flag = False
        self.pulse_starting_time = None
        self.start_interruption = None
        self.interruption_duration = None
        self.dc_opt = 0.35
        self.cfg_power_opt = 0
        self.calc_cfg_once_flag = 0 # This flag is riased after calculating the opt cfg in the no charging case, so it will be calculated only once.
        self.prev_max_current = None
        self.min_time_between_cfg_change = 500
        self.hyst_state_flag = 0 # 1 for upper bound was achieved, 0 for lower bound was achieved
        self.change_cfg_command_time = -self.min_time_between_cfg_change
        self.time_axis_bat_sim = []
        self.time_axis_avg_current = []
        #Debugging:
        self.avg_current_opt_list = []
        self.max_current_list = []
        self.dc_opt_list = []
        self.cfg_power_opt_list = []
        self.measured_bat_level_list = []
        self.bat_level_est_list = []
        self.time_axis_bat_meas_est = []

    
    def power_optimization_test(self): #DB: run_power_optimization
        self.load_csv_data_base()
        self.match_const_val_to_board_type()
        self.execute_test()

    def load_csv_data_base(self):
        self.data_df = pd.read_csv(BOARDS_DATA_CSV_PATH)
        self.data_df = self.data_df[self.data_df['bridge_type'] == self.bridge_type]

    def match_const_val_to_board_type(self):
        try:
            bridge_type = BridgeType[self.bridge_type.upper()]
            bridge_data = bridge_type.value
            self.avg_beacon_current = bridge_data["avg_beacon_current"]
            self.beacon_tx_time = bridge_data["beacon_tx_time"]
            self.avg_rx_current = bridge_data["avg_rx_current"]
            self.board_max_output_power = bridge_data["board_max_output_power"]
            self.lower_bound_output_power_cfg = bridge_data["lower_bound_output_power_cfg"]
            self.upper_bound_output_power_cfg = bridge_data["upper_bound_output_power_cfg"]
        except KeyError:
            print(f"Invalid bridge type: {self.bridge_type}. Please choose from the following bridge types: {', '.join(member.name.lower() for member in BridgeType)}.")
            sys.exit(1) 
   
            
    def generate_window_charging_func(self, zero_charging_time_minutes, const_charging_time_minutes, const_charging_current):
        total_cycle_time = zero_charging_time_minutes + const_charging_time_minutes
        current_time_in_cycle = self.elapsed_time_minutes % total_cycle_time

        if current_time_in_cycle < zero_charging_time_minutes:
            # Zero charging phase
            return 0
        else:
            # Constant charging phase
            return const_charging_current

    def step_charging_func(self, pulse_duration, pulse_charging_current):
        earliest_pulse_start = 0.1 * self.sim_time * 60
        latest_pulse_start = 0.4 * self.sim_time * 60

        if self.pulse_start_time_calculated_flag == False:
            self.pulse_starting_time = random.randint(earliest_pulse_start, latest_pulse_start)
            self.pulse_start_time_calculated_flag = True

        if (self.elapsed_time_minutes >= self.pulse_starting_time) and (self.elapsed_time_minutes <= self.pulse_starting_time + pulse_duration):
            return pulse_charging_current
        else:
            return 0
        
    def calc_interrupt_param(self):
        earliest_interruption_start = 24 * 60
        latest_interruption_start = self.sim_time * 60 - (24 * 60)
        min_interruption_duration = 6 * 60
        max_interruption_duration = 24 * 60
        
        self.start_interruption = random.randint(earliest_interruption_start, latest_interruption_start)
        self.interruption_duration = random.randint(min_interruption_duration, max_interruption_duration)

    def bat_level_sim(self):
         # determine the charging current, the charging voltage is const:
        if self.charging_func == "sin":
            charging_current = max(math.sin(self.elapsed_time_minutes * 0.0045) * 200, 0)
        elif self.charging_func == "gauss":
            charging_current = max(random.gauss(60,100), 0)
        elif self.charging_func == "const":
            charging_current = self.constant_charging_current
        elif self.charging_func == "window":
             charging_current = self.generate_window_charging_func(1000, 440, 220)
        elif self.charging_func == "pulse":
            charging_current = self.step_charging_func(pulse_duration = 8 * 60, pulse_charging_current = 200)

        if self.interrupt_charging and (self.elapsed_time_minutes >= self.start_interruption and self.elapsed_time_minutes <= (self.start_interruption + self.interruption_duration)):
            charging_current = 0

        self.charging_power = self.charging_voltage * (charging_current/1000) ### This claims that the voltage is contant and only the current changes
        charging_cap = self.charging_power * (1 / 60)
        # self.avg_current_opt = self.calc_avg_current(12, 0.3)
        self.avg_current_opt = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
        discharging_cap = self.board_voltage * (self.avg_current_opt / 1000) * (1 / 60)
        prev_bat_cap = self.bat_cap_wh * (self.bat_level / 100)
        curr_bat_cap = min((prev_bat_cap - discharging_cap + charging_cap), self.bat_cap_wh)
        self.bat_level = max((curr_bat_cap / self.bat_cap_wh) * 100, 0)
        # to plot the battery level
        self.time_axis_bat_sim.append(self.elapsed_time_minutes)
        self.bat_level_list.append(self.bat_level)
        self.charging_current.append(charging_current)

    def charging_power_filter(self):
         self.time_axis_charging_power.append(self.elapsed_time_minutes)
         self.charging_power_list.append(self.charging_power)

        ##### To add time window for the charging power, uncomment the code below #####
         
        #  if len(self.charging_power_list) >= math.floor(self.time_window_est_filter / self.sampling_period): #There is a sample every sampling period, so this condition is valid if we have more samples than the window
        #          samples_in_time_window = self.charging_power_list[-math.floor(self.time_window_est_filter / self.sampling_period):]                           
        #          avg_charging_power = statistics.mean(samples_in_time_window)
        #          self.charging_power_avg_list.append(avg_charging_power)
        #          self.time_axis_filter.append(self.elapsed_time_minutes)

        #  if len(self.charging_power_list) < self.time_window_est_filter / self.sampling_period and len(self.charging_power_list) >= 1:
         
         avg_charging_power = statistics.mean(self.charging_power_list)
         self.charging_power_avg_list.append(avg_charging_power)
         self.time_axis_filter.append(self.elapsed_time_minutes)


    def charging_estimator(self): 
            discharging_cap = self.board_voltage * (self.avg_current_opt / 1000) * (self.sampling_period / 60) # In Wh
            prev_bat_cap_est = self.bat_cap_wh * (self.bat_level_est / 100)
            curr_bat_cap_est = prev_bat_cap_est - discharging_cap
            self.bat_level_est = (curr_bat_cap_est / self.bat_cap_wh) * 100 # Consider plotting this as well
            curr_measured_bat_level = None

            if round(self.bat_level, 0) % 10 == 0: 
                curr_measured_bat_level = round(self.bat_level, self.bat_level_meas_resolution)  

                if (curr_measured_bat_level != self.prev_measured_bat_level) and (curr_measured_bat_level > self.bat_level_est): # only when there is a change in the battery level, refresh the estimation
                    time_interval_samples = self.elapsed_time_minutes - self.prev_charging_power_meas_time_est
                    self.charging_power_est = (((curr_measured_bat_level - self.bat_level_est) / 100) * self.bat_cap_wh) / (time_interval_samples / 60) # divide this by the time interval from the last measurment. This assumes a constant charging power between measurments
                    # the dict holdes the time in minutes for every estimated charging power, and the time interval of this sample. this is used to cut relevent samples for the time window filtering and weighted average according to the interval length
                    self.est_cp_time_dict[self.elapsed_time_minutes] = [self.charging_power_est, time_interval_samples] # This assumes that the charging power is constant between two samples
                    self.bat_level_est = curr_measured_bat_level 
                    self.prev_charging_power_meas_time_est = self.elapsed_time_minutes 

                    self.charging_power_list_est.append(self.charging_power_est)
                    self.time_axis_est.append(self.elapsed_time_minutes)                 
                    
                if self.bat_level_est > curr_measured_bat_level:
                    # since the estimated bat_level cant be larger than the real one (the estimated is calculated with 0 charging power)
                    self.bat_level_est = curr_measured_bat_level
                
            self.prev_measured_bat_level = curr_measured_bat_level

            self.measured_bat_level_list.append(self.prev_measured_bat_level)
            self.bat_level_est_list.append(self.bat_level_est)
            self.time_axis_bat_meas_est.append(self.elapsed_time_minutes)


    def estimator_filter(self):
            time_weighted_cp_sum = 0
            total_time = 0

            if self.elapsed_time_minutes >= self.time_window_est_filter and len(self.est_cp_time_dict) >= 1:
                 
                 threshold_time = self.elapsed_time_minutes - self.time_window_est_filter
                 cp_samples_in_window_list = [cp_time_interval_list_est for time, cp_time_interval_list_est in self.est_cp_time_dict.items() if time > threshold_time]
                 # a list where each elements is a list where the first elements is the charging power and second element is the time interval from the last sample.
                 if len(cp_samples_in_window_list) >= 1:
                    for cp_time_pair in cp_samples_in_window_list:
                        cp, time_interval = cp_time_pair
                        time_weighted_cp_sum += cp * time_interval
                        total_time += time_interval

                    self.filtered_charging_power_est = time_weighted_cp_sum / total_time 
                    self.filtered_charging_power_list_est.append(self.filtered_charging_power_est)
                    self.time_axis_filter_est.append(self.elapsed_time_minutes)

            if self.elapsed_time_minutes < self.time_window_est_filter and len(self.est_cp_time_dict) >= 1:

                for cp_time_pair in self.est_cp_time_dict.values():
                    cp, time_interval = cp_time_pair
                    time_weighted_cp_sum += cp * time_interval
                    total_time += time_interval

                self.filtered_charging_power_est = time_weighted_cp_sum / total_time 
                self.filtered_charging_power_list_est.append(self.filtered_charging_power_est)
                self.time_axis_filter_est.append(self.elapsed_time_minutes)                 


    def calc_avg_current(self, cfg_power, dc_sub1g):
        cfg_power = round(cfg_power)
        x_current = self.data_df[self.data_df['PA'] == 1] # PA is always engaged due to gain loss.
        x_current = x_current.loc[x_current['cfg_power'] == cfg_power, 'X_current'].values[0]

        avg_current = round(
            dc_sub1g * x_current +
            self.avg_beacon_current * (self.beacon_tx_time / self.rxtx_period_beacons) +
            self.avg_rx_current * ((self.rxtx_period_beacons - self.beacon_tx_time) /self.rxtx_period_beacons), 4)
        return round(avg_current, 2)
    

    def calc_relative_gain(self, cfg_power, dc):
        cfg_power = round(cfg_power)
        real_output_power = self.data_df[self.data_df['PA'] == 1]
        real_output_power = real_output_power.loc[real_output_power['cfg_power'] == cfg_power, 'output_power'].values[0]
        relative_gain = round(real_output_power - self.board_max_output_power  + 10 * math.log10(dc / 0.33), 1)
        return relative_gain
        

    def update_opt_cfg(self,max_current):
        cfg_power_max = self.upper_bound_output_power_cfg
        cfg_power_min = self.lower_bound_output_power_cfg
        relative_gain_prev = -100
        avg_current_prev = 10000
        prev_cfg_power = self.cfg_power_opt
        prev_cfg_dc = self.dc_opt
        new_cfg_power = self.cfg_power_opt
        new_dc = self.dc_opt

        if (self.elapsed_time_minutes < self.change_cfg_command_time + self.min_time_between_cfg_change) and (self.charging_func != "const"): 
            return
        
        for power_cfg in range(cfg_power_min, cfg_power_max + 1, 1):
            for dc in range(5, 61, 5):
                avg_current = self.calc_avg_current(power_cfg, dc/100)
                relative_gain = self.calc_relative_gain(power_cfg, dc/100)
                if avg_current <= max_current:
                    if  relative_gain > relative_gain_prev or (relative_gain == relative_gain_prev and avg_current < avg_current_prev):
                        avg_current_prev = avg_current
                        relative_gain_prev = relative_gain
                        new_cfg_power = power_cfg
                        new_dc = dc/100
        if new_dc != prev_cfg_dc or new_cfg_power != prev_cfg_power:
            self.change_cfg_command_time = self.elapsed_time_minutes
            self.avg_current_opt = self.calc_avg_current(new_cfg_power, new_dc)
            self.dc_opt = new_dc
            self.cfg_power_opt = new_cfg_power
 

    def optimal_cfg(self):
        max_current = 0
        max_allowed_current = 300
        net_power = 0
        max_current_multiplier_boost = 2
        max_current_lower_bound = 30
        max_current_upper_bound = 200

        if not(self.inf_time_mode):
            time_target_mins = self.time_target_hr * 60
            time_to_last_min = (time_target_mins) - self.elapsed_time_minutes
            if time_to_last_min <= 0:
                time_to_last_min = self.elapsed_time_minutes + (24*60)

            if len(self.filtered_charging_power_list_est) < 1 and (self.charging_func not in ["const", "zero"]) :
                self.dc_opt = 0.3
                self.cfg_power_opt = self.upper_bound_output_power_cfg
                self.avg_current_opt = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
            else:
                avg_current_mA = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
                net_power = self.filtered_charging_power_est - ((avg_current_mA / 1000) * self.board_voltage) # positive for charging, negative for discharging
            if self.charging_func == "const":
                avg_current_mA = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
                net_power = (self.constant_charging_current/1000) * self.charging_voltage - ((avg_current_mA / 1000) * self.board_voltage)

            measured_bat_level = round((round(self.bat_level, self.bat_level_meas_resolution) / 5)) * 5 
            # to get the nearest mulitple of 5 of the battery level
            remaining_bat_cap = (measured_bat_level / 100) * self.bat_cap_wh 
            expected_final_bat_cap = remaining_bat_cap + net_power * (time_to_last_min / 60)
            if (expected_final_bat_cap < (self.bat_cap_wh * 0.03) or expected_final_bat_cap > (self.bat_cap_wh * 0.05)) and not(self.calc_cfg_once_flag) and len(self.filtered_charging_power_list_est) > 1:
            # if the battery is not going to last the equired time, or the cfg does not yield the max possible current, calculate the maximum current such that the capacity will be between 0 and 2% and then find the matching configuration. 
                max_current = min(round((remaining_bat_cap / ((time_to_last_min / 60) * self.board_voltage)) * 1000, 0), max_allowed_current) + self.constant_charging_current * (self.charging_voltage/self.board_voltage)
                remaining_time_percent = round(time_to_last_min / time_target_mins, 2) * 100
                
                if measured_bat_level >= remaining_time_percent:
                    bat_diff_percent = measured_bat_level - remaining_time_percent
                    max_current = (max_current + (((bat_diff_percent/100) * self.bat_cap_wh) / ((time_to_last_min / 60) * self.board_voltage)) * 1000) * max_current_multiplier_boost
                    if net_power > 0:
                        max_current = max_current + (net_power / self.board_voltage) * 1000

                max_current = round(max_current)
                max_current = max(max_current, max_current_lower_bound)
                max_current = min(max_current, max_current_upper_bound)
                if max_current != self.prev_max_current:
                    if max_current > self.avg_current_opt + 50:
                        max_current = self.avg_current_opt + 50
                    if max_current < self.avg_current_opt - 50:
                        max_current = self.avg_current_opt - 50
                    self.update_opt_cfg(max_current)

                if self.charging_func == "zero" or self.charging_func == "const":
                    self.calc_cfg_once_flag = 1

                self.prev_max_current = max_current
            
        # infinite time control: the device has no battery changing deadline.
        if self.inf_time_mode:
            hyst_lower_bound = 20
            hyst_upper_bound = 80
            current_decrease_lower_hyst = 0.8
            current_boost_upper_hyst = 1.5

            if len(self.filtered_charging_power_list_est) < 1 and (self.charging_func not in ["const", "zero"]):
                self.dc_opt = 0.3
                self.cfg_power_opt = self.upper_bound_output_power_cfg
                self.avg_current_opt = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
            else:
                avg_current_mA = self.calc_avg_current(self.cfg_power_opt, self.dc_opt)
                measured_bat_level = round((round(self.bat_level, self.bat_level_meas_resolution) / 5)) * 5 
                max_current = round((self.filtered_charging_power_est / self.board_voltage) * 1000)

                if  measured_bat_level <= hyst_lower_bound:
                    max_current = max_current * current_decrease_lower_hyst
                    # self.hyst_state_flag = 0
                if  measured_bat_level >= hyst_upper_bound:
                    max_current = max_current * current_boost_upper_hyst
                    # self.hyst_state_flag = 1

                max_current = max(max_current, max_current_lower_bound)
                max_current = min(max_current, max_current_upper_bound)
                if max_current != self.prev_max_current:
                    if max_current > self.avg_current_opt + 50:
                        max_current = self.avg_current_opt + 50
                    if max_current < self.avg_current_opt - 50:
                        max_current = self.avg_current_opt - 50
                    self.update_opt_cfg(max_current)
            self.prev_max_current = max_current

        # plot the avg current
        self.max_current_list.append(max_current)
        self.avg_current_opt_list.append(self.avg_current_opt)  
        self.time_axis_avg_current.append(self.elapsed_time_minutes) 
        self.dc_opt_list.append(self.dc_opt)
        self.cfg_power_opt_list.append(self.cfg_power_opt) 


    def plot_charging_data(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_axis_est, y=self.charging_power_list_est, mode='lines', name='EST charging capacity', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=self.time_axis_charging_power, y=self.charging_power_list, mode='lines', name='ACTUAL charging capacity', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.time_axis_bat_sim, y=self.bat_level_list, mode='lines', name='Actual battery level [%]', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=self.time_axis_avg_current, y=self.avg_current_opt_list, mode='lines', name='avg discharge current [mA]', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.time_axis_avg_current, y=self.max_current_list, mode='lines', name='max current', line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=self.time_axis_avg_current, y=self.dc_opt_list , mode='lines', name='Duty Cycle', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.time_axis_avg_current, y=self.cfg_power_opt_list , mode='lines', name='cfg power', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.time_axis_filter_est, y=self.filtered_charging_power_list_est, mode='lines', name='EST avg charging capacity', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.time_axis_bat_meas_est, y=self.measured_bat_level_list, mode='lines', name='Quantisized bat level', line=dict(color='magenta')))
        fig.add_trace(go.Scatter(x=self.time_axis_bat_meas_est, y=self.bat_level_est_list, mode='lines', name='Estimated batter level [%]', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=self.time_axis_bat_sim, y=self.charging_current, mode='lines', name='charging current', line=dict(color='magenta')))
        fig.update_layout(
            xaxis_title='elapsed time [minutes]',
            yaxis_title='AU',
            title='Actual vs EST charging power for: '+ self.charging_func  +' charging function',
            legend=dict(
                        x=1,  
                        y=1,    
                        traceorder='normal',

                        font=dict(
                        family='sans-serif',
                        size=12,  # Change the size to your desired value
                        color='black'
                        ),
            )
        )
        fig.show()


    def plot_avg_filter_performance(self):
        error_percentage_list = []
        for i in range(len(self.time_axis_filter_est)):
            for j in range(len(self.time_axis_charging_power)):
                if self.time_axis_filter_est[i] == self.time_axis_charging_power[j]:
                    if self.charging_power_avg_list[j] == 0:
                        if self.filtered_charging_power_list_est[i] > 0.2:
                            error_percentage_list.append(-1)
                        # When the actual charging is zero we can't calculate the error so if the est value is not negligble, consider this as -1 so we can note this in the graph
                        else:
                            error_percentage_list.append(0)
                    if self.charging_power_avg_list[j] != 0:
                        error_percentage = (abs(self.charging_power_avg_list[j] - self.filtered_charging_power_list_est[i]) / self.charging_power_avg_list[j]) * 100
                        error_percentage_list.append(error_percentage)
                    break       

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.time_axis_filter_est, y=self.filtered_charging_power_list_est, mode='lines', name='EST avg charging capacity', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=self.time_axis_charging_power, y=self.charging_power_avg_list, mode='lines', name='ACTUAL avg charging capacity', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.time_axis_filter_est, y=error_percentage_list, mode='lines', name='Estimator error (%)', line=dict(color='red')))

        Avg_error_percentage = statistics.mean(error_percentage_list)
        fig.add_annotation(
            text=f'Average error: {Avg_error_percentage:.2f} %',
            showarrow=False,
            xref='paper',  
            yref='paper',  
            x=0.5,  
            y=0.6,   
            font=dict(color='black', size=14)
        )
        fig.update_layout(
            xaxis_title='elapsed time [minutes]',
            yaxis_title='%',
            title='Actual vs EST AVG charging power for: '+ self.charging_func +' charging function'
                #         legend=dict(
                #         x=0.83,  
                #         y=0.6,    
                #         traceorder='normal',

                #         font=dict(
                #         family='sans-serif',
                #         size=12,  # Change the size to your desired value
                #         color='black'
                #         ),
                #  )
        )
        fig.show()

    def execute_test(self):
        if self.interrupt_charging:
            self.calc_interrupt_param()
            print(f"start interruption: {self.start_interruption}")
            print(f"end of interruption: {self.start_interruption + self.interruption_duration}")
        while self.elapsed_time_minutes < self.sim_time * 60: # This will be DB job handeled
            self.bat_level_sim()
            if self.elapsed_time_minutes % self.sampling_period == 0: 
                self.charging_power_filter()
                self.charging_estimator()
                self.estimator_filter() 
                self.optimal_cfg()   
            self.elapsed_time_minutes += 1 
        self.plot_charging_data()
        self.plot_avg_filter_performance()

       



        


        








 

