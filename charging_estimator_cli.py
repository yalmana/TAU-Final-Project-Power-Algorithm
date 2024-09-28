from argparse import ArgumentParser
from wiliot_deployment_tools.power_optimization_tool.charging_estimator import ChargingEstimatorSimulator

def main():
    parser = ArgumentParser(description="plot the simulated charging regime and the estimated charging regime")
    required = parser.add_argument_group('required arguments')
    required.add_argument("-bridge_type", type=str, help="Bridge type (minew or fanstel)", required=True)
    required.add_argument("-rxtx_period", type=int, help="The 2.4Ghz BLE (beacons) rxtx period in mili-sec", required=True)
    required.add_argument("-sim_time", type=int, help="How long the simulation will be in hours", required=True)
    required.add_argument("-dis_volt", type=float, help="The discharge voltage of the battery (supplied voltage to device) in Volts", required=True)
    required.add_argument("-charge_volt", type=float, help="The charging voltage of the battery", required=True)
    required.add_argument("-sam_period", type=int, help="The sampling peroid of battery level in minutes", required=True)
    required.add_argument("-charging_func", type=str, help="The time dependent charing function that will be used", required=True)
    required.add_argument("-bat_level_meas_res", type=int, help="The resolution in which the battery level will be sampled. The value entred determines the number of digits after the decimal point", required=True)
    parser.add_argument("-time_to_last", type=int, help="How much the user wishs the battery will last in hr", default=1)
    parser.add_argument("-inf_time_mode", type=bool, help="In infinite time mode, the algorithm will match the bridge power consumption to the chraging power in order to allow inf. usage time.", default=False)
    parser.add_argument("-bat_cap", type=int, help="The capacity of the connected battery in W/hr. default is the koamtac's samsung battery", default=15.21) # KOAMTAC battery 15.21Wh, Voltaic V75 battery 72Wh
    parser.add_argument("-int_bat_level", type=int, help="The initial battery level in %", default=100)
    parser.add_argument("-const_charge_current", type=int, help="If you choose constant charging function, enter here the charging current in mA", default=0)
    parser.add_argument("-interrupt_charging", type=bool, help="interrupt the charging (zero charging current), at a random time for a random duration", default=False)
    
    args = parser.parse_args()

    c = ChargingEstimatorSimulator(args.bridge_type, args.rxtx_period, args.sim_time, args.bat_cap, args.int_bat_level, args.dis_volt, args.charge_volt, args.sam_period, args.charging_func, args.const_charge_current, args.bat_level_meas_res, args.time_to_last, args.inf_time_mode, args.interrupt_charging)
    c.power_optimization_test()
    
def main_cli():
    main()

if __name__ == '__main__':
    main()