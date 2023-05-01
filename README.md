# Hybrid-Power-Source
A controller software controlling a hybrid system: supercapacitor and battery. The goal of this project is to have a software runned in the Simulink. 

1. Maintain
Real-time monitoring and control of the battery system, allowing users to track the state of charge, state of health, and other important parameters of the hybrid (supercapacitor/ battery).

2. Protect
Ability to set and adjust charging and discharging profiles, ensuring optimal performance and safety of the hybrid system.
Robust security features, protecting the hybrid system from cyber threats and unauthorized access.

3. Prolong
Predicting the behavior of the hybrid by analysising the data. 

The following code is the prototype of the controller software. However, there are bugs needed to be fixed. 

import time
import random

class BatterySystem:
    def __init__(self):
        self.state_of_charge = 0.0
        self.state_of_health = 1.0
        self.charging_profile = []
        self.discharging_profile = []

    def set_charging_profile(self, profile):
        self.charging_profile = profile

    def set_discharging_profile(self, profile):
        self.discharging_profile = profile

    def update_state_of_charge(self):
        # Simulate charging and discharging
        if self.charging_profile and self.state_of_charge < 1.0:
            self.state_of_charge += self.charging_profile.pop(0)
        elif self.discharging_profile and self.state_of_charge > 0.0:
            self.state_of_charge -= self.discharging_profile.pop(0)
        else:
            # No charging or discharging, state of charge remains the same
            pass

    def update_state_of_health(self):
        # Simulate gradual degradation
        self.state_of_health -= random.uniform(0.0, 0.01)

class BatterySystemController:
    def __init__(self, battery_system):
        self.battery_system = battery_system
        self.security_key = '1234'

    def monitor_battery_system(self):
        # Real-time monitoring of battery system parameters
        while True:
            print("State of charge: {:.2f}".format(self.battery_system.state_of_charge))
            print("State of health: {:.2f}".format(self.battery_system.state_of_health))
            time.sleep(5)

    def set_charging_profile(self, profile, key):
        # Check security key before allowing changes to charging profile
        if key == self.security_key:
            self.battery_system.set_charging_profile(profile)
        else:
            print("Unauthorized access!")

    def set_discharging_profile(self, profile, key):
        # Check security key before allowing changes to discharging profile
        if key == self.security_key:
            self.battery_system.set_discharging_profile(profile)
        else:
            print("Unauthorized access!")

    def update_battery_system(self):
        # Regularly update state of charge and state of health
        while True:
            self.battery_system.update_state_of_charge()
            self.battery_system.update_state_of_health()
            time.sleep(1)

battery_system = BatterySystem()
controller = BatterySystemController(battery_system)

# Start monitoring and control threads
monitor_thread = threading.Thread(target=controller.monitor_battery_system)
monitor_thread.start()

update_thread = threading.Thread(target=controller.update_battery_system)
update_thread.start()

# Set charging and discharging profiles (requires security key)
controller.set_charging_profile([0.1, 0.2, 0.3], '1234')
controller.set_discharging_profile([0.2, 0.3, 0.1], '1234')


4. Cyber Security (Additional function to the software). This is a prototype

function [SOC_ref, ChargePower_ref, DischargePower_ref] = battery_controller(SOC, SOC_min, SOC_max, SOC_target, ChargePower_max, DischargePower_max)
% Inputs:
% - SOC: current state of charge of the battery
% - SOC_min: minimum allowable state of charge
% - SOC_max: maximum allowable state of charge
% - SOC_target: desired target state of charge
% - ChargePower_max: maximum allowable charging power
% - DischargePower_max: maximum allowable discharging power
% Outputs:
% - SOC_ref: reference state of charge
% - ChargePower_ref: reference charging power
% - DischargePower_ref: reference discharging power

% Controller gains
Kp = 0.2;
Ki = 0.05;
Kd = 0.1;

% Controller limits
ChargePower_min = 0;
DischargePower_min = 0;

% Calculate error and error derivatives
error = SOC_target - SOC;
d_error = error - SOC_ref;

% Integrate error
error_int = error_int + error;

% Calculate control outputs
ChargePower_ref = Kp*error + Ki*error_int + Kd*d_error;
DischargePower_ref = -ChargePower_ref;

% Apply limits
ChargePower_ref = max(ChargePower_min, min(ChargePower_ref, ChargePower_max));
DischargePower_ref = max(DischargePower_min, min(DischargePower_ref, DischargePower_max));

% Calculate reference SOC
SOC_ref = SOC + (ChargePower_ref - DischargePower_ref)/capacity;
SOC_ref = max(SOC_min, min(SOC_ref, SOC_max));

end



Another Prototype

import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)  # replace with the serial port and baud rate of your device

while True:
    data = ser.readline().decode('utf-8').rstrip()  # read the data from the serial port and decode it
    # assuming the data is in the format "voltage, current"
    voltage, current = data.split(',')
    voltage = float(voltage)
    current = float(current)
    
    # insert your control algorithm here
    # for example, adjust the charging and discharging profiles based on the voltage and current readings
    
    print(f"Voltage: {voltage:.2f} V, Current: {current:.2f} A")
batteries = ['sodium-ion', 'aluminum-ion', 'solid state']
supercapacitors = ['type A', 'type B', 'type C']

def check_compatibility(battery_type, supercap_type):
    if battery_type in batteries and supercap_type in supercapacitors:
        return True
    else:
        return False

# Example usage
if check_compatibility('sodium-ion', 'type A'):
    print('Battery and supercapacitor are compatible')
else:
    print('Battery and supercapacitor are not compatible')
import time

# Define the function to get the state of charge (SOC)
def get_SOC():
    # Placeholder code to get the SOC from the hybrid energy storage system
    SOC = 80  # Example value
    return SOC

# Define the function to get the state of health (SOH)
def get_SOH():
    # Placeholder code to get the SOH from the hybrid energy storage system
    SOH = 90  # Example value
    return SOH

# Loop to continuously display the SOC and SOH
while True:
    # Get the SOC and SOH values
    SOC = get_SOC()
    SOH = get_SOH()
    
    # Display the SOC and SOH values
    print("State of Charge (SOC): {}%".format(SOC))
    print("State of Health (SOH): {}%".format(SOH))
    
    # Wait for 1 second before getting the next values
    time.sleep(1)
# Assume we have a hybrid energy storage system with a battery and supercapacitor
# We will set the charging and discharging profiles for the battery

# Set the charging profile for the battery
battery_charging_profile = {
    "voltage_limit": 4.2,  # Maximum voltage limit for the battery during charging
    "current_limit": 10,  # Maximum current limit for the battery during charging
    "charge_time": 2,  # Amount of time the battery will be charged
}

# Set the discharging profile for the battery
battery_discharging_profile = {
    "voltage_limit": 3.2,  # Minimum voltage limit for the battery during discharging
    "current_limit": 15,  # Maximum current limit for the battery during discharging
    "discharge_time": 4,  # Amount of time the battery will be discharged
}

# Use the profiles to charge and discharge the battery
def charge_battery():
    # Code to charge the battery according to the charging profile

def discharge_battery():
    # Code to discharge the battery according to the discharging profile
import numpy as np
from sklearn.linear_model import LinearRegression

# Assume battery data is already collected and stored in arrays

# Use linear regression to predict battery behavior and estimate remaining life
X = np.array([charging_time, discharging_time, temperature, age]).T
y = np.array(battery_health)
model = LinearRegression()
model.fit(X, y)

# Use the model to predict battery behavior and estimate remaining life
prediction = model.predict([[new_charging_time, new_discharging_time, new_temperature, new_age]])
import pandas as pd

# Assume energy consumption data is already collected and stored in a dataframe

# Use time series forecasting to predict energy demand and optimize supply
model = ARIMA(df['energy_consumption'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=24)

# Use the forecast to optimize energy supply and balance demand
supply = optimize_supply(forecast)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Assume energy consumption and production data is already collected and stored in dataframes

# Preprocess the data using scaling and principal component analysis (PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pd.concat([df_consumption, df_production], axis=1))
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Use K-means clustering to group similar data points and identify patterns
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(pca_data)
labels = kmeans.predict(pca_data)

# Use the clustering results to improve productivity, safety, sustainability, and reduce costs
improve_productivity(labels)
improve_safety(labels)
improve_sustainability(labels)
reduce_costs(labels)
