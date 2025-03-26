# rFpro (2024b.1) auto generated test script [25/03/2025 12:15:04]
import os

import clr  # pythonnet
import time

# Load the rFpro.Controller DLL/Assembly
clr.AddReference("C:/rFpro/2023b/API/rFproControllerExamples/Controller/rFpro.Controller")

# Import rFpro.controller and some other helpful .NET objects
from rFpro import Controller
from System import DateTime, Decimal

# Create an instance of the rFpro.Controller
rFpro = Controller.DeserializeFromFile('LA_Remote.json')
print(rFpro)

while (rFpro.NodeStatus.NumAlive < rFpro.NodeStatus.NumListeners):
    print(rFpro.NodeStatus.NumAlive, ' of ', rFpro.NodeStatus.NumListeners, ' Listeners connected.')
    time.sleep(1)
print(rFpro.NodeStatus.NumAlive, ' of ', rFpro.NodeStatus.NumListeners, ' Listeners connected.')

rFpro.DynamicWeatherEnabled = True
rFpro.Camera = 'TVCockpit'
rFpro.ParkedTrafficDensity = Decimal(0.5)

fog_level = 0.0
rFpro.Vehicle = 'SaloonBlack_RH'
rFpro.Location = 'LA_Road_Loop'
rFpro.StartTime = DateTime.Parse('2025-02-03T11:00:00')
rFpro.VehiclePlugin = 'TorrenceCrossroads_Hosted'

rFpro.StartSession()
time.sleep(30)
previous_time = 0
ascending = True
while rFpro.NodeStatus.PhysicsStatus == "Running":
    temp_time = rFpro.ElapsedTime
    if abs(temp_time - previous_time) >= 2.0:
        previous_time = temp_time
        if fog_level >= 0.05:
            ascending = False
        elif fog_level < 0:
            break
        if ascending:
            fog_level += 0.0005
        else:
            fog_level -= 0.0005
        print(f'\rChanging fog to {fog_level}', end="")
        rFpro.Fog = fog_level
        time.sleep(1)

rFpro.StopSession()
