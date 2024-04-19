import pandas as pd
import matplotlib.pyplot as plt

db = pd.read_csv("household_power_consumption.txt", delimiter=";") 

not_needed = ["Date","Time","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"] 

  

db["Datetime"] = (db['Date'].astype(str) +","+ db['Time'].astype(str)).astype(str) 

db["Datetime"] = pd.to_datetime(db["Datetime"], format="%d/%m/%Y,%H:%M:%S") 

db = db.drop(not_needed, axis=1) 

  

db.replace('?', float(0), inplace=True) #float("NaN")

  

db["Power"] = db["Global_active_power"].astype(float)#.interpolate()
db = db.dropna()

  

db = db.drop("Global_active_power", axis=1) 

  

db = db.set_index(pd.DatetimeIndex(db["Datetime"])).drop("Datetime",axis=1) 

db = db.resample("15T").mean() 


plt.figure(figsize=(12,8))
plt.title("Household Consumtion Interpolated over 15 Min")
plt.plot(db.index, db["Power"])


db.to_csv("Modified_Dataset_Null_as_zero.csv")  

print(db.info()) 