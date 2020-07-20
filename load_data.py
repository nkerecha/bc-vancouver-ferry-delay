import numpy as np
import pandas as pd
import pathlib
import os
import datetime
import time

from sklearn.preprocessing import Imputer

STITCH_DATA = False

def get_data():
    if not STITCH_DATA:
        train_data = load_data("Data/train.csv", 0)
        train_data = clean_trips(train_data)
        train_data = clean_date_time(train_data)
        train_data = clean_status(train_data)
        train_data = clean_vessels(train_data)
        train_data.pop('Trip.Duration')
    elif(os.path.isfile("Data/clean_train.csv")):
        train_data = load_data("Data/clean_train.csv", 0)
    else:
        traffic_dataset = load_data("Data/traffic.csv", 0)
        vancouver_dataset = load_data("Data/vancouver.csv", 0)
        victoria_dataset = load_data("Data/victoria.csv", 0)
        train_data = load_data("Data/train.csv", 0)
        train_data = clean_trips(train_data)
        train_data = clean_date_time(train_data)
        train_data = clean_status(train_data)
        train_data = clean_vessels(train_data)
        train_data = stitch_traffic(train_data, traffic_dataset)
        #train_data = stitch_weather(train_data, vancouver_dataset, "vancouver")
        #train_data = stitch_weather(train_data, victoria_dataset, "victoria")
        train_data.pop('Trip.Duration')
        train_data.to_csv("Data/clean_train.csv")

    if not STITCH_DATA:
        test_data = load_data("Data/test.csv", 0)
        test_data = clean_trips(test_data)
        test_data = clean_date_time(test_data)
        test_data = clean_vessels(test_data)
    elif(os.path.isfile("Data/clean_test.csv")):
        test_data = load_data("Data/clean_test.csv", 0)
    else:
        test_data = load_data("Data/test.csv", 0)
        test_data = clean_trips(test_data)
        test_data = clean_date_time(test_data)
        test_data = clean_vessels(test_data)
        test_data = stitch_traffic(test_data, traffic_dataset)
        #test_data = stitch_weather(test_data, vancouver_dataset, "vancouver")
        #test_data = stitch_weather(test_data, victoria_dataset, "victoria")
        test_data.to_csv("Data/clean_test.csv")

    return train_data, test_data

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def clean_trips(data, train=True):
    trip = data.pop('Trip')
    data['Tsawwassen to Swartz Bay'] = (trip == 'Tsawwassen to Swartz Bay')*1.0
    data['Tsawwassen to Duke Point'] = (trip == 'Tsawwassen to Duke Point')*1.0
    data['Swartz Bay to Fulford Harbour (Saltspring Is.)'] = (trip == 'Swartz Bay to Fulford Harbour (Saltspring Is.)')*1.0
    data['Swartz Bay to Tsawwassen'] = (trip == 'Swartz Bay to Tsawwassen')*1.0
    data['Duke Point to Tsawwassen'] = (trip == 'Duke Point to Tsawwassen')*1.0
    data['Departure Bay to Horseshoe Bay'] = (trip == 'Departure Bay to Horseshoe Bay')*1.0
    data['Horseshoe Bay to Snug Cove (Bowen Is.)'] = (trip == 'Horseshoe Bay to Snug Cove (Bowen Is.)')*1.0
    data['Horseshoe Bay to Departure Bay'] = (trip == 'Horseshoe Bay to Departure Bay')*1.0
    data['Horseshoe Bay to Langdale'] = (trip == 'Horseshoe Bay to Langdale')*1.0
    data['Langdale to Horseshoe Bay'] = (trip == 'Langdale to Horseshoe Bay')*1.0
    return data

def clean_date_time(data):
    dates = []
    for date in data['Full.Date']:
        dates.append(datetime.datetime.strptime(date, '%d %B %Y').timestamp())

    data.insert(1, 'timestamp', dates)
    data = data.drop(columns=['Month','Day.of.Month','Year','Full.Date'])
    
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    days = []
    for day in data['Day']:
        days.append(day_names.index(day))

    data = data.drop(columns=['Day'])
    data.insert(1, 'Day', days)
    
    midnight = datetime.datetime.strptime("12:00 AM", "%I:%M %p")
    times = []
    for time in data['Scheduled.Departure']:
        times.append(datetime.datetime.strptime(time, "%I:%M %p").timestamp() - midnight.timestamp())
    
    data = data.drop(columns=['Scheduled.Departure'])
    data.insert(1, 'Scheduled.Departure', times)

    return data

def clean_status(data):
    status_names = dict()
    status_index = 0
    statuses = []
    for status in data['Status']:
        if status not in status_names:
            status_names[status] = status_index
            status_index += 1
        statuses.append(status_names[status])
    
    data = data.drop(columns=['Status'])
    # data.insert(1, 'Status', statuses)
    return data

def clean_vessels(data):
    vessel_name = data.pop('Vessel.Name')
    data['Spirit of British Columbia'] = (vessel_name == 'Spirit of British Columbia')*1.0
    data['Queen of New Westminster'] = (vessel_name == 'Queen of New Westminster')*1.0
    data['Spirit of Vancouver Island'] = (vessel_name == 'Spirit of Vancouver Island')*1.0
    data['Coastal Celebration'] = (vessel_name == 'Coastal Celebration')*1.0
    data['Queen of Alberni'] = (vessel_name == 'Queen of Alberni')*1.0
    data['Coastal Inspiration'] = (vessel_name == 'Coastal Inspiration')*1.0
    data['Skeena Queen'] = (vessel_name == 'Skeena Queen')*1.0
    data['Coastal Renaissance'] = (vessel_name == 'Coastal Renaissance')*1.0
    data['Queen of Oak Bay'] = (vessel_name == 'Queen of Oak Bay')*1.0
    data['Queen of Cowichan'] = (vessel_name == 'Queen of Cowichan')*1.0
    data['Queen of Capilano'] = (vessel_name == 'Queen of Capilano')*1.0
    data['Queen of Surrey'] = (vessel_name == 'Queen of Surrey')*1.0
    data['Queen of Coquitlam'] = (vessel_name == 'Queen of Coquitlam')*1.0
    data['Bowen Queen'] = (vessel_name == 'Bowen Queen')*1.0
    data['Queen of Cumberland'] = (vessel_name == 'Queen of Cumberland')*1.0
    data['Island Sky'] = (vessel_name == 'Island Sky')*1.0
    data['Mayne Queen'] = (vessel_name == 'Mayne Queen')*1.0
    return data

def stitch_traffic(data, traffic_data):
    # "Year","Month","Day","Hour","Minute","Second","Traffic.Ordinal"
    # for each entry in the data, add a column with the corresponding traffic ordinal
    traffic = []
    for index, row in data.iterrows():
        date = row['timestamp'] + row['Scheduled.Departure']
        date = datetime.datetime.fromtimestamp(date)
        traffic_point = traffic_data.loc[
            (traffic_data['Year'] == date.year) &
            (traffic_data['Month'] == date.month) &
            (traffic_data['Day'] == date.day) &
            (traffic_data['Hour'] == date.hour) &
            (traffic_data['Minute'] == date.minute)
            ]
        if not traffic_point.shape[0]:
            traffic_point = traffic_data.loc[
                    (traffic_data['Month'] == date.month) &
                    (traffic_data['Day'] == date.day) &
                    (traffic_data['Hour'] == date.hour)
                ]
            if not traffic_point.shape[0]:
                traffic_point = traffic_data.loc[
                        (traffic_data['Month'] == date.month)
                    ]['Traffic.Ordinal'].mean()
            else:
                traffic_point = traffic_point.iloc[0]['Traffic.Ordinal']
        else:
            traffic_point = traffic_point.iloc[0]['Traffic.Ordinal']
        progress_perc = int(100*len(traffic)/data.shape[0])
        print("stitching traffic data: " + str(progress_perc) + "% [" + "="*progress_perc + " "*(100-progress_perc) +"] ", end="    \r")
        traffic.append(traffic_point)
    print("\nDone.")
    data.insert(1, 'Traffic', traffic)
    return data

def stitch_weather(data, weather_data, city):
    # vancouver
    # "Date.Time","Year","Month","Day","Time","Temperature.in.Celsius","Dew.Point.Temperature.in.Celsius",
    # "Relative.Humidity.in.Percent","Humidex.in.Celsius","Hour"
    
    # for each entry in the data, add a column with the corresponding weather
    tempsinc = []
    dewpointtempsinc = []
    relhumidinpercents = []
    humidinc = []
    winddirs = []
    windspds = []
    visinkms = []
    stprssrs = []
    for index, row in data.iterrows():
        date = row['timestamp']
        # 2016-08-01 00:00:00
        # date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        date = datetime.datetime.fromtimestamp(date)
        weather_point = weather_data.loc[
            weather_data['Date.Time'] == date.strftime("%Y-%m-%d %H:%M:00")
        ]
        weather_point_similar = weather_data.loc[
            (weather_data['Month'] == str(date.month))
            # (weather_data['Time'] == date.strftime("%H:00"))
        ]
        tempinc = weather_point.iloc[0]['Temperature.in.Celsius']
        dewpointtempinc = weather_point.iloc[0]['Dew.Point.Temperature.in.Celsius']
        relhumidinpercent = weather_point.iloc[0]['Relative.Humidity.in.Percent']
        if str(tempinc) == "nan":
            tempinc = weather_point_similar['Temperature.in.Celsius'].mean()
        if str(dewpointtempinc) == "nan":
            dewpointtempinc = weather_point_similar['Dew.Point.Temperature.in.Celsius'].mean()
        if str(relhumidinpercent) == "nan":
            relhumidinpercent = weather_point_similar['Relative.Humidity.in.Percent'].mean()
        tempsinc.append(tempinc)
        dewpointtempsinc.append(dewpointtempinc)
        relhumidinpercents.append(relhumidinpercent)
        if city == "vancouver":
            huminc = weather_point.iloc[0]['Humidex.in.Celsius']
            if str(huminc) == "nan":
                huminc = weather_point_similar['Humidex.in.Celsius'].mean() 
            humidinc.append(huminc)
        if city == "victoria":
            # victoria
            # "Wind.Direction.in.Degrees","Wind.Speed.km.per.h","Visibility.in.km","Station.Pressure.in.kPa","Weather"
            
            winddir = weather_point.iloc[0]['Wind.Direction.in.Degrees']
            windspd = weather_point.iloc[0]['Wind.Speed.km.per.h']
            visinkm = weather_point.iloc[0]['Visibility.in.km']
            stprssr = weather_point.iloc[0]['Station.Pressure.in.kPa']
            
            if str(winddir) == "nan":
                winddir = weather_point_similar['Wind.Direction.in.Degrees'].mean() 
            winddirs.append(winddir)
            if str(windspd) == "nan":
                windspd = weather_point_similar['Wind.Speed.km.per.h'].mean() 
            windspds.append(winddir)
            if str(visinkm) == "nan":
                visinkm = weather_point_similar['Visibility.in.km'].mean() 
            visinkms.append(winddir)
            if str(stprssr) == "nan":
                stprssr = weather_point_similar['Station.Pressure.in.kPa'].mean() 
            stprssrs.append(winddir)

        progress_perc = int(100*len(tempsinc)/data.shape[0])
        print("stitching " + city + " weather data: " + str(progress_perc) + "% [" + "="*progress_perc + " "*(100-progress_perc) +"] ", end="    \r")
        
    data.insert(1, city + ".TempinC", tempsinc)
    data.insert(1, city + ".DewPointTempInC", dewpointtempsinc)
    data.insert(1, city + ".RelHumidInPercent", relhumidinpercents)
    if city =="vancouver":
        data.insert(1, city + ".HumidInC", humidinc)
    if city =="victoria":
        data.insert(1, city + ".WindDir", winddirs)
        data.insert(1, city + ".WindSpeed", windspds)
        data.insert(1, city + ".VisInKm", visinkms)
        data.insert(1, city + ".StationPressurekPa", stprssrs)
    print("\nDone.")
    return data
