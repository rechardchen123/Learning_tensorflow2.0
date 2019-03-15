import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import os

# read the data
data_address = '/home/richardchen123/Downloads/Route1_Data_Filtered.csv'
route_data = pd.read_csv(data_address)
actual_journey_time = route_data['actual_journeytime']
planned_journey_time = route_data['planned_journeytime']
actual_value = route_data['delay']
week_number = route_data['Week_Number']
day_number = route_data['Day_Number']
temp_group = route_data['temp_group']
precipitation_group = route_data['precipitation_group']


def normalization(feature):
    '''
    The function is to normalize the data above and then transfer the data into [-1,1]. Min-Max normalization
    :param feature: the six features
    :return: the normalized data
    '''
    norm_list = list()
    min_value = min(feature)
    max_value = max(feature)

    for value in feature:
        tmp = (value - min_value) / (max_value - min_value)
        norm_list.append(tmp)
    return norm_list

def save_value(actual_journey_time_normalized,
               planned_journey_time_normalized,
               week_number_normalized,
               day_number_normalized,
               temp_group_normalized,
               precipitation_group_normalized,
               actual_value_normalized):
    '''
    This function is for storing the data and output the data into a file.
    :param actual_journey_time_normalized:
    :param planned_journey_time_normalized:
    :param day_number_normalized:
    :param temp_group_normalized:
    :param precipitation_group_normalized:
    :return: week_number_normalized
    '''
    save_dict = {'actual_journeytime':actual_journey_time_normalized,
                 'planned_journeytime':planned_journey_time_normalized,
                 'Week_Number':week_number_normalized,
                 'Day_Number':day_number_normalized,
                 'temp_group':temp_group_normalized,
                 'precipitation_group':precipitation_group_normalized,
                 'delay':actual_value_normalized}
    data = pd.DataFrame(save_dict)
    #output the file
    data.to_csv('/home/richardchen123/Downloads/processed_data.csv',index=False)

#get the processed data after the noormalization
'''
Here, we normalize the data and then we can easiy elimate the unit factor and the deviation.
'''
actual_journey_time_normalized = normalization(actual_journey_time)
planned_journey_time_normalized = normalization(planned_journey_time)
week_number_normalized = normalization(week_number)
day_number_normalized = normalization(day_number)
temp_group_normalized = normalization(temp_group)
precipitation_group_normalized = normalization(precipitation_group)
actual_value_normalized = normalization(actual_value)

#save the data into files
save_value(actual_journey_time_normalized,
           planned_journey_time_normalized,
           week_number_normalized,
           day_number_normalized,
           temp_group_normalized,
           precipitation_group_normalized,
           actual_value_normalized)




