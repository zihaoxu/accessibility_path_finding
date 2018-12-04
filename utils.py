import json, requests
import pandas as pd 
import numpy as np
import seaborn as sns
import osmnx as ox
import matplotlib.pyplot as plt
from collections import defaultdict

dataset_path = '../0-datasets/'
raw_path = dataset_path + 'raw/'
mst_path = dataset_path + 'mst/'

# Helper functions
def show_str_unique(series):
	series = series.apply(lambda x: str(x))
	return series.value_counts()

def get_path_coords(path_str):
    path_str = path_str.replace("LINESTRING (", "").replace(")", "")
    coord_pairs = path_str.split(", ")
    coord_pairs = [[float(i) for i in a.split(" ")] for a in coord_pairs]
    return [i[0] for i in coord_pairs], [i[1] for i in coord_pairs]

def process_edge_rows(edge, idx):
    y,x = get_path_coords(edge.loc[idx, 'geometry'])
    y_start = [y[i] for i in range(len(x) - 1)]
    y_end = [y[i+1] for i in range(len(x) - 1)]
    x_start = [x[i] for i in range(len(x) - 1)]
    x_end = [x[i+1] for i in range(len(x) - 1)]
    return y_start, y_end, x_start, x_end, \
           [edge.loc[idx, 'highway']] * (len(x)-1), [edge.loc[idx, 'oneway']] * (len(x)-1),\
           [idx] * (len(x)-1)


def visualize_path(edge, idx, scale_factor = 0.1):
    x,y = get_path_coords(edge.loc[idx, 'geometry'])
    plt.scatter(x,y)
    diff_x = max(x) - min(x)
    diff_y = max(y) - min(y)
    plt.xlim(min(x) - diff_x * scale_factor, max(x) + diff_x * scale_factor)
    plt.ylim(min(y) - diff_y * scale_factor, max(y) + diff_y * scale_factor)
    plt.show()

def get_elevation(lat, lng):
    apikey = "AIzaSyDk4njF9iZ4zYOYLm54eqExvEYaICH-Zes"
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    request = requests.get(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)
    try:
        results = json.loads(request.text).get('results')
        print(results)
        if 0 < len(results):
            elevation = results[0].get('elevation')
            return elevation
        else:
            print('HTTP GET Request failed.')
            return 0
    except ValueError as e:
        print('JSON decode failed: '+str(request) + str(e))

def get_elevation_path(s_lat, s_lng, e_lat, e_lng, seg_length):
    apikey = "AIzaSyDk4njF9iZ4zYOYLm54eqExvEYaICH-Zes"
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    length = np.sqrt(abs(s_lat - e_lat) ** 2 + abs(s_lng - e_lng) ** 2)
    print(length)
    pass
    samples = np.sqrt(abs(s_lat - e_lat) ** 2 + abs(s_lng - e_lng) ** 2) // seg_length
    request = requests.get(url+"?path="+str(s_lat)+","+str(s_lng)+"|"+str(e_lat)+","+str(e_lng)+"&samples="+str(samples)+"&key="+apikey)
    try:
        results = json.loads(request.text).get('results')
        print(results)
        if 0 < len(results):
            elevation = results[0].get('elevation')
            return elevation
        else:
            print('HTTP GET Request failed.')
            return 0
    except ValueError as e:
        print('JSON decode failed: '+str(request) + str(e))