import json, requests, pickle
import pandas as pd 
import numpy as np
import seaborn as sns
import osmnx as ox
import matplotlib.pyplot as plt
from collections import defaultdict

# galbal paths
dataset_path = '../0-datasets/'
raw_path = dataset_path + 'raw/'
mst_path = dataset_path + 'mst/'

# unit conversions
lat_to_m = 111 * 1000
lng_to_m = 111.3 * 1000

# david's api key
apikey = "AIzaSyDk4njF9iZ4zYOYLm54eqExvEYaICH-Zes"

# helper functions
def show_str_unique(series):
    ''' show the unique occurances of each line, as a string'''
    series = series.apply(lambda x: str(x))
    return series.value_counts()

def get_path_coords(path_str):
    ''' get the coordinates from the strange LINESTRING (...) format of the edges dataset '''
    path_str = path_str.replace("LINESTRING (", "").replace(")", "")
    coord_pairs = path_str.split(", ")
    coord_pairs = [[float(i) for i in a.split(" ")] for a in coord_pairs]
    return [i[0] for i in coord_pairs], [i[1] for i in coord_pairs]

def get_path_length(df, idx):
    ''' return length of a path segment for given index in clean_edge dataset '''
    y_start, y_end, x_start, x_end = df.loc[idx, 'y_start'], df.loc[idx, 'y_end'], df.loc[idx, 'x_start'], df.loc[idx, 'x_end']
    y_diff_m = abs(y_start - y_end) * lat_to_m
    x_diff_m = abs(x_start - x_end) * lng_to_m
    return np.sqrt(y_diff_m ** 2 + x_diff_m ** 2)

def process_edge_csv_rows(df, idx):
    ''' process the geometry column of the edges column.
        return the y_start, y_end, x_start, x_end, highway, oneway, index '''
    y,x = get_path_coords(df.loc[idx, 'geometry'])
    y_start = [y[i] for i in range(len(x) - 1)]
    y_end = [y[i+1] for i in range(len(x) - 1)]
    x_start = [x[i] for i in range(len(x) - 1)]
    x_end = [x[i+1] for i in range(len(x) - 1)]
    return y_start, y_end, x_start, x_end, \
           [df.loc[idx, 'highway']] * (len(x)-1), [df.loc[idx, 'oneway']] * (len(x)-1),\
           [idx] * (len(x)-1)

def break_up_edges_to_nodes(df, idx, seg_length):
    ''' break up long edges in clean_edge into small segments with nodes 
        returns a dataframe '''
    if df.loc[idx, 'length'] < seg_length * 2:
        return df[df.index == idx]
    n_segments = df.loc[idx, 'length'] // seg_length + 1
    ys = list(np.linspace(df.loc[idx, 'y_start'], df.loc[idx, 'y_end'], n_segments))
    xs = list(np.linspace(df.loc[idx, 'x_start'], df.loc[idx, 'x_end'], n_segments))
    res = defaultdict(list)
    res['y_start'] += ys[:-1]
    res['y_end'] += ys[1:]
    res['x_start'] += xs[:-1]
    res['x_end'] += xs[1:]
    res['highway'] += [df.loc[idx, 'highway']] * (len(xs)-1)
    res['oneway'] += [df.loc[idx, 'oneway']] * (len(xs)-1)
    res['edge_index'] += [df.loc[idx, 'edge_index']] * (len(xs)-1)
    res = pd.DataFrame(res)
    res['length'] = [get_path_length(res, i) for i in range(len(res))]
    return res


def visualize_path(df, idx, scale_factor = 0.1):
    ''' visualize a path in edges df based on the index '''
    x,y = get_path_coords(df.loc[idx, 'geometry'])
    plt.scatter(x,y)
    diff_x = max(x) - min(x)
    diff_y = max(y) - min(y)
    plt.xlim(min(x) - diff_x * scale_factor, max(x) + diff_x * scale_factor)
    plt.ylim(min(y) - diff_y * scale_factor, max(y) + diff_y * scale_factor)
    plt.show()

def get_elevation(lat, lng):
    ''' Get the elevation using Google evelation API for a given lat, lng pair '''
    lat, lng = float(lat), float(lng)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    request = requests.get(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)
    try:
        results = json.loads(request.text).get('results')
        # print(results)
        if 0 < len(results):
            elevation = results[0].get('elevation')
            return elevation
        else:
            print('HTTP GET Request failed.')
            return 0
    except ValueError as e:
        print('JSON decode failed: '+str(request) + str(e))

def get_elevation_path(s_lat, s_lng, e_lat, e_lng, seg_length):
    ''' Get the elevation along a path using Google evelation API '''
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    length = np.sqrt(abs(s_lat - e_lat) ** 2 + abs(s_lng - e_lng) ** 2)
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

# algorithm related functions
def find_neighbors(nodes, edges, y, x):
    ''' find the neighbors of a given y,x coordinates within the edges dataset '''
    ends = edges[(edges['y_start'] == y) & (edges['x_start'] == x)][['y_end', 'x_end']].values
    starts = edges[(edges['y_end'] == y) & (edges['x_end'] == x)][['y_start', 'x_start']].values
    return set([nodes[(nodes['y'] == y) & (nodes['x'] == x)].index.values[0] for y,x in starts] +\
               [nodes[(nodes['y'] == y) & (nodes['x'] == x)].index.values[0] for y,x in ends])