# coding:utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMU_COLUMN_NAMES = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl',
                    'vu', 'ax', 'ay', 'az', 'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl',
                    'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']


def read_imu(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = IMU_COLUMN_NAMES
    return df


def compute_great_circle_distance(lat1, lon1, lat2, lon2):
    delta_theta = float(np.sin(lat1*np.pi/180)*np.sin(lat2*np.pi/180) +
                        np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(lon1*np.pi/180 - lon2*np.pi/180))

    return 6371000.0*np.arccos(np.clip(delta_theta, -1, 1))


# driving distance
DATA_PATH = './'
prev_imu_data = None
gps_distances = []
imu_distances = []
for frame in range(150):
    imu_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt'%frame))
    if prev_imu_data is not None:
        gps_distances += [compute_great_circle_distance(imu_data.lat, imu_data.lon,
                                                     prev_imu_data.lat, prev_imu_data.lon)]
        # compute imu_distance related to vf and vl
        imu_distances += [0.1*np.linalg.norm(imu_data[['vf', 'vl']])]
    prev_imu_data = imu_data

plt.figure(figsize=(15,10))
plt.plot(gps_distances, label='gps distance')
plt.plot(imu_distances, label='imu distance')
plt.legend()
plt.show()


# driving path
prev_imu_data = None
locations = []
for frame in range(150):
    imu_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt' % frame))
    if prev_imu_data is not None:
        displacement = 0.1*np.linalg.norm(imu_data[['vf', 'vl']])
        yaw_change = np.float(imu_data['yaw'] - prev_imu_data['yaw'])
        # change the locations of all previous points
        for i in range(len(locations)):
            x0, y0 = locations[i]
            # update locations
            x1 = x0*np.cos(yaw_change) + y0*np.sin(yaw_change) - displacement
            y1 = -x0*np.sin(yaw_change) + y0*np.cos(yaw_change)
            locations[i] = np.array([x1, y1])

    locations += [np.array([0, 0])]
    prev_imu_data = imu_data

plt.figure(figsize=(15,10))
plt.plot(np.array(locations)[:, 0], np.array(locations)[:, 1])
plt.show()