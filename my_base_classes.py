import csv
import math
import numpy as np
import os
from pandas import DataFrame, concat


class MyConfig:
    n_past = 8
    n_next = 12

class ConstVelModel:
    def __init__(self, conf=MyConfig()):
        self.my_conf = conf

    def predict(self, inp): # use config for
        #inp.ndim = 2
        avg_vel = np.array([0, 0])
        if inp.ndim > 1 and inp.shape[0] > 1:
            for i in range(1, inp.shape[0]):
                avg_vel = avg_vel + inp[i, :]-inp[i-1, :]
            avg_vel = avg_vel / (inp.shape[0]-1)

        cur_pos = inp[-1, :]
        out = np.empty((0, 2))
        for i in range(0, self.my_conf.n_next):
            out = np.vstack((out, cur_pos + avg_vel * (i+1)))

        return out


class Scale(object):
    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf

    def normalize(self, data):
        data[:, 0] = (data[:, 0] - self.min_x) / (self.max_x - self.min_x)
        data[:, 1] = (data[:, 1] - self.min_y) / (self.max_y - self.min_y)
        return data

    def denormalize(self, data):
        data_copy = data

        sx = (self.max_x - self.min_x)
        sy = (self.max_y - self.min_y)
        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] * sx + self.min_x
            data_copy[1] = data[1] * sy + self.min_y
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] * sx + self.min_x
            data_copy[:, 1] = data[:, 1] * sy + self.min_y
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] * sx + self.min_x
            data_copy[:, :, 1] = data[:, :, 1] * sy + self.min_y

        return data_copy


# seyfried rows are like this:
# Few Header lines for Obstacles
# id, timestamp, pos_x, pos_y, pos_z
def load_seyfried(filename='/home/jamirian/workspace/crowd_sim/tests/sey_all/*.sey', down_sample=4):
    pos_data_list = list()
    time_data_list = list()

    # check to search for many files?
    file_names = list()
    if '*' in filename:
        files_path = filename[:filename.index('*')]
        extension = filename[filename.index('*')+1:]
        for file in os.listdir(files_path):
            if file.endswith(extension):
                file_names.append(files_path+file)
    else:
        file_names.append(filename)

    for file in file_names:
        with open(file, 'r') as data_file:
            csv_reader = csv.reader(data_file, delimiter=' ')
            id_list = list()
            i = 0
            for row in csv_reader:
                i += 1
                if i == 4:
                    fps = row[0]

                if len(row) != 5:
                    continue

                id = row[0]
                # print(row)
                ts = float(row[1])
                if ts % down_sample != 0:
                    continue

                px = float(row[2])/100.
                py = float(row[3])/100.
                pz = float(row[4])/100.
                if id not in id_list:
                    id_list.append(id)
                    pos_data_list.append(list())
                    time_data_list.append(np.empty((1,0)))
                pos_data_list[-1].append(np.array([px, py]))
                time_data_list[-1] =  np.hstack(time_data_list[-1], ts)

    p_data = list()
    track_length_list = []

    scale = Scale()
    for d in pos_data_list:
        len_i = len(d)
        track_length_list.append(len_i)
        ped_i = np.array(d)
        scale.min_x = min(scale.min_x, min(ped_i[:, 0]))
        scale.max_x = max(scale.max_x, max(ped_i[:, 0]))
        scale.min_y = min(scale.min_y, min(ped_i[:, 1]))
        scale.max_y = max(scale.max_y, max(ped_i[:, 1]))
        p_data.append(ped_i)

    t_data = np.array(time_data_list)

    return p_data, scale, t_data


def to_supervised(data, n_in=1, n_out=1, diff_in=False, diff_out=True, drop_nan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        names += [('var_in%d(t-%d)' % (j + 1, i-1)) for j in range(n_vars)]
        if diff_in:
            cols.append(df.shift(i-1) - df.shift(i))
        else:
            cols.append(df.shift(i-1))

    # forecast sequence (t, t+1, ... t+n)
    for i in range(1, n_out+1):
        names += [('var_out%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        if diff_out:
            cols.append(df.shift(-i) - df.shift(0))  # displacement
        else:
            cols.append(df.shift(-i))  # position

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)

    return agg.values

