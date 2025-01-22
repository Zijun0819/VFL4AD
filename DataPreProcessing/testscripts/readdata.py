"""
Copyright 2020, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of Travia.

Travia is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Travia is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Travia.  If not, see <https://www.gnu.org/licenses/>.
"""
import datetime
import os

import numpy as np


class PNeumaDataset():
    @staticmethod
    def read_pneuma_csv(file_path):
        vehicles = []
        vehicle_trajectories = []

        with open(file_path, 'r') as file:
            for index, line in enumerate(file):
                if not index:
                    # skip the header
                    line.replace('\n', '').split(';')
                else:
                    as_list = line.replace('\n', '').split(';')
                    vehicle = [int(as_list[0]), str(as_list[1]), float(as_list[2]), float(as_list[3])]
                    vehicles.append(vehicle)

                    lat_lon_time_list = []
                    time_data_only = np.array([float(value) for value in as_list[4:-1]])
                    time_data_only.resize(int(len(time_data_only) / 6), 6)

                    time_lat_lon_only = time_data_only[:, [0, 1, -1]]

                    for row in time_data_only:
                        lat_lon_time_list.append((row[0], row[1], row[-1]))

                    vehicle_trajectories.append(lat_lon_time_list)

        return vehicle_trajectories



if __name__ == '__main__':
    file_path = os.path.join('../../Data', "20181024", "20181024_d2_0830_0900.csv")
    res = PNeumaDataset.read_pneuma_csv(file_path)
    print(res)
