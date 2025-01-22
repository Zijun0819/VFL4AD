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
import pickle
import os
import numpy as np
from DataPreProcessing.enums import PNeumaDatasetID

class PNeumaDataset():
    def __init__(self, track_data, dataset_id):
        self.dataset_id = dataset_id
        self.track_data = track_data

    # def save(self):
    #     file_path = os.path.join('data', self.dataset_id.data_sub_folder, self.dataset_id.data_file_name + '.pkl')
    #     save_encrypted_pickle(file_path, self)

    @staticmethod
    def load(dataset_ids: list):
        file_path = os.path.join('..\\Data', dataset_ids[0].data_sub_folder, f'{dataset_ids[0].get_time_identifier}.pkl')

        if os.path.isfile(file_path):
            print(f"load trajectories of vehicles from path: {file_path}!")
            with open(file_path, 'rb') as file:
                tracks_data = pickle.load(file)
                formatted_tracks = [[item[0], float(item[1]), float(item[2]), float(item[3])] for item in tracks_data]
        else:
            print(f"try to process trajectories of vehicles!")
            tracks_data = PNeumaDataset.read_data_selected_areas(dataset_ids)
            formatted_tracks = [[item[0], float(item[1]), float(item[2]), float(item[3])] for item in tracks_data]
            with open(file_path, 'wb') as file:
                pickle.dump(formatted_tracks, file)

        return formatted_tracks

    @staticmethod
    def read_pneuma_csv(dataset_id: PNeumaDatasetID):
        stacked_tracks = np.empty((0, 4))

        # get number of lines in file first
        file_path = os.path.join('..\\Data', dataset_id.data_sub_folder, dataset_id.data_file_name + '.csv')
        car_count = 0
        with open(file_path, 'r') as file:
            for index, line in enumerate(file):
                if index:
                    car_count += 1
                    car_id = f"{dataset_id.data_file_name}_{car_count}"
                    as_list = line.replace('\n', '').split(';')

                    time_data_only = np.array([float(value) for value in as_list[4:-1]])
                    time_data_only.resize(int(len(time_data_only) / 6), 6)

                    time_lat_lon_only = time_data_only[:, [0, 1, -1]]
                    car_ids = np.full((time_lat_lon_only.shape[0], 1), car_id)
                    time_lat_lon_with_id = np.hstack((car_ids, time_lat_lon_only))

                    stacked_tracks = np.vstack((stacked_tracks, time_lat_lon_with_id))
                    print(f"==============Data_{index} processed!==============")

        return stacked_tracks

    @staticmethod
    def read_data_selected_areas(dataset_ids: list):
        selected_areas_tracks = np.empty((0, 4))

        for i in range(len(dataset_ids)):
            tracks = PNeumaDataset.read_pneuma_csv(dataset_ids[i])
            print(f"The data belong to {i+1}-th dataset has been processed!")
            selected_areas_tracks = np.vstack((selected_areas_tracks, tracks))
            print(f"The data belong to {i + 1}-th dataset has been merged!")

        selected_areas_tracks = selected_areas_tracks.tolist()

        return selected_areas_tracks


if __name__ == '__main__':
    d2 = PNeumaDatasetID.D181024_T0830_0900_DR2
    d3 = PNeumaDatasetID.D181024_T0830_0900_DR3
    d5 = PNeumaDatasetID.D181024_T0830_0900_DR5
    selected_area_tracks = PNeumaDataset.load([d2, d3, d5])
    selected_area_tracks.sort(key=lambda x: x[3])
    print(selected_area_tracks[-1])

    # 使用groupby进行分组
    # for key, group in itertools.groupby(selected_area_tracks, key=lambda x: x[3]):
    #     if int(float(key)) == 10:
    #         print(key, len(list(group)))

