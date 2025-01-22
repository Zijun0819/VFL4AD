
from DataPreProcessing.enums import PNeumaDatasetID
from DataPreProcessing.utils.roadnetutils import roadnet_data_processing
from codecarbon import EmissionsTracker


if __name__ == '__main__':
    d2 = PNeumaDatasetID.D181024_T0830_0900_DR2
    d3 = PNeumaDatasetID.D181024_T0830_0900_DR3
    d5 = PNeumaDatasetID.D181024_T0830_0900_DR5

    tracker = EmissionsTracker()
    # Start energy monitoring
    tracker.start()
    road_density_flow = roadnet_data_processing([d2, d3, d5])
    # Stop monitor and collect the energy cost data
    emissions = tracker.stop()
    # print energy cost info
    print(f"Estimated energy consumption: {emissions:.8f} kWh")