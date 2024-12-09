import pandas as pd


class Parameters:
    __instance = None
    __dataset_location: str = None
    __dataset: pd.Dataframe = None
    __cluster_count: int = None
    __cluster_method: str = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating the object")
            cls._instance = super(Parameters, cls).__new__(cls)
        return cls._instance

    def get_name(cls):
        return cls._class_name
