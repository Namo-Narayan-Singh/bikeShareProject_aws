import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import pre_pipeline_preparation


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data: dict) -> dict:
    """Make a prediction using a saved model """

    data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    #data=data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    
    results = {"predictions": None, "version": _version, }
    
    predictions = bikeshare_pipe.predict(data)

    results = {"predictions": predictions,"version": _version}
    print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday':["2012-11-05"],'season':["winter"],'hr':["6am"],'holiday':['No'],'weekday':["Mon"],
                'workingday':["Yes"],'weathersit':["Mist"],'temp':[6.10],'atemp':[3.0014],'hum':[49.0],'windspeed':[19.0012],'casual':[4],'registered':[135]}
    
    make_prediction(input_data=data_in)