import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import bikeshareImputer
from bikeshare_model.processing.features import Mapper
# from bikeshare_model.processing.features import age_col_tfr

bikeshare_pipe=Pipeline([
    
    ("season_imputation", bikeshareImputer(variables=config.model_config.season_var)
     ),
    ("hr_imputation", bikeshareImputer(variables=config.model_config.hr_var)
     ),
    ("holiday_imputation", bikeshareImputer(variables=config.model_config.holiday_var)
     ),
    ("weekday_imputation", bikeshareImputer(variables=config.model_config.weekday_var)
     ),    
    ("weathersit_imputation", bikeshareImputer(variables=config.model_config.weathersit_var)
     ),      
    #  ##==========Mapper======##
      ("map_season",Mapper(config.model_config.season_var, config.model_config.season_mappings)
       ),
     ("map_hr",Mapper(config.model_config.hr_var, config.model_config.hr_mappings )
     ),
     ("map_holiday",Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings )
     ), 
      ("map_weekday",Mapper(config.model_config.weekday_var, config.model_config.weekday_mappings)
       ),         
      ("map_weathersit",Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)
       ),       
      ("map_workingday",Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)
       ),         
    #  # Transformation of age column
    #  ("age_transform", age_col_tfr(config.model_config.age_var)
    #  ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
     ])