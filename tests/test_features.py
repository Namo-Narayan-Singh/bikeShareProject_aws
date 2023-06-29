
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import Mapper
# from bikeshare_model.processing.features import age_col_tfr



def test_Mapper_transformer(sample_input_data):
    # Given
    transformer = Mapper(
        variables=config.model_config.holiday_var, mappings=config.model_config.holiday_mappings
    )
    #assert np.isnan(sample_input_data.loc[5,'holiday'])
    assert sample_input_data.loc[5,'holiday'] == 'No'

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[5,'holiday'] == 0







# def test_age_variable_transformer(sample_input_data):
#     # Given
#     transformer = age_col_tfr(
#         variables=config.model_config.age_var,  # cabin
#     )
#     assert np.isnan(sample_input_data.loc[709,'Age'])

#     # When
#     subject = transformer.fit(sample_input_data).transform(sample_input_data)

#     # Then
#     assert subject.loc[709,'Age'] == 21