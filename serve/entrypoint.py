#  Copyright (c) University College London Hospitals NHS Foundation Trust
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging

def init():
    # TODO: Perform any initialization of the model
    logging.info("Model initialized")
    return {"init": "DONE"}


def run(model_inputs: dict = None):
    # TODO: Add code here that calls your model
    #       model_inputs is a dictionary containing any inputs that were passed to the model endpoint
    #       This function should return a dictionary containing the model results
    
    logging.info("Model run started")
    model_results = {"result": "Hello World!"}
    logging.info("Model run completed")
    
    # return model results
    return model_results
