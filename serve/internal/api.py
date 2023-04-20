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
from fastapi import FastAPI

from serve import entrypoint
from .about import generate_about_json
from .azure_logging import initialize_logging, disable_unwanted_loggers


logger = logging.getLogger(__name__)


# create fastapi app
app = FastAPI()

@app.on_event("startup")
async def initialize_logging_on_startup():
    initialize_logging(logging.INFO)
    disable_unwanted_loggers()


@app.get("/")
def root():
    logging.info("Root endpoint called")
    return generate_about_json()


@app.get("/run")
def run(rawdata: dict = None):
    logging.info("Run endpoint called")
    return entrypoint.run(rawdata)
