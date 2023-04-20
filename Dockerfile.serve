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

FROM python:3.10.10-bullseye

WORKDIR /endpoint

COPY requirements.txt requirements.txt
COPY ./ . 

# Install the Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Create and use a non-root user
RUN useradd -m appUser
USER appUser

# Serve the model
CMD ["uvicorn", "serve.internal.api:app", "--host", "0.0.0.0", "--port", "5000"]
