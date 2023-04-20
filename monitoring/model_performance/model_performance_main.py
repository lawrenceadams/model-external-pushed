from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml import load_component
from azure.ai.ml.entities import Environment
from azure.ai.ml import dsl
from azure.ai.ml.entities import RecurrenceTrigger, JobSchedule
from datetime import datetime
import json
import os

with open("./config.json") as f:
    config = json.load(f)


@dsl.pipeline(compute=config["compute_target"])
def model_performance_pipeline(
    inference_data_path,
    ground_truth_data_path,
    mlflow_uri,
    logger_connection_string,
    model_name,
    model_version,
    index_name
):
    measure_model_performance_component = load_component(
        "./model_performance/model_performance.yml")

    # using data_prep_function like a python call with its own inputs
    model_performance_job = measure_model_performance_component(inference_data_path=inference_data_path,
                                                                groundtruth_data_path=ground_truth_data_path,
                                                                mlflow_uri=mlflow_uri,
                                                                logger_connection_string=logger_connection_string,
                                                                model_name=model_name,
                                                                model_version=model_version,
                                                                index_name=index_name)


def main():

    input_folder = config["input_folder"]
    inference_file_name = config["inference_file_name"]
    groundtruth_file_name = config["ground_truth_file_name"]
    experiment_name = config["perf_experiment_name"]
    compute_target = config["compute_target"]
    model_name = config["model_name"]
    model_version = config["model_version"]
    index_name = config["index_name"]
    log_handler_connection_string = config["log_handler_connection_string"]

    ml_client = MLClient.from_config(
        DefaultAzureCredential(),
    )

    env_docker_image = Environment(
        image=config["docker_image"],
        name="model-performance-env",
        conda_file="./model_performance/env.yml",
    )

    pipeline_job_env = ml_client.environments.create_or_update(
        env_docker_image)

    data_store_prefix = config["datastore_path"]
    # Retrieve files from a remote location such as the Blob storage
    inference_data_path = Input(
        # this path needs to be adjusted to your datastore path
        path=f"{data_store_prefix}/{input_folder}/{inference_file_name}",
        type="uri_file"
    )

    groundtruth_data_path = Input(
        # this path needs to be adjusted to your datastore path
        path=f"{data_store_prefix}/{input_folder}/{groundtruth_file_name}",
        type="uri_file"
    )

    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name).mlflow_tracking_uri

    pipeline_job = model_performance_pipeline(inference_data_path=inference_data_path,
                                              ground_truth_data_path=groundtruth_data_path,
                                              mlflow_uri=mlflow_tracking_uri,
                                              logger_connection_string=log_handler_connection_string,
                                              model_name=model_name,
                                              model_version=model_version,
                                              index_name=index_name)

    pipeline_job.settings.default_compute = compute_target

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Create a schedule for the job

    schedule_name = "model_performance_schedule"

    schedule_start_time = datetime.utcnow()
    recurrence_trigger = RecurrenceTrigger(
        frequency="minute",
        interval=10,
    )

    job_schedule = JobSchedule(
        name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
    )
    job_schedule = ml_client.schedules.begin_create_or_update(
        schedule=job_schedule
    )


if __name__ == '__main__':
    main()
