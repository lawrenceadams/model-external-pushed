import argparse
import mlflow
import os
import pandas as pd
from sklearn.compose import make_column_selector as selector
import numpy as np
from matplotlib import pyplot as plt
from alibi_detect.cd import TabularDrift
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
import seaborn as sns


def kde_plot(x_ref, x_new, title):
    '''
    Plot the distribution of the reference and new data using a kernel density estimate.
    input: 
    x_ref, x_new: the reference and new data
    title: the title of the plot
    output: fig: the figure object
    '''
    plt.figure()
    dist_plot = sns.kdeplot(x_ref, shade=True, color='blue', label='reference')
    sns.kdeplot(x_new, shade=True, color='red', label='new')
    dist_plot.set_title(title)
    fig = dist_plot.get_figure()
    return fig


def cat_bar_plot(reference_df, new_df, col, drift_detected):
    '''
    Plot the distribution of the reference and new data using a bar plot.
    input: reference_df, new_df: the reference and new data
    drift_detected: boolean indicating whether a drift was detected
    output: fig: the figure object
    '''
    ref_counts = reference_df[col].value_counts(normalize=True).reset_index()
    ref_counts['source'] = 'reference data'

    new_counts = new_df[col].value_counts(normalize=True).reset_index()
    new_counts['source'] = 'new data'

    counts_df = pd.concat([ref_counts, new_counts])
    plt.figure()
    fig = sns.barplot(x=col, y='index', hue='source', data=counts_df)
    fig.set_title(f'{col}: drift detected: {bool(drift_detected)}')
    fig = fig.get_figure()

    return fig


def pval_heatmap(col_name, p_vals):
    '''
    Plot the p values of the statistical tests used to detect the drift.
    inputs:
    col_name: the name of the columns
    p_vals: the p values
    output: fig: the figure object
    '''
    p_values = {col_name[i]: p_vals[i] for i in range(len(col_name))}

    p_values = pd.DataFrame([p_values]).T

    plt.figure()
    heatplot = sns.heatmap(p_values, annot=True, cmap=[
                           'red', 'blue'], center=0.05)
    heatplot.figure.suptitle('P vals summary')
    fig = heatplot.get_figure()
    return fig


def read_data(reference_data_path, new_data_path):
    '''
    In the template, the assumption is that the data are stored in csv files,
    Change this function to read the data using the appropriate method for your data type.

    '''
    reference_df = pd.read_csv(reference_data_path)
    new_df = pd.read_csv(new_data_path)
    return reference_df, new_df


def get_category_columns(df):
    '''
    This function is used to get the categorical columns in the data.
    '''
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(df)
    return categorical_columns


def compare_distributions(reference_df, new_df, cat_col):
    '''
    This function is used to compare the distributions of the reference and new data.
    inputs: reference_df, new_df: the reference and new data, cat_col: the categorical columns
    outputs: is_drift: boolean indicating whether a drift was detected, fpreds: the results of the statistical tests 
    used to detect the drift
    '''
    categorical_columns_dict = {
        list(reference_df.columns).index(i): None for i in cat_col}
    cd = TabularDrift(reference_df.values, p_val=.05,
                      categories_per_feature=categorical_columns_dict)
    fpreds = cd.predict(
        new_df[reference_df.columns].values, drift_type='feature')
    is_drift = int(max(fpreds['data']['is_drift']))
    return is_drift, fpreds['data']


def compute_severity_level(drift_pred):
    '''
    This function is used to compute the severity level of the drift.
    The severity level is used to determine the priority of the alert.
    The default implementation is to return 0, which means that the drift is not severe.
    You can change this function to return a different severity level based on the drift detected.
    '''

    return 0


def gen_categorical_metrics(reference_df, new_df, col):
    '''
    This function is used to generate the metrics for categorical columns.
    inputs: reference_df, new_df: the reference and new data, col: the categorical column
    outputs: metrics_dict: a dictionary containing the metrics
    '''
    metrics_dict = {}
    most_frequent_category_new = new_df[col].value_counts(normalize=True)
    metrics_dict['most_common_category_new'] = most_frequent_category_new.index[0]
    metrics_dict['most_common_category_freq_new'] = str(
        most_frequent_category_new[0])

    most_frequent_category_ref = reference_df[col].value_counts(normalize=True)
    metrics_dict['most_common_category_ref'] = most_frequent_category_ref.index[0]
    metrics_dict['most_common_category_freq_ref'] = str(
        most_frequent_category_ref[0])
    return metrics_dict


def gen_cont_metrics(reference_df, new_df, col):
    '''
    This function is used to generate the metrics for continuous columns.
    inputs: reference_df, new_df: the reference and new data, col: the continuous column
    outputs: metrics_dict: a dictionary containing the metrics
    '''
    metrics_dict = {}
    metrics_dict['mean_new'] = new_df[col].mean()
    metrics_dict['mean_ref'] = reference_df[col].mean()
    return metrics_dict


def main(args):

    logger = logging.getLogger(__name__)
    logger.addHandler(AzureLogHandler(
        connection_string=args.logger_connection_string))

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    reference_data_path = args.reference_data_path
    new_data_path = args.new_data_path
    #output_path = args.output_path

    reference_df, new_df = read_data(reference_data_path, new_data_path)

    cat_col = get_category_columns(reference_df)  # get categorical columns
    is_drift, drift_pred = compare_distributions(reference_df, new_df, cat_col)

    severity_level = compute_severity_level(drift_pred)
    properties = {'custom_dimensions': {'is_drift': is_drift,
                                        'severity': severity_level, 'run_id': run_id}}
    logger.info(f'{args.model_name}_data_drift_total', extra=properties)

    heatmap_fig = pval_heatmap(reference_df.columns, drift_pred['p_val'])
    mlflow.log_figure(heatmap_fig, f'pvalues_summary.png')

    for id, col in enumerate(reference_df.columns):
        properties = {'custom_dimensions': {'model_name': args.model_name, 'model_version': args.model_version, 'feature_name': col, 'is_drift': int(drift_pred['is_drift'][id]),
                                            'distances': str(drift_pred['distance'][id]),
                                            'p_values': str(drift_pred['p_val'][id]),
                                            'run_id': run_id
                                            }}

        mlflow.log_metrics({f'{col}_drift': int(drift_pred['is_drift'][id]),
                            f'{col}_distance': drift_pred['distance'][id],
                            f'{col}_p_value': drift_pred['p_val'][id]})

        cat_cols_name = [list(reference_df.columns).index(i) for i in cat_col]

        if id in cat_cols_name:
            fig = cat_bar_plot(reference_df, new_df, col,
                               int(drift_pred['is_drift'][id]))
            mlflow.log_figure(fig, f'{col}_frequency.png')
            feature_metrics = gen_categorical_metrics(
                reference_df, new_df, col)
        else:
            drift_detected = bool(drift_pred['is_drift'][id])
            title = f'{col}: drift: {drift_detected}'
            plt.figure()
            fig = kde_plot(reference_df[col], new_df[col], title)
            mlflow.log_figure(fig, f'{col}_kde.png')
            feature_metrics = gen_cont_metrics(reference_df, new_df, col)
        properties['custom_dimensions'].update(feature_metrics)

        logger.info(f'{args.model_name}_data_drift_features', extra=properties)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--reference_data_path', type=str)
    parser.add_argument('--new_data_path', type=str)
    parser.add_argument('--mlflow_uri', type=str, default='.')
    parser.add_argument('--logger_connection_string', type=str, default='.')
    parser.add_argument('--model_version', type=str)
    args = parser.parse_args()

    main(args)
