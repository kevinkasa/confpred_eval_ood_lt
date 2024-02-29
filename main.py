"""Main script for running conformal prediction and saving results"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import conformal_prediction as cp
import evaluation


def main(root_path: str, save_path: str, percent_cal: float, alpha: float, n_trials: int = 10,
         ood_datasets: list = None, INC_datasets: list = None, ):
    """
    Performs conformal prediction and evaluation on ImageNet and given OOD datasets.
    Metrics are averaged across n_trials random splits, and the resulting mean & std are saved to .csv's
    :param root_path: Root path where softmax scores are stored
    :param save_path: Where to save results
    :param percent_cal: Percent of ImageNet validation data to use for calibration
    :param alpha: Desired error level
    :param n_trials: Number of trials to run
    :param ood_datasets: list of ood imagenet datasets to evaluate on
    :param INC_datasets: List of  imagenet-c corruption types to evaluate.
            Chose from ['contrast', 'brightness', 'motion_blur', 'gaussian_noise']
    """
    root_path = Path(root_path)
    save_path = Path(save_path)

    results_list = defaultdict(list)  # used to keep track of results across multiple trials
    models = ['deit3S', 'deit3B', 'vitS', 'vitB', 'resnet50', 'resnet152']
    for trial in tqdm(range(n_trials)):  # do multiple trials
        print('Trial: {}'.format(trial))
        # set up dict that will hold the results on each trial
        results = defaultdict(lambda: defaultdict(list))
        for model in models:
            print('Working on {} model'.format(model))
            print('Working on ImageNet')
            in1k_path = root_path / 'IN1k' / ('imagenet-' + model + '.npz')
            # start by loading and calibrating on imagenet1k validation set
            in1k_data = np.load(in1k_path)
            in1k_smx = in1k_data['smx']  # get softmax scores
            in1k_labels = in1k_data['labels'].astype(int)

            # Split the softmax scores into calibration and validation sets
            n = int(len(in1k_labels) * percent_cal)
            idx = np.array([1] * n + [0] * (in1k_smx.shape[0] - n)) > 0
            np.random.shuffle(idx)
            cal_smx, val_smx = in1k_smx[idx, :], in1k_smx[~idx, :]
            cal_labels, val_labels = in1k_labels[idx], in1k_labels[~idx]

            # evaluate accuracy
            acc = evaluation.compute_accuracy(val_smx, val_labels)

            # get right regularization penalties for RAPS
            if model in ['resnet50', 'resnet152']:
                k_reg = 5
                lambda_reg = 0.01
            else:
                k_reg = 2
                lambda_reg = 0.1

            # calibrate on imagenet calibration set
            tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)  # get conformal quantile
            tau_raps = cp.calibrate_raps(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            # APS with no regularization
            tau_aps = cp.calibrate_raps(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=None, lambda_reg=None)

            # get confidence sets
            conf_set_thr = cp.predict_threshold(val_smx, tau_thr)
            conf_set_raps = cp.predict_raps(val_smx, tau_raps, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            conf_set_aps = cp.predict_raps(val_smx, tau_aps, rng=True)

            # evaluate coverage
            cov_thr = float(evaluation.compute_coverage(conf_set_thr, val_labels))
            cov_raps = float(evaluation.compute_coverage(conf_set_raps, val_labels))
            cov_aps = float(evaluation.compute_coverage(conf_set_aps, val_labels))

            # evaluate set size
            size_thr, _ = evaluation.compute_size(conf_set_thr)
            size_raps, _ = evaluation.compute_size(conf_set_raps)
            size_aps, _ = evaluation.compute_size(conf_set_aps)

            # save results for this trial
            results['imagenet']['model'].append(model)
            results['imagenet']['acc'].append(acc)
            results['imagenet']['cov_thr'].append(cov_thr)
            results['imagenet']['cov_raps'].append(cov_raps)
            results['imagenet']['cov_aps'].append(cov_aps)
            results['imagenet']['size_thr'].append(float(size_thr))
            results['imagenet']['size_raps'].append(float(size_raps))
            results['imagenet']['size_aps'].append(float(size_aps))

            # evaluate on OOD datasets, using threshold calibrated from original calibration set
            for dataset in ood_datasets:
                print('Evaluation on {}'.format(dataset))
                # get all subdirectories, to be compatible with imagenet-C
                data_paths = []
                root_ood_path = root_path / dataset
                # need to do some string manipulate due to the way inference results were saved, should change in future
                for p in root_ood_path.glob("**/" + dataset.replace('_', "") + '-' + model + '.npz'):
                    data_paths.append(p)

                for data_path in data_paths:
                    eval = True
                    if dataset == 'imagenet_c' and data_path.parts[-3] not in INC_datasets:
                        eval = False  # used to skip IN-C corruption types that are not wanted
                    if eval:
                        data = np.load(data_path)
                        val_smx = data['smx']
                        val_labels = data['labels'].astype(int)

                        # evaluate accuracy
                        acc = evaluation.compute_accuracy(val_smx, val_labels)

                        # get confidence sets
                        conf_set_thr = cp.predict_threshold(val_smx, tau_thr)
                        conf_set_raps = cp.predict_raps(val_smx, tau_raps, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
                        conf_set_aps = cp.predict_raps(val_smx, tau_aps, rng=True)

                        # evaluate coverage
                        cov_thr = float(evaluation.compute_coverage(conf_set_thr, val_labels))
                        cov_raps = float(evaluation.compute_coverage(conf_set_raps, val_labels))
                        cov_aps = float(evaluation.compute_coverage(conf_set_aps, val_labels))

                        # evaluate set size
                        size_thr, _ = evaluation.compute_size(conf_set_thr)
                        size_raps, _ = evaluation.compute_size(conf_set_raps)
                        size_aps, _ = evaluation.compute_size(conf_set_aps)

                        if dataset == 'imagenet_c':  # get the right IN-C corruption type and level
                            data_name = dataset + '_' + data_path.parts[-3] + '_' + data_path.parts[-2]
                        else:
                            data_name = dataset

                        # save results for this trail
                        results[data_name]['model'].append(model)
                        results[data_name]['acc'].append(acc)
                        results[data_name]['cov_thr'].append(cov_thr)
                        results[data_name]['cov_raps'].append(cov_raps)
                        results[data_name]['cov_aps'].append(cov_aps)
                        results[data_name]['size_thr'].append(float(size_thr))
                        results[data_name]['size_raps'].append(float(size_raps))
                        results[data_name]['size_aps'].append(float(size_aps))

        # create a dataframe for each dataset, then append trial results to the corresponding dataframe
        for key, value in results.items():  # key: dataset, value: trial results
            results_list[key].append(pd.DataFrame.from_dict(value))

    for dataset, results in results_list.items():
        concatenated = pd.concat(results_list[dataset])  # concatenate the different trail results for this dataset
        avg = concatenated.groupby('model').mean()
        std = concatenated.groupby('model').std()

        # save results
        avg_file = save_path / (dataset + '_mean' + '.csv')
        avg_file.parent.mkdir(parents=True, exist_ok=True)
        avg.to_csv(avg_file)

        std_file = save_path / (dataset + '_std' + '.csv')
        std_file.parent.mkdir(parents=True, exist_ok=True)
        std.to_csv(std_file)


if __name__ == '__main__':
    main('/inference_results/', save_path='results/', percent_cal=0.5,
         alpha=0.1, ood_datasets=['imagenet_r', 'imagenet_a', 'imagenet_v2'], INC_datasets=['contrast'], n_trials=10)
