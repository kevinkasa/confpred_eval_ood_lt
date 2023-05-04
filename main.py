"""Main script for running conformal prediction and saving outputs"""
import numpy as np

import conformal_prediction as cp
import eval


def main(in1k_path, num_cal, alpha, n_trials=10, ood_paths=None, tau=None):
    """

    :param in1k_path:
    :param num_cal:
    :param alpha: Desired error rate (1-alpha is the target coverage)
    :param n_trials:
    :param ood_paths:
    :param tau:
    :return:
    """
    # start by loading and calibrating on imagenet1k validation set
    in1k_data = np.load(in1k_path)
    in1k_smx = in1k_data['smx']  # get softmax scores
    in1k_labels = in1k_data['labels'].astype(int)

    # Split the softmax scores into calibration and validation sets
    n = int(len(in1k_labels) * num_cal)
    idx = np.array([1] * n + [0] * (in1k_smx.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = in1k_smx[idx, :], in1k_smx[~idx, :]
    cal_labels, val_labels = in1k_labels[idx], in1k_labels[~idx]
    # # get conformal quantile
    # tau = cp.calibrate_threshold(cal_smx, cal_labels, alpha)
    # # get confidence sets
    # conf_set = cp.predict_threshold(val_smx, tau)
    # avg_set_size = conf_set.sum() / len(val_labels)
    # # evaluate
    # coverage = float(eval.compute_coverage(conf_set, val_labels))
    # avg_set_size = conf_set.sum() / len(val_labels)


    tau = cp.calibrate_raps(cal_smx, cal_labels, alpha=alpha, rng=True)
    conf_set = cp.predict_raps(val_smx, tau, rng=True)
    avg_set_size = conf_set.sum() / len(val_labels)
    coverage = float(eval.compute_coverage(conf_set, val_labels))


if __name__ == '__main__':
    main('/scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-deit3B.npz', num_cal=0.2, alpha=0.1)


in1k_path = '/scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-resnet152.npz'
