import numpy as np

import conformal_prediction as cp


def compute_conditional_multi_coverage(
        confidence_sets: np.ndarray, one_hot_labels: np.ndarray,
        conditional_labels: np.ndarray, conditional_label: int) -> float:
    """
    Compute coverage of confidence sets, potentially for multiple labels.

    The given labels are assumed to be one-hot labels and the implementation
    supports checking coverage of multiple classes, i.e., whether one of
    the given ground truth labels is in the confidence set.

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param one_hot_labels: ground truth labels on test set in one-hot format
    :param conditional_labels: conditional labels to compute coverage on a subset
    :param conditional_label: selected conditional to compute coverage for
    :return:
    """
    # TODO: debug and step through this
    selected = (conditional_labels == conditional_label)  # select subset of labels
    num_examples = np.sum(selected)
    coverage = selected * np.clip(
        np.sum(confidence_sets * one_hot_labels, axis=1), 0, 1)
    coverage = np.where(num_examples == 0, 1, np.sum(coverage / num_examples))
    return coverage


def compute_coverage(
        confidence_sets: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute unconditional coverage using compute_conditional_multi_coverage

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param labels: ground truth labels on test set (not in one-hot format)
    :return: Coverage
    """
    one_hot_labels = np.eye(confidence_sets.shape[1])[labels]
    return compute_conditional_multi_coverage(
        confidence_sets, one_hot_labels, np.zeros(labels.shape, int), 0)


def compute_conditional_coverage(
        confidence_sets: np.ndarray, labels: np.ndarray,
        conditional_labels: np.ndarray, conditional_label: int) -> float:
    """
    Compute conditional coverage using compute_conditional_multi_coverage.

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param labels: truth labels on test set (not in one-hot format)
    :param conditional_labels: conditional labels to compute coverage on a subset
    :param conditional_label: selected conditional to compute coverage for
    :return: Conditional coverage.
    """

    one_hot_labels = np.eye(confidence_sets.shape[1])[labels]
    return compute_conditional_multi_coverage(
        confidence_sets, one_hot_labels, conditional_labels, conditional_label)
