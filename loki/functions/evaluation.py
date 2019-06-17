"""Useful functions for evaluating a model's performance"""
import sklearn.metrics as skmet

def get_confusion_matrix(actual, predicted):
    """Get the confusion matrix and statistics

    Arguments:
    ----------
    actual -- np.ndarray:
        The atual classes,
    predicted -- np.ndarray:
        The inferred classes

    Return:
    -------
    """
    cm = skmet.confusion_matrix(test_values, results)

    accuracy = (cm[0,0] + cm[1,1]) / len(test_values)

    precision = cm[1,1] / (cm[0,1] + cm[1,1]) # true positives over false positives and true positives
    recall = cm[1,1] / (cm[1,0] + cm[1,1]) # True positives over false negatives and true positives

    results = {"cm":cm, "accuracy":accuracy, "precision":precision, "recall":recall}

    return results

def print_confusion_matrix(actual, predicted):
    """Print out the confusion matrix"""

    results = get_confusion_matrix(actual, predicted)

    print(cm)

    print(f"Accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
