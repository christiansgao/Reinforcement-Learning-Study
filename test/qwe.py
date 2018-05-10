import numpy as np
import sklearn.metrics as sk_metrics


def get_success( i, predictions, expected):
    indexes = np.where(np.array(predictions) == i)[0]
    if len(indexes) == 0:
        return .5
    predictions = [predictions[i] for i in indexes]
    expected = [expected[i] for i in indexes]

    confusion_matrix = sk_metrics.confusion_matrix(y_true=expected, y_pred=predictions)
    correct_classifications = np.diagonal(confusion_matrix);
    success_rate = sum(correct_classifications) / np.sum(confusion_matrix)
    return success_rate

test = get_success(1,[0,2,0,4,0],[0,4,3,1,0])
test = list(range(0, 10))
pass