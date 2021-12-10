from sklearn.metrics import precision_recall_fscore_support, roc_curve, accuracy_score
import numpy as np
import time
# import matplotlib.pyplot as plt


def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary',zero_division=1)
    acc = accuracy_score(np.asarray(y_true).astype(np.int32), np.asarray(y_pred))
    return acc, precision, recall, f1

def time_check(msg=''):
    print('{} -- {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))






# def roc(y_true, y_pred):
#     fpr, tpr, threshold = roc_curve(np.asarray(y_true).astype(np.int32), np.asarray(y_pred))
#     return fpr, tpr, threshold


# def plot_roc(fpr, tpr, fn="result.png"):
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw,
#              label='ROC curve (area = __)')
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.savefig(fn)

