"""
Plot overall classification performance metrics.
"""

import os
import sys
import numpy as np
import itertools
from distutils.spawn import find_executable
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp

try:
    import matplotlib
    import matplotlib.pyplot as plt

    # Check if latex is installed
    if find_executable('latex'):
        plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

    font = {'family': 'normal',
            'size': 34}

    matplotlib.rc('font', **font)
except ImportError:
    print("Warning: You will need to install matplotlib if you want to plot any metric")

COLORS = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:brown', '#aaffc3', 'tab:olive',
          'tab:cyan', '#FF1493', 'navy', 'tab:pink', 'lightcoral', '#228B22', '#aa6e28', '#FFA07A']


def plasticc_log_loss(y_true, y_pred, relative_class_weights=None):
    """
    Implementation of weighted log loss used for the Kaggle challenge
    """

    if np.nonzero(y_true[:, 0])[0].size == 0:
        start_index = 1
    else:
        start_index = 0
    print(start_index)

    predictions = y_pred.copy()

    # sanitize predictions
    epsilon = sys.float_info.epsilon  # this is machine dependent but essentially prevents log(0)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

    predictions = np.log(predictions)
    # multiplying the arrays is equivalent to a truth mask as y_true only contains zeros and ones
    class_logloss = []
    for i in range(start_index, predictions.shape[1]):
        # average column wise log loss with truth mask applied
        result = np.average(predictions[:, i][y_true[:, i] == 1])
        class_logloss.append(result)
    return -1 * np.average(class_logloss, weights=relative_class_weights[start_index:])


def compute_precision_recall(classes, y_test, y_pred_prob, name='', fig_dir='.', title=None):
    """
    Plot Precision-Recall curves.
    """

    if np.nonzero(y_test[:, 0])[0].size == 0:
        start_index = 1
    else:
        start_index = 0

    nclasses = len(classes)
    # For each class
    precision = dict()
    recall = dict()
    save_auc = dict()
    average_precision = dict()
    for i in range(start_index, nclasses):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_prob[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred_prob[:, i])
        save_auc[classes[i]] = average_precision[i]

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred_prob.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred_prob,
                                                         average="micro")
    save_auc[classes[i]] = average_precision["micro"]
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure(figsize=(12, 16))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        # l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        # plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    # lines.append(l)
    # labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='navy', linestyle=':', lw=2)
    lines.append(l)
    labels.append('micro-average ({0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i in range(start_index, nclasses):
        l, = plt.plot(recall[i], precision[i], color=COLORS[i], lw=2)
        lines.append(l)
        labels.append('{0} ({1:0.2f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title is not None:
        plt.title(title, fontsize=34)
    plt.legend(lines, labels, loc=(0.1, -.6), fontsize=24, frameon=True, ncol=2)
    plt.tight_layout()
    figname = os.path.join(fig_dir, 'precision_%s.pdf' % name)
    plt.savefig(figname)
    figname = os.path.join(fig_dir, 'precision_%s.png' % name)
    plt.savefig(figname)
    plt.close()

    return figname, save_auc


def compute_multiclass_roc_auc(classes, y_test, y_pred_prob, name='', fig_dir='.', title=None, logyscale=False):
    """
    Plot multiclass Receiver Operating Characteristic curves.
    """

    if np.nonzero(y_test[:, 0])[0].size == 0:
        start_index = 1
    else:
        start_index = 0

    nclasses = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    save_auc = dict()
    for i in range(start_index, nclasses):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        save_auc[classes[i]] = roc_auc[i]

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    save_auc['micro'] = roc_auc['micro']

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(start_index, nclasses)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(start_index, nclasses):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes[start_index:])

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    save_auc['macro'] = roc_auc['macro']

    # Plot all ROC curves
    fig = plt.figure(figsize=(13, 12))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ({0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='navy', linestyle=':', linewidth=6)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ({0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='deeppink', linestyle=':', linewidth=6)

    lw = 2
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(start_index, nclasses):
        plt.plot(fpr[i], tpr[i], lw=lw, color=COLORS[i],
                 label='{0} ({1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    # # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    if logyscale:
        plt.yscale("log")
    else:
        pass  # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title is not None:
        plt.title(title, fontsize=34)  # plt.title(title, fontsize=70, fontweight="bold", y=1.02) # was size 34
    plt.legend(loc="lower right", frameon=True, fontsize=26)
    plt.tight_layout()
    figname = os.path.join(fig_dir, 'roc_%s_mask_rare.pdf' % name)
    plt.savefig(figname)
    figname = os.path.join(fig_dir, 'roc_%s.png' % name)
    plt.savefig(figname)

    return figname, save_auc


def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.RdBu, fig_dir='.', name='',
                          combine_kfolds=False, show_uncertainties=False):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if combine_kfolds:
        uncertainties = np.std(cm, axis=0)
        cm = np.sum(cm, axis=0)

    if normalize:
        if combine_kfolds:
            uncertainties = uncertainties.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # Multiply off diagonal by -1
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    np.savetxt(os.path.join(fig_dir, 'confusion_matrix_%s.csv' % name), cm)
    print(cm)

    cms = [cm]
    deleterows = [False]
    if np.all(np.isnan(cm[0])):
        cmDelete = np.delete(cm, 0, 0)
        cms.append(cmDelete)
        deleterows.append(True)

    for cm, deleterow in zip(cms, deleterows):
        fig = plt.figure(figsize=(15, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
        # plt.title(title)
        cb = plt.colorbar()
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=27)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=27)
        if deleterow:
            plt.yticks(tick_marks[:-1], classes[1:], fontsize=27)
        else:
            plt.yticks(tick_marks, classes, fontsize=27)

        fmt = '.2f' if normalize else 'd'
        thresh = 0.5  # cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            value = format(abs(cm[i, j]), fmt)
            if combine_kfolds and show_uncertainties:
                unc = format(uncertainties[i, j], fmt)
                cell_text = r"{} $\pm$ {}".format(value, unc)
            else:
                cell_text = value
            if cell_text == 'nan':
                cell_text = '-'
            plt.text(j, i, cell_text, horizontalalignment="center",
                     color="white" if abs(cm[i, j]) > thresh else "black", fontsize=26)

        if title is not None:
            plt.title(title, fontsize=34)  # plt.title(title, fontsize=70, fontweight="bold", y=1.02) # was size 33
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        figname_pdf = os.path.join(fig_dir, 'confusion_matrix_%s_mask_rare.pdf' % name)
        plt.savefig(figname_pdf, bbox_inches="tight")
        if not deleterow:
            figname_png = os.path.join(fig_dir, 'confusion_matrix_%s.png' % name)
            plt.savefig(figname_png, bbox_inches="tight")

    return figname_png
