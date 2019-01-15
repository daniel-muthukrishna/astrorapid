import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from chainconsumer import ChainConsumer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

font = {'family': 'normal',
        'size': 34}

matplotlib.rc('font', **font)

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
        pass # plt.ylim([0.0, 1.05])
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


def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.RdBu, fig_dir='.', name='', combine_kfolds=False, show_uncertainties=False):
    """
    This function prints and plots the confusion matrix.
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


def plot_feature_importance(classifier, feature_names, num_features, fig_dir, num_features_plot=50, name=''):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    num_features = min(num_features, len(indices))
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_features):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    n = num_features_plot
    if n > num_features:
        n = num_features
    # Plot the feature importances of the forest
    fig = plt.figure(figsize=(20, 10))
    plt.bar(range(n), importances[indices][:n], color='tab:blue', yerr=std[indices][:n], align="center")
    plt.xticks(range(n), feature_names[indices][:n], rotation=90)
    plt.tick_params(axis='both', labelsize=24)
    plt.xlim([-1, n])
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance{}.pdf'.format(name)))


def scatter(x, colors, feature_names, models, sntypes_map):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

    xaxisclass = 0
    yaxisclass = 2

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,xaxisclass], x[:,yaxisclass], lw=0, s=10,
                    c=palette[colors.astype(np.int)], alpha=0.4)
    plt.xlim(min(x[:,xaxisclass]), max(x[:,yaxisclass]))
    plt.ylim(min(x[:,xaxisclass]), max(x[:,yaxisclass]))
    # ax.axis('off')
    ax.axis('tight')
    plt.xlabel(feature_names[xaxisclass])
    plt.ylabel(feature_names[yaxisclass])

    # We add the labels for each digit.
    txts = []
    for i, model in enumerate(models):
        model_name = sntypes_map[model]
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :][:,[xaxisclass,yaxisclass]], axis=0)
        txt = ax.text(xtext, ytext, model_name, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir, add_save_name=''):
    colors = np.copy(y)
    for i, m in enumerate(models):
        colors[y == m] = i

    scatter(X, colors, feature_names, models, sntypes_map)
    plt.savefig(fname=os.path.join(fig_dir, 'feature_space_sns_%s' % add_save_name), dpi=120)
    plt.show()

    print(X.shape, y.shape)
    for i in range(len(X[0])):
        mask = np.where(X[:,i] != 0)[0]
        X = X[mask]
        y = y[mask]
    print(X.shape, y.shape)

    for i, f in enumerate(feature_names):
        feature_names[i] = "$" + f + "$"

    c = ChainConsumer()
    for model in models[::-1]:
        model_name = sntypes_map[model]
        c.add_chain(X[y == model], parameters=list(feature_names), name=model_name)
    c.configure(colors=["#B32222", "#D1D10D", "#455A64", "#1f77b4", "#2ca02c"], shade=True, shade_alpha=0.2, bar_shade=True)
    fig = c.plotter.plot()
    fig.savefig(fname=os.path.join(fig_dir, 'feature_space_contours_%s.pdf' % add_save_name), transparent=False)
