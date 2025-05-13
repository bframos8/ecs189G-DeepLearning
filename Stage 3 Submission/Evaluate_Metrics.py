'''
Concrete Evaluate class for a specific evaluation metrics
'''
from networkx.classes import non_edges

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class Evaluate_Metrics(evaluate):
    data = None
    accuracy = []
    precision = []
    recall = []
    f1score = []

    def evaluate(self):
        """
                Called once (by Setting_….) after training+testing to produce
                the final test‐set metrics. Does NOT touch the training lists.
                """
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, prec, rec, f1

    def evaluateAccuracy(self):
        print('evaluating accuracy...')
        self.accuracy.append(accuracy_score(self.data['true_y'], self.data['pred_y']))
        return self.accuracy[-1]

    def evaluatePrecision(self, type):
        print('evaluating precision...')
        self.precision.append(precision_score(self.data['true_y'], self.data['pred_y'], average = type, zero_division=0))
        return self.precision[-1]

    def evaluateRecall(self, type):
        print('evaluating recall...')
        self.recall.append(recall_score(self.data['true_y'], self.data['pred_y'], average = type))
        return self.recall[-1]

    def evaluateF1(self, type):
        print('evaluating F1 score...')
        self.f1score.append(f1_score(self.data['true_y'], self.data['pred_y'], average = type))
        return self.f1score[-1]

    def setPlotData(self, epochs, data, label):
        plt.plot(epochs, data, label=label)

    def setPlotLabels(self, x_label, y_label, title):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

    def showPlot(self):
        plt.legend()
        plt.grid(True)
        plt.savefig("metrics_plot_firstArc.png")
        plt.show()