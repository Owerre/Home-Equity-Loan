# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Supervised classification models
import xgboost as xgb
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

class SupervisedModels:
    """
    Class for training and testing supervised classification models
    """

    def __init__(self):
        """
        Parameter initialization
        """

    def model_selection_cv(self, model, X_train, y_train, cv_fold, scoring = None):
        """
        Cross-validation on the training set

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold
        scoring: performance metric

        Returns
        _____________
        Performance metrics on the cross-validation training set
        """

        # Fit the training set
        model.fit(X_train, y_train)

        # Compute accuracy on k-fold cross validation
        score = cross_val_score(model, X_train, y_train,cv=cv_fold, scoring = scoring)

        # Make prediction on k-fold cross validation
        y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # Make probability prediction on k-fold cross validation
        y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method='predict_proba')[:,1]

        # Print results
        print('-' * 75)
        print('Cross-validation accuracy (std): %f (%f)' % (score.mean(), score.std()))
        print('AUROC: %f' % (roc_auc_score(y_train, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_train, y_pred_proba)))
        print('Predicted classes:', np.unique(y_cv_pred))
        print('Confusion matrix:\n', confusion_matrix(y_train, y_cv_pred))
        print('Classification report:\n', classification_report(y_train, y_cv_pred))
        print('-' * 75)


    def plot_auc_ap_svm(self, X_train, y_train, cv_fold):
        """
        Plot of cross-validation AUC and AP for SVM

        Parameters
        ___________
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(0,11,2)]
        gamma_list = [2**x for x in range(-5,1,2)]
        auc_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]
        ap_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]

        axes_labels = ['2^0', '2^2', '2^4', '2^6', '2^8', '2^10']
        gamma_labels = ['2^-5', '2^-3', '2^-1']
        plt.rcParams.update({'font.size': 15})
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVC(C = val2, gamma = val1, probability = True, kernel = 'rbf', random_state = 42)
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method='predict_proba')[:,1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax1)
            ap_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax2)

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("AUC", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with SVM".format(cv_fold), fontsize = 15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')

        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("AP", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with SVM".format(cv_fold), fontsize = 15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()

    def plot_auc_ap_lr(self, X_train, y_train, cv_fold):
        """
        Plot of cross-validation AUC and AP for Logistic regression

        Parameters
        ___________
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(-2,9,2)]
        class_wgt_list = [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}]
        auc_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(class_wgt_list))]
        ap_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(class_wgt_list))]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8']
        plt.rcParams.update({'font.size': 15})
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(class_wgt_list):
            for j, val2 in enumerate(C_list):
                model = LogisticRegression(C = val2, class_weight = val1, random_state = 42)
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method='predict_proba')[:,1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(label = "class_weight="+str(class_wgt_list[i]), marker = "o", linestyle = "-", ax = ax1)
            ap_list[i].plot(label = "class_weight="+str(class_wgt_list[i]), marker = "o", linestyle = "-", ax = ax2)

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("AUC", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with Logistic Regression".format(cv_fold), fontsize = 15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')

        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("AP", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with Logistic Regression".format(cv_fold), fontsize = 15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()

    def test_prediction(self, model, X_train, y_train, X_test, y_test):
        """
        Predictions on the test set

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        X_test: feature matrix of the test set
        y_train: training set class labels
        y_test: test set class labels

        Returns
        _____________
        Performance metrics on the test set
        """

        # Fit the training set
        model.fit(X_train, y_train)

        # Make prediction on the test set
        y_pred = model.predict(X_test)

        # Compute the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        # Predict probability
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print('-' * 75)
        print('Test accuracy:  %f' % (accuracy))
        print('AUROC: %f' % (roc_auc_score(y_test, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_test, y_pred_proba)))
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
        print('Classification report:\n', classification_report(y_test, y_pred))
        print('-' * 75)

    def plot_roc_pr_curves(self, model, X_train, y_train, X_test, y_test, cv_fold, color=None, label=None):
        """
        Plot ROC and PR curves for cross-validation and test sets

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: training set class labels
        X_test: feature matrix of the test set
        y_test: test set class labels
        cv_fold: number of k-fold cross-validation
        color: matplotlib color
        label: matplotlib label

        Returns
        _____________
        Matplotlib line plot
        """

        # Fit the model
        model.fit(X_train, y_train)

        ########################## ROC and PR curves for cross-validation set #####################################

        # Make prediction on k-fold cross validation
        y_cv_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method="predict_proba")

        # Compute the fpr and tpr for each classifier
        fpr_cv, tpr_cv, thresholds = roc_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the precisions and recalls for the classifier
        precisions_cv, recalls_cv, thresholds = precision_recall_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the ROC curve for each classifier
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the PR curve for the classifier
        area_prc_cv = auc(recalls_cv, precisions_cv)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(221)
        plt.plot(fpr_cv, tpr_cv, color=color, label=(label) % area_auc_cv)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for the Cross-Validation Training Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(222)
        plt.plot(recalls_cv, precisions_cv, color=color, label=(label) % area_prc_cv)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for the {}-Fold Cross-Validation Training Set'.format(cv_fold))
        plt.legend(loc='best')

        ############################## ROC and PR curves for Test set #####################################
        # Predict probability
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the fpr and tpr for each classifier
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Compute the precisions and recalls for the classifier
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Compute the area under the ROC curve for each classifier
        area_auc = roc_auc_score(y_test, y_pred_proba)

        # Compute the area under the PR curve for the classifier
        area_prc = auc(recalls, precisions)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(223)
        plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for the Test Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(224)
        plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for the Test Set')
        plt.legend(loc='best')

    def plot_aucroc_aucpr(self, model, X_train, y_train, X_test, y_test, 
                        cv_fold,  marker= None, color = None, label = None):
        """
        Plot AUC-ROC  and AUC-PR curves for cross-validation vs. test sets

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: training set class labels
        X_test: feature matrix of the test set
        y_test: test set class labels
        cv_fold: number of k-fold cross-validation
        color: matplotlib color
        marker: matplotlib marker

        Returns
        _____________
        Matplotlib line plot
        """

        # Fit the model
        model.fit(X_train, y_train)

        ############################## AUC-ROC  and AUC-PR for Test set #####################################

        # Predict probability on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the precisions and recalls of the test set
        test_precisions, test_recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Compute the area under the ROC curve of the test set
        area_auc_test = roc_auc_score(y_test, y_pred_proba)

        # Compute the area under the PR curve on the test set
        area_prc_test = auc(test_recalls, test_precisions)

        ########################### AUC-ROC  and AUC-PR cross-validation training set ##############################

        # Make prediction on the k-fold cross-validation set
        y_cv_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method="predict_proba")

        # Compute the precisions and recalls of the cross-validation set
        cv_precisions, cv_recalls, thresholds = precision_recall_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the ROC curve of the cross-validation set
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the PR curve of the cross-validation set
        area_prc_cv = auc(cv_recalls, cv_precisions)

        ############################ Plot #######################################################
        # AUC-ROC 
        plt.subplot(121)
        plt.plot([area_auc_cv], [area_auc_test], color = color, marker = marker, label = label)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0.75,1.001,0.75,1.001])
        plt.xticks(np.arange(0.75,1.001,0.05))
        plt.yticks(np.arange(0.75,1.001,0.05))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-ROC for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')

        # AUC-PR
        plt.subplot(122)
        plt.plot([area_prc_cv], [area_prc_test], color = color, marker = marker, label = label)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0.55,1.001,0.55,1.001])
        plt.xticks(np.arange(0.55,1.01,0.05))
        plt.yticks(np.arange(0.55,1.01,0.05))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-PR for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')


