####################################
# Author: S. A. Owerre
# Date modified: 12/03/2021
####################################

import warnings
warnings.filterwarnings("ignore")

# Pyspark modules
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import  BinaryClassificationEvaluator

# Data manipulation & visualization
import matplotlib.pyplot as plt
import numpy as np

# scikit-learn performance metrics
from sklearn.metrics import confusion_matrix, classification_report

class TransformationPipeline:
    """
    A class for transformation pipelines in PySpark
    """

    def __init__(self, label_col):
        """
        Define default parameters
        """
        self.label_col = label_col
        
    def preprocessing(self, trainDF, validDF, testDF):
        """
        Data preprocessing steps involving  the following transformations:

        1. One-Hot encoding of categorical variables
        2. Imputation of missing values in numerical variables
        3. Standardization of numerical variables

        Parameters
        -----------
        trainDF: training data set
        validDF: test data set
        testDF: test data set

        Returns
        -----------
        Transformed training and test data sets with the assembler vector
        """
        # Extract numerical and categorical column names
        cat_cols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
        num_cols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & \
                    (field != self.label_col))]

        # Create output columns
        index_output_cols = [x + "Index" for x in cat_cols]
        ohe_output_cols = [x + "OHE" for x in cat_cols]
        # num_output_cols = [x + "scaled" for x in num_cols]

        # strinf indexer for categorical variables
        s_indexer = StringIndexer(inputCols = cat_cols, outputCols = index_output_cols, 
                                    handleInvalid="skip")

        # One-hot code categorical columns
        cat_encoder = OneHotEncoder(inputCols = index_output_cols, outputCols = ohe_output_cols)

        # Impute missing values in numerical columns
        num_imputer = Imputer(inputCols = num_cols, outputCols = num_cols)

        # Vector assembler
        assembler_inputs = ohe_output_cols + num_cols
        assembler = VectorAssembler(inputCols = assembler_inputs, outputCol = "unscaled_features")

        # Features scaling using StandardScaler
        scaler = StandardScaler(inputCol = assembler.getOutputCol(), outputCol = "features")
        
        # Create pipeline
        stages = [s_indexer, cat_encoder, num_imputer, assembler, scaler]
        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(trainDF)

        # Preprocess training and test data
        trainDF_scaled = pipelineModel.transform(trainDF)
        validDF_scaled = pipelineModel.transform(validDF)
        testDF_scaled = pipelineModel.transform(testDF)
        return assembler, trainDF_scaled, validDF_scaled, testDF_scaled

    def one_val_imputer(self, df, cols, impute_with):
        """
        Impute column(s) with one specific value

        Parameters
        ----------
        df: spark dataframe
        num_cols: list of column name(s)
        impute_with: imputation value

        Returns
        --------
        Dataframe with imputed column(s) 
        """
        df = df.fillna(impute_with, subset=cols)
        return df
    
    def df_to_numeric(self, df, cat_cols):
        """
        Convert numerical columns to double type

        Parameters
        ----------
        df: spark dataframe
        cat_cols: list of true categorical column names

        Returns
        --------
        Transformed spark dataframe
        """
        cols = [x for x in df.columns if x not in cat_cols]
        for col in cols:
            df = df.withColumn(col, df[col].cast(DoubleType()))
        return df

    def print_eval_metrics(self, model_pred, model_nm = None):
        """
        Print performance metrics

        Parameters
        -----------
        model_pred: model prediction dataframe
        model_nm: name of the model

        Returns
        -----------
        Print metrics
        """
        # Extract true and predicted labels
        y_true = np.array(model_pred.select(self.label_col).toPandas())
        y_pred = np.array(model_pred.select('prediction').toPandas())

        # Compute AUROC and AUPR
        eval =  BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', 
                                                labelCol=self.label_col,
                                                 metricName="areaUnderROC")
        AUROC = eval.evaluate(model_pred)
        AUPRC = eval.evaluate(model_pred, {eval.metricName: "areaUnderPR"})

        # Print results
        print('Performance metrics for {}'.format(str(model_nm)))
        print('-' * 60)
        print('AUROC: %f' % AUROC)
        print('AUPRC: %f' % AUPRC)
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print('Classification report:\n', classification_report(y_true, y_pred))
        print('-' * 60)

    def plot_roc_pr_curves(self, model, model_pred, title = None, label=None):
        """
        Plot ROC and PR curves for training set

        Parameters
        ------------
        model: trained supervised  model
        model_pred: model prediction dataframe
        title: matplotlib title
        label: matplotlib label

        Returns
        ------------
        Matplotlib line plot
        """
        # Compute the fpr and tpr
        pdf_roc = model.summary.roc.toPandas()

        # Compute the recall and precision
        pdf_pr = model.summary.pr.toPandas()

        # Compute AUROC and AUPR
        eval =  BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', 
                                                labelCol=self.label_col, 
                                                metricName="areaUnderROC")
        area_auc_cv = eval.evaluate(model_pred)
        area_prc_cv = eval.evaluate(model_pred, {eval.metricName: "areaUnderPR"})

        # ROC curve
        plt.subplot(121)
        plt.plot(pdf_roc['FPR'], pdf_roc['TPR'], color= 'b', label=(label) % area_auc_cv)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for {}'.format(str(title)))
        plt.legend(loc='best')

        # PR curve
        plt.subplot(122)
        plt.plot(pdf_pr['recall'],pdf_pr['precision'], color= 'b', label=(label) % area_prc_cv)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for {}'.format(str(title)))
        plt.legend(loc='best')