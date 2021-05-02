import warnings
warnings.filterwarnings("ignore")

# Data visualization
import matplotlib.pyplot as plt

# Pyspark modules
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import  BinaryClassificationEvaluator

class TransformationPipeline:
    """
    A class for transformation pipelines in PySpark
    """

    def __init__(self, label_col):
        """
        Define parameters
        """
        self.label_col = label_col

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
    
    def df_to_numeric(self, df, dont_cols):
        """
        Convert numerical columns to double type
        """
        cols = [x for x in df.columns if x not in dont_cols]
        for col in cols:
            df = df.withColumn(col, df[col].cast(DoubleType()))
        return df
        
    def preprocessing(self, trainDF, validDF, testDF):
        """
        Data preprocessing steps involving  the following transformations

        1. One-Hot encode categorical variables
        2. Impute missing values in numerical variables
        3. Standardize numerical variables

        Parameters
        -----------
        trainDF: training data set
        validDF: test data set
        testDF: test data set
        label_col: column name for the labels or target variable

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

    def eval_metrics(self, model_pred, model_nm):
        """
        Print regression evaluation metrics

        Parameters
        -----------
        model_pred: model prediction
        model_nm: name of the model
        label_col: column name for the labels or target variable

        Returns
        -----------
        Print metrics
        """
        eval =  BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', 
                                                labelCol=self.label_col,
                                                 metricName="areaUnderROC")

        AUROC = eval.evaluate(model_pred)
        AUPRC = eval.evaluate(model_pred, {eval.metricName: "areaUnderPR"})

        print("Performance metrics for {}".format(str(model_nm)))
        print('-'*60)
        print("AUROC: %.3f" % AUROC)
        print("AUPRC: %.3f" % AUPRC)
    
    def plot_roc_pr_curves(self, model, model_pred, title, label=None):
        """
        Plot ROC and PR curves for training set

        Parameters
        ___________
        model: trained supervised  model
        cv_fold: number of k-fold cross-validation
        color: matplotlib color
        label: matplotlib label

        Returns
        _____________
        Matplotlib line plot
        """
        # Compute the fpr and tpr for each classifier
        pdf_roc = model.summary.roc.toPandas()

        # Compute the recall and precision for each classifier
        pdf_pr = model.summary.pr.toPandas()

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