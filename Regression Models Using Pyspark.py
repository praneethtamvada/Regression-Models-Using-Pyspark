from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import six
import random

if __name__ == "__main__":
    sc = SparkContext()
    sqlC = SQLContext(sc)
    responses = sqlC.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('auto-miles-per-gallon.csv')
    responses.show(1)
    # print(autoDF)

    # autoDF.show()
    # autoDF.cache()
    # autoDF.printSchema()
    auto_df = responses.withColumn('HORSEPOWER', regexp_replace('HORSEPOWER', '\?', '104'))
    auto_df.show(3)

    auto_df = auto_df.withColumn("HORSEPOWER", auto_df["HORSEPOWER"].cast(IntegerType()))


    print(auto_df.describe().toPandas().transpose())
    #auto_df.avg("HORSEPOWER").show()

    for i in auto_df.columns:
        if not (isinstance(auto_df.select(i).take(1)[0][0], six.string_types)):
            print("Correlation to MPG for ", i, auto_df.stat.corr('MPG', i))

    auto_dfd = auto_df.drop('NAME')
    #auto_dfd = auto_df.drop('HORSEPOWER')
    print(auto_dfd)

    # Prepare data for Machine Learning. And we need two columns only
    # — features and label(“MPG”):

    vectorAssembler = VectorAssembler(inputCols=['CYLINDERS', 'DISPLACEMENT', 'HORSEPOWER', 'WEIGHT', 'ACCELERATION', 'MODELYEAR'],outputCol='features')
    vauto_df = vectorAssembler.transform(auto_dfd)
    vauto_df = vauto_df.select(['features', 'MPG'])
    #print(vauto_df)
    vauto_df.show(3)

    # Test and Train Split
    random.seed(100)
    splits = vauto_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    #############################---LINEAR REGRESSION---##################################

    lr = LinearRegression(featuresCol='features', labelCol='MPG', maxIter=10)
    lr_model = lr.fit(train_df)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    # Summarize the model over the training set and print out some metrics

    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    # Predicting on test dataset

    lr_predictions = lr_model.transform(test_df)
    lr_predictions.select("prediction", "MPG", "features").show(5)
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="r2")
    print("R Squared (R2) for Linear Regression on test data = %g" % lr_evaluator.evaluate(lr_predictions))

    # RMSE on test data
    test_result = lr_model.evaluate(test_df)
    print("Root Mean Squared Error (RMSE) for Linear Regression on test data = %g\n" % test_result.rootMeanSquaredError)

    #############################---DECISION TREE REGRESSION---##################################

    dt = DecisionTreeRegressor(featuresCol='features', labelCol='MPG')
    decisionTree_model = dt.fit(train_df)
    decisionTree_model_predictions = decisionTree_model.transform(test_df)
    decisionTree_model_evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName="rmse")
    rmse = decisionTree_model_evaluator.evaluate(decisionTree_model_predictions)
    print("Root Mean Squared Error (RMSE) for Decision Tree on test data = %g" % rmse)
    r2_dt = ecisionTree_model_evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction",
                                                              metricName="r2")
    print("R Squared (R2) for Decision Tree on test data = %g" % r2_dt.evaluate(decisionTree_model_predictions))

    ############################---RANDOM FOREST REGRESSION---##################################

    train_rdd_rf = train_df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))
    test_rdd_rf = test_df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

    RandomForest_model = RandomForest.trainRegressor(train_rdd_rf, categoricalFeaturesInfo={},
                                                     numTrees=50, featureSubsetStrategy="auto", maxDepth=10,
                                                     maxBins=100)

    predictions = RandomForest_model.predict(test_rdd_rf.map(lambda x: x.features))
    labelsAndPredictions = test_rdd_rf.map(lambda lp: lp.label).zip(predictions)
    metrics = RegressionMetrics(labelsAndPredictions)
    print("RMSE of randomForest on Test data = %s" % metrics.rootMeanSquaredError)
    print("R-squared of randomForest on Test data = %s" % metrics.r2)


