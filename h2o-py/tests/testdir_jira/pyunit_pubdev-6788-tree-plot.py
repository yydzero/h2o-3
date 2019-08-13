from __future__ import print_function
import sys

sys.path.insert(1, "../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
import matplotlib as plt


def test_tree_plot():
    seed = 123456789
    loan_level = h2o.import_file(pyunit_utils.locate("bigdata/laptop/loan_level_500k.csv"))
    train, valid, test = loan_level.split_frame([0.7, 0.15], seed=42)
    y = "DELINQUENT"
    ignore = ["DELINQUENT", "PREPAID", "PREPAYMENT_PENALTY_MORTGAGE_FLAG", "PRODUCT_TYPE"]
    x = [i for i in train.names if i not in ignore]
    GBM = H2OGradientBoostingEstimator(seed=42, model_id='GBM')
    GBM.train(x, y, train, validation_frame=valid)
    GBM.summary().as_data_frame()
    GBM.plot()
    GBM.varimp_plot(5)


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_tree_plot)
else:
    test_tree_plot()
