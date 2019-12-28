import pyspark
from load_data import LoadDataset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

class ModelBasedRecommender():

	def __init__(self):
		ld = LoadDataset()
		self.ratings_df, self.meta_df = ld.load_files()
		self.transformed_ratings_df = ld.transform_data(self.ratings_df)
		(self.train_df, self.test_df) = ld.split_data(self.transformed_ratings_df)
	
	def train_model(self):
		als = ALS(maxIter=5, regParam=0.01, userCol='user_index', itemCol='item_index', ratingCol='rating',
				coldStartStrategy='drop')
		print(self.train_df)
		model = als.fit(self.train_df)
		return(model)
	
	def test_model(self, model):
		predictions = model.transform(self.test_df)
		evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
										predictionCol="prediction")
		rmse = evaluator.evaluate(predictions)
		print("Root-mean-square error = " + str(rmse))

model_based_rec = ModelBasedRecommender()
model = model_based_rec.train_model()
model_based_rec.test_model(model)