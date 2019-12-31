import pyspark
import numpy as np
from numpy import linalg as LA
from load_data import LoadDataset
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

class ItemBasedRecommender():

	def __init__(self):
		ld = LoadDataset()
		self.ratings_df, self.meta_df = ld.load_files()
		self.transformed_ratings_df = ld.transform_data(self.ratings_df)
		(self.train_df, self.test_df) = ld.split_data(self.transformed_ratings_df)
	
	def train_model(self):
		als = ALS(maxIter=5, regParam=0.01, userCol='user_index', itemCol='item_index', ratingCol='rating',
				coldStartStrategy='drop', seed=5)
		model = als.fit(self.train_df)
		itemFactors = model.itemFactors
		return(model, itemFactors)
	
	def test_model(self, model):
		predictions = model.transform(self.test_df)
		evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
										predictionCol="prediction")
		rmse = evaluator.evaluate(predictions)
		print("Root-mean-square error = " + str(rmse))

	def compute_similarity(self, itemFactors, movie_id):
		get_movie_id = movie_id
		item = itemFactors.where(col('id') == get_movie_id).select(col('features'))
		item_features = item.rdd.map(lambda x: x.features).first()
		
		# map each row: [item, features] to compute cosine similarity with given item
		# merge the output dataframe with train_df to get the actual movie_id,
		# get the list of movies from the movie_ids

	def _cosine_similarity(self, vector_1, vector_2):
		v1 = np.asarray(vector_1)
		v2 = np.asarray(vector_2)
		cs = v1.dot(v2) / (LA.norm(v1) * LA.norm(v2))
		return(cs)



movie_id = int(input('Enter movie_id (0-9723) to generate similar recommendations: '))
item_based_rec = ItemBasedRecommender()
model, itemFactors = item_based_rec.train_model()
item_based_rec.test_model(model)
item_based_rec.compute_similarity(itemFactors, movie_id)
