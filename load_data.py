import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

#rating_file_path = 'data/ratings_Clothing_Shoes_and_Jewelry.csv'
rating_file_path = 'data/ratings.csv'
meta_file_path = 'data/meta.json'

class LoadDataset():

	def __init__(self):
		self.sc = pyspark.SparkContext('local[*]')
		self.sqlContext = SQLContext(self.sc)

	def load_files(self):
		# read ratings csv, build schema struct
		csv_struct = StructType([StructField('user', StringType(), True), 
								StructField('item', StringType(), True),
								StructField('rating', DoubleType(), True),
								StructField('timestamp', StringType(), True)])
		ratings_df = self.sqlContext.read.csv(rating_file_path, schema=csv_struct) 
		ratings_df = ratings_df.select('user', 'item', 'rating')

		# read metadata for each product
		meta_df = self.sqlContext.read.json(meta_file_path)
		meta_df = meta_df.select('asin', 'imUrl')
		return(ratings_df, meta_df)

	def transform_data(self, df):
		indexers = [StringIndexer(inputCol=column, outputCol=column+'_index', handleInvalid='skip')
					for column in list(set(df.columns)-set(['rating']))]
	
		pipeline = Pipeline(stages=indexers)
		transformed = pipeline.fit(df).transform(df)
		return(transformed)

	def split_data(self, df):
		(training, test) = df.randomSplit([0.8, 0.2], seed=5)
		return(training, test)

	def get_sql_context(self):
		return self.sqlContext
