import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType

rating_file_path = 'data/ratings_Clothing_Shoes_and_Jewelry.csv'
meta_file_path = 'data/meta.json'

class LoadDataset():

	def __init__(self):
		self.sc = pyspark.SparkContext('local[*]')
		self.sqlContext = SQLContext(self.sc)

	def load_files(self):
		# read ratings csv, build schema struct
		csv_struct = StructType([StructField('user', StringType(), True), 
								StructField('item', StringType(), True),
								StructField('rating', StringType(), True), 
								StructField('timestamp', StringType(), True)])
		ratings_df = self.sqlContext.read.csv(rating_file_path, schema=csv_struct)
		# read metadata for each product
		meta_df = self.sqlContext.read.json(meta_file_path)
		meta_df = meta_df.select('asin', 'imUrl')
		return(ratings_df, meta_df)

ld = LoadDataset()
ratings_df, meta_df = ld.load_files()
(training, test) = ratings_df.randomSplit([0.8, 0.2])
print(training.count())
print(test.count())
