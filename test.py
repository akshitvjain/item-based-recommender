import pyspark
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer

sc = pyspark.SparkContext('local[*]')
sqlContext = SQLContext(sc)

df = sqlContext.createDataFrame(
	[(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
	["id", "category"])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()

print(sc.getConf.toDebugString)