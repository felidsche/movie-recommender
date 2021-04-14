import pyspark
from pyspark import SQLContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.mllib.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import sys
import time
import pandas
from lenskit.algorithms.basic import Bias


class MovieRecommendation():
    # constructor
    def __init__(self, conf, spark, sc):
        self.conf = conf
        self.spark = spark
        self.sc = sc

    def create_dataframe(self, path, type):
        """
        read csv into a DataFrame
        """
        sqlContext = SQLContext(self.sc)
        if type.lower() == 'movies':
            schema = StructType([
                StructField('movieId', IntegerType(), False),
                StructField('title', StringType(), False),
                StructField('genres', StringType(), False),
            ])
            # sqlContext so it knows how to do SQL on it
            df = sqlContext.read \
                .format('csv') \
                .option('delimiter', ',') \
                .option('header', 'true') \
                .load(path=path, schema=schema)
            # pipe needs to be escaped
            df = df.withColumn("genres", F.split(
                F.col("genres"), "\\|").cast("array<string>"))

            return df

        elif type.lower() == 'ratings':

            # sqlContext so it knows how to do SQL on it
            # DataFrame needs to infer schema otherwise null values
            df = sqlContext.read \
                .format('csv') \
                .option('delimiter', ',') \
                .option('header', 'true') \
                .option('inferSchema', 'true') \
                .load(path=path)
            return df

        else:
            print("please specify a valid type.")
            sys.exit()

    def explore_dataset(self, movies, ratings):

        movies = movies.alias("movies")
        ratings = ratings.alias("ratings")

        print("The names of the top-10 movies with the largest number of ratings")
        df_top_10_ratings = movies.join(ratings, F.col("movies.movieId") == F.col("ratings.movieId")) \
            .groupBy("movies.movieId", "movies.title") \
            .agg({"rating": "count"}) \
            .alias("rating") \
            .orderBy("rating.count(rating)", ascending=0) \
            .select("rating.title", "rating.count(rating)") \
            .withColumnRenamed("count(rating)", "num_ratings") \
            .limit(10) \
            .show()

        print("The names of the top-10 movies with the highest average rating grouped by genre")
        # explode array column genres into multiple rows
        df_top_10_ratings_genre = movies.join(ratings, F.col("movies.movieId") == F.col("ratings.movieId")) \
            .select(F.explode("movies.genres").alias("genre"), "ratings.rating", "movies.title", "movies.movieId") \
            .groupBy("genre", "movies.title") \
            .agg({"rating": "avg"}) \
            .alias("rating")

        # add row number and filter for row_number = 1
        df_top_10_ratings_genre = df_top_10_ratings_genre \
            .withColumn("row_number", F.row_number().over(
                Window.partitionBy("rating.genre").orderBy(F.col("rating.avg(rating)").desc()))) \
            .filter(F.col("row_number") == 1) \
            .select("rating.genre", "rating.avg(rating)", "rating.title") \
            .limit(10) \
            .show()

        print("The common support for all pair of movies")
        # get all pairs of ratings (cartesian product)
        df_ratings1 = ratings.select("userId", "movieId", "rating") \
            .withColumnRenamed("movieId", "movieId1") \
            .withColumnRenamed("userId", "userId1") \
            .withColumnRenamed("rating", "rating1")
        df_ratings2 = ratings.select("userId", "movieId", "rating") \
            .withColumnRenamed("movieId", "movieId2") \
            .withColumnRenamed("userId", "userId2") \
            .withColumnRenamed("rating", "rating2")

        # cross join to get all pairs (using a limit to save resources on my local machine)
        df_rating_pairs = df_ratings1.crossJoin(df_ratings2) \
            .filter(F.col("userId1") != F.col("userId2")) \
            .filter(F.col("movieId1") != F.col("movieId2")) \
            .limit(5000) \
            .groupBy("movieId1", "movieId2") \
            .agg(F.count(F.lit(1)).alias("num_users")) \
            .filter(F.col("num_users") > F.lit(1)) \
            .show()
        pass

    def bias_predictor(self, ratings_path):
        print("Baseline predictor to recommend movies")
        # start runtime timer
        t0 = time.time()
        # create pandas DataFrame for ratings
        ratings = pandas.read_csv(ratings_path)
        ratings = ratings.drop("timestamp", axis=1)
        ratings = ratings.rename(columns={"userId": "user", "movieId": "item"})
        bias = Bias(items=True, users=True, damping=0.0)
        bias = bias.fit(ratings)
        df = bias.transform(ratings)

        # get a list of all userIds to loop over them
        userIds = df["user"].unique().tolist()

        # get a list of all movies for predictions
        movieIds = df["item"].unique().tolist()
        scores = None
        first = True
        for u in userIds:

            # create the base DataFrame for the first movie
            if first:
                scores = bias.predict_for_user(
                    user=u, items=movieIds).reset_index()
                scores["userId"] = u
                first = False
            # append to the base DataFrame for all other movies
            else:
                it_scores = bias.predict_for_user(
                    user=u, items=movieIds).reset_index()
                it_scores["userId"] = u
                scores = scores.append(it_scores)
        # stop the runtime timer
        t1 = time.time()
        total_time = t1 - t0
        print("Efficiency (runtime) of prediction: " +
              str(total_time) + " seconds.")
        # (using a limit to save resources on my local machine)
        scores = scores.head(5000)
        # output the DataFrame
        scores = scores.rename(
            columns={"index": "movieId", 0: "predicted_rating"})
        scores['row_number'] = scores.sort_values(['predicted_rating'], ascending=False) \
            .groupby(['userId']) \
            .cumcount() + 1
        # filter out ratings > 5.0 since prediction is unbounded
        scores = scores[scores.predicted_rating <= 5.0]
        # get only the highest rating
        scores = scores[scores.row_number == scores.row_number.min()]

        return scores

    def col_filtering(self, ratings_path, movies):
        print("Collaborative filtering to recommend movies")
        # start runtime timer
        t0 = time.time()

        # load the data into an RDD
        lines = spark.read.text(ratings_path).rdd
        # drop header row
        header = lines.first()
        lines = lines.filter(lambda line: line != header)
        parts = lines.map(lambda row: row.value.split(","))
        ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                             rating=float(p[2]), timestamp=int(p[3])))
        ratings = spark.createDataFrame(ratingsRDD)
        # split DataFrame into 80% training data and 20% testing data
        (training, test) = ratings.randomSplit([0.8, 0.2])

        # Build the recommendation model using ALS on the training data
        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
                  coldStartStrategy="drop")
        model = als.fit(training)

        # Evaluate the model by computing the RMSE on the test data
        predictions = model.transform(test)
        # stop the runtime timer
        t1 = time.time()
        total_time = t1 - t0
        print("Efficiency (runtime) of prediction: " +
              str(total_time) + " seconds.")
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        # measure the error of our model in predicting
        print("Root-mean-square error = " + str(rmse))
        accuracy = 100 * float(rmse) / float(5.0)
        print("Accuracy in % = " + str(accuracy))

        # get the total number of rated movies
        num_movies = ratings.select("movieId").distinct().count()
        # Generate a predicted rating for each movie for each user
        userRecs = model.recommendForAllUsers(num_movies)

        # ------ Output the DataFrame -------- #
        # explode the recommendations array to filter it
        userRecs = userRecs.withColumn(
            "prediction", F.explode(userRecs.recommendations))
        # filter out only recommendations in the range of 0.5 to 5.0 since regression is unbounded
        userRecs = userRecs.filter(userRecs.prediction.rating <= 5.0)
        # get only the first of the exploded predictions
        userRecs = userRecs.withColumn("row_number", F.row_number().over(
            Window.partitionBy(userRecs.userId).orderBy(userRecs.prediction.rating.desc()))) \
            .filter(F.col("row_number") == 1)
        # join with movies df to get the title of the recommended movie and select only relevant columns
        userRecs = userRecs.join(movies, userRecs.prediction.movieId == movies.movieId) \
            .select(userRecs.userId, userRecs.prediction, movies.title) \
            .orderBy(F.col("userId"))
        return userRecs


if __name__ == '__main__':
    # Spark session & context
    conf = pyspark.SparkConf().set('spark.driver.host', '127.0.0.1')
    spark = SparkSession.builder.master('local').getOrCreate()
    sc = spark.sparkContext

    mr = MovieRecommendation(conf=conf, spark=spark, sc=sc)

    movies_csv = '../data/movies.csv'
    ratings_csv = '../data/ratings.csv'

    # load csvs into DataFrames
    df_movies = mr.create_dataframe(path=movies_csv, type='movies')
    df_ratings = mr.create_dataframe(path=ratings_csv, type='ratings')

    mr.explore_dataset(movies=df_movies, ratings=df_ratings)

    # movie recommender systems
    df = mr.bias_predictor(ratings_path=ratings_csv)
    print("The baseline predictor DataFrame result: \n")
    print(df.head())

    df = mr.col_filtering(ratings_path=ratings_csv, movies=df_movies)
    print("The collaborative filtering DataFrame result: \n")
    df_out = df.select(df.userId, df.title)
    df_out.show()
    # reduce partitions to 1 to only output 1 file
    df_out.coalesce(1).write.csv('output.csv')

    # terminate the spark job
    spark.stop()
