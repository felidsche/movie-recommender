# movie-recommender
A movie recommendation system built using Apache Spark’s ML library

## Setup on your local machine
### Download Apache Spark 2.4.6 distribution pre-built for Apache Hadoop 2.7 [link](http://spark.apache.org/downloads.html).
- unpack the archive
- set the `$SPARK_HOME` environment variable `export SPARK_HOME=$(pwd)`
### add the Apache Spark librariers to an IDE (i.e. PyCharm)
- navigate to `PyCharm → Preferences ... → Project spark-demo → Project Structure → Add Content Root` in the main menu
- select all `.zip` files from `$SPARK_HOME/python/lib` 
- click apply and save changes

### create a new run configuration
- navigate to `Run → Edit Configurations → + → Python` in the main menu
- select `movie_recommendation.py` for `Script`
- name it `movie_recommendation`
### add environment variables in the run configuration
- `PYSPARK_PYTHON=python3`
- `PYTHONPATH=$SPARK_HOME/python`
- `PYTHONUNBUFFERED=1`

### provide the input data
- `movies.csv` and `ratings.csv`need to be under `../data/*.csv` relative to the script path

### run the script within Apache Spark context
- click `Run → Run 'movie_recommendation'` in the main menu

### check the [webUI](http://localhost:4040) to monitor a running Apache Spark job

