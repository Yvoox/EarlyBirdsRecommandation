# EarlyBirdsRecommandation

Realization of Recommandation Matrix Factorization & SVD:

Project Schedule:
  - Research : 5 hours
  - Implementation & Documentation : 2 hours

- matrixFacto.ipynb : Research doc fully commented
- ./implementation : Contains implementation of research in an API

*: just need to add a ./data & ./implementation/data folder with ratings.csv, movies_metadata.csv and evaluation_ratings.csv 

# API Documentation

Launch : python ./implementation/api.js

Initialization: http://127.0.0.1:5000/startApi and wait 'API INITIALIZED' - create the prediction factorized matrix

Usage:

  - http://127.0.0.1:5000/getUserAlreadySeen?userId=* : return a dataFrame into a JSON object with films already seen by userId
  
  - http://127.0.0.1:5000/getUserRecommandation?userId=*&number=* : return a dataFrame into a JSON object with number recommandation for userId


# Pain Points & Improvement

- The evaluation function in the research part is not good: calculating an RMSE based on Ids never brought anything good!
It would have been preferable to set up an evaluation function based on the number of similarities in the train set.
- Possibility of calculating the factored matrix and storing it in a Spark Parquet File
- Possibility to use a Gradient Descent to compute the factorized Matrix
