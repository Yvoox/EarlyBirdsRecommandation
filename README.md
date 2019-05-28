# EarlyBirdsRecommandation

Realization of Recommandation Matrix Factorization & SVD:

- matrixFacto.ipynb : Research doc fully commented
- ./implementation : Contains implementation of research in an API

*: just need to add a ./data & ./implementation/data folder with ratings.csv, movies_metadata.csv and evaluation_ratings.csv 

# API Documentation

Launch : python ./implementation/api.js

Initialization: http://127.0.0.1:5000/startApi and wait 'API INITIALIZED' - create the prediction factorized matrix

Usage:

  - http://127.0.0.1:5000/getUserAlreadySeen?userId=* : return a dataFrame into a JSON object with films already seen by userId
  
  - http://127.0.0.1:5000/getUserRecommandation?userId=*&number=* : return a dataFrame into a JSON object with number recommandation for userId
