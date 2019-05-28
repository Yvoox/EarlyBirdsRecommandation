import pandas as pd
import re
import numpy as np
from scipy.sparse.linalg import svds

from functions import *

from flask import (Flask,request)
from flask_cors import CORS

global evaluations,simplifiedMovies,ratings,predDf

app = Flask(__name__)
CORS(app)

path = './data/'

@app.route('/startApi')
def startApi():
    global evaluations,simplifiedMovies,ratings,predDf
    evaluations,simplifiedMovies,ratings,predDf=initAPI()

    return 'API INITIALIZED'

@app.route('/getUserAlreadySeen')
def getUserAlreadySeen():
    global evaluations,simplifiedMovies,ratings,predDf
    userId = int(request.args.get('userId'))

    return alreadySeen(userId,ratings,simplifiedMovies).to_json(orient='split')

@app.route('/getUserRecommandation')
def getUserRecommandation():
    global evaluations,simplifiedMovies,ratings,predDf
    userId = int(request.args.get('userId'))
    number = int(request.args.get('number'))

    return makeReco(userId,number,predDf,simplifiedMovies,ratings).to_json(orient='split')

if __name__ == '__main__':
    app.run(debug=True)
