import pandas as pd
import re
import numpy as np
from scipy.sparse.linalg import svds

data='./data/'

def cleanedGenre(row):
    cleanned=''
    if row['genres']!='[]':
        splitted = row['genres'].split('},')
        for i in range(len(splitted)):
            m=re.search('([A-Z])\w+', splitted[i])
            cleanned+=m.group(0).replace("'",'')
            if  i+1<len(splitted):
                cleanned+='|'
        return cleanned
    else:
        return ''



def importFiles():
    evaluations = pd.read_csv(data+'evaluation_ratings.csv')
    movies = pd.read_csv(data+'movies_metadata.csv')
    ratings = pd.read_csv(data+'ratings.csv')
    return evaluations,movies,ratings

def formatDataframes(evaluations,movies,ratings):
    simplifiedMovies = movies[['id','title','genres']]
    simplifiedMovies['Genres'] = simplifiedMovies.apply(lambda x: cleanedGenre(x),axis=1)
    simplifiedMovies.drop(['genres'],axis=1,inplace=True)
    simplifiedMovies=simplifiedMovies.dropna()
    simplifiedMovies['id']=pd.to_numeric(simplifiedMovies['id'])
    return evaluations,simplifiedMovies,ratings

def createMatrix(ratings,limit,fillType=True):
    #fillType : used to fillna of Factorized Matrix
    #   True : fillna with users mean
    #   False : fillna with zero
    #   Default: True
    #
    #limit: used to limit number of ratings rows taking into account (in case of limited computation Power)
    #   Server case: len(ratings)
    #   laptop case: 5000


    if fillType:
        matrix = pd.pivot_table(ratings.head(limit),index='userId',columns='movieId',values='rating')
        matrix = matrix.fillna(matrix.mean())
    else:
        matrix = pd.pivot_table(ratings.head(limit),index='userId',columns='movieId',values='rating',fill_value=0)
    #Convert to np array
    R = matrix.as_matrix()

    #Normalize by each users mean
    means = np.mean(R,axis=1)
    Rnormalized = R-means.reshape(-1, 1)
    return Rnormalized,matrix,means

def generatePredictedMatrix(Rnormalized,matrix,means,K):
    U, sig, Vt = svds(Rnormalized, k = K)
    sig = np.diag(sig) # svds return only values for sig, need to diagonalize (see svds doc.)
    #Generate prediction and add mean to return on a */5 value - denormalize
    predictedArray = np.dot(np.dot(U, sig), Vt) + means.reshape(-1, 1)
    predDf = pd.DataFrame(predictedArray, columns = matrix.columns)
    return predDf

def initAPI():
    evaluations,movies,ratings=importFiles()
    evaluations,simplifiedMovies,ratings=formatDataframes(evaluations,movies,ratings)
    Rnormalized,matrix,means=createMatrix(ratings,5000,True)
    predDf = generatePredictedMatrix(Rnormalized,matrix,means,30)
    return evaluations,simplifiedMovies,ratings,predDf

def alreadySeen(userId,ratings,movies):

    useId = userId - 1 # UserID starts at 1, shift for entry convenience
    alreadySeen = ratings[ratings.userId == (userId)]

    #Just add Film informations
    alreadySeen = alreadySeen.merge(movies, how = 'left', left_on = 'movieId', right_on = 'id').sort_values(['rating'], ascending=False)
    return alreadySeen.dropna()

def makeReco(userId,nbreco,pred,movies,ratings):
    useId = userId - 1 # UserID starts at 1, shift for entry convenience

    #Generate Movie list rated and predicted for this user
    userPred = pred.iloc[useId].sort_values(ascending=False)
    userPred = pd.DataFrame(userPred).reset_index()

    userSeen = alreadySeen(userId,ratings,movies)



    unseen = movies[~movies['id'].isin(userSeen['movieId'])]

    #Keep nbreco best predicted rated unseen films
    reco = (unseen.merge(userPred, how = 'left',left_on = 'id',right_on = 'movieId').
         rename(columns = {useId: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:nbreco, :-1]
                      )


    return reco
