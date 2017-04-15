import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds


def loadinputData(input_file,test_size=0.25):

    header = ['userId', 'movieId', 'rating', 'timestamp']
    input_data = pd.read_csv(input_file,sep='\t', names=header)
    print('>> %s Moving Ratings Loaded' %len(input_data.index))

    n_users = input_data.userId.unique().shape[0]
    n_items = input_data.movieId.unique().shape[0]
    print('>> Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    train_data, test_data = train_test_split(input_data, test_size=test_size)
    print('>> Training Size : %s Testing Size: %s' %(train_data.size, test_data.size))

    # training and testing user-movie matrix
    train_data_matrix = np.zeros((n_users, n_items))
    for row in train_data.itertuples():
        train_data_matrix[row[1] - 1, row[2] - 1] = row[3]
    print('>> Training Matrix Created')

    test_data_matrix = np.zeros((n_users, n_items))
    for row in test_data.itertuples():
        test_data_matrix[row[1] - 1, row[2] - 1] = row[3]
    print('>> Test Matrix Created')

    return train_data_matrix, test_data_matrix

def similarity(matrix, metrics, kind = 'user'):

    # metrics options: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
    if kind == 'user':
        similarity = pairwise_distances(matrix, metric=metrics)
    if kind == 'item':
        similarity = pairwise_distances(matrix.T, metric=metrics)
    print('>> Similarity Matrix Created')

    return similarity

def CFPredict(train_matrix, kind='user', k=40, similarity_matrics = 'cosine'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)

    if kind == 'user':
        mean_user_rating = train_matrix.mean(axis=1)
        ratings_diff = (train_matrix - mean_user_rating[:, np.newaxis])
        pred = similiarty_matrix.dot(ratings_diff) / np.array([np.abs(similiarty_matrix).sum(axis=1)]).T
        pred += mean_user_rating[:, np.newaxis]

    elif kind == 'item':
        mean_movie_rating = train_matrix.mean(axis=0)
        ratings_diff = (train_matrix - mean_movie_rating[np.newaxis, :])
        pred = ratings_diff.dot(similiarty_matrix) / np.array([np.abs(similiarty_matrix).sum(axis=1)])
        pred += mean_movie_rating[np.newaxis, :]

    return pred

def SVDPredict(train_matrix, max=20):

    train_data, test_data = train_test_split(train_matrix, test_size=0.25)
    best = 10
    best_k = 1
    for k in range(1, max):
        u, s, vt = svds(train_data, k=k)
        s_diag_matrix = np.diag(s)
        X_pred_inner = np.dot(np.dot(u, s_diag_matrix), vt)
        score = RMSE(X_pred_inner, test_data)

        if score < best:
            best = score
            best_k = k

    u, s, vt = svds(train_matrix, k=best_k)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    print('>> %s latent factors' %best_k)
    return X_pred

def MeanPredict(train_matrix, kind='user'):
    zero_matrix = np.zeros(shape=(train_matrix.shape))

    if kind == 'user':
        mean_user_rating = train_matrix.mean(axis=1)
        for i in range(len(zero_matrix)):
            zero_matrix[i] = mean_user_rating[i]
    if kind == 'item':
        mean_item_rating = train_matrix.mean(axis=0)
        for i in range(len(zero_matrix)):
            zero_matrix[:,i] = mean_item_rating[i]

    pred = zero_matrix
    return pred

def CFPredict_topk(train_matrix, kind='user', k=50,similarity_matrics='correlation'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)
    print(similarity_matrix)
    pred = np.zeros(train_matrix.shape)
    if kind == 'user':
        for i in range(train_matrix.shape[0]):
            top_k_users = [np.argsort(similiarty_matrix[:,i])[:-k-1:-1]]
            for j in range(train_matrix.shape[1]):
                pred[i, j] = similiarty_matrix[i, :][top_k_users].dot(train_matrix[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similiarty_matrix[i, :][top_k_users]))
    if kind == 'item':
        for j in range(train_matrix.shape[1]):
            top_k_items = [np.argsort(similiarty_matrix[:,j])[:-k-1:-1]]
            for i in range(train_matrix.shape[0]):
                pred[i, j] = similiarty_matrix[j, :][top_k_items].dot(train_matrix[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similiarty_matrix[j, :][top_k_items]))
    return pred

def predict_slow_simple(train_matrix, kind='user',similarity_matrics='correlation'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)
    print(similiarty_matrix)
    pred = np.zeros(train_matrix.shape)

    if kind == 'user':
        for i in range(train_matrix.shape[0]):
            for j in range(train_matrix.shape[1]):
                print('similiarty_matrix[i, :]',similiarty_matrix[i, :])
                print('train_matrix[:, j]',train_matrix[:, j])
                print('similiarty_matrix[i, :]', similiarty_matrix[i, :])
                pred[i, j] = similiarty_matrix[i, :].dot(train_matrix[:, j])\
                             /np.sum(np.abs(similiarty_matrix[i, :]))
        return pred
    elif kind == 'item':
        for i in range(train_matrix.shape[0]):
            for j in range(train_matrix.shape[1]):
                pred[i, j] = similiarty_matrix[j, :].dot(train_matrix[i, :].T)\
                             /np.sum(np.abs(similiarty_matrix[j, :]))

        return pred


def predict_topk_nobias_org(train_matrix, kind='user', max_k=2, similarity_matrics='correlation'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)

    train_data, test_data = train_test_split(train_matrix, test_size=0.25)
    best = 10
    best_k = 1
    pred = np.zeros(train_data.shape)

    for k in range(1, max_k):
        if kind == 'user':
            user_bias = train_data.mean(axis=1)
            ratings = (train_data - user_bias[:, np.newaxis]).copy()
            for i in range(train_data.shape[0]):
                top_k_users = [np.argsort(similiarty_matrix[:, i])[:-k - 1:-1]]
                for j in range(train_data.shape[1]):
                    pred[i, j] = similiarty_matrix[i, :][top_k_users].dot(train_data[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similiarty_matrix[i, :][top_k_users]))
            X_pred_inner += user_bias[:, np.newaxis]

        if kind == 'item':
            item_bias = train_data.mean(axis=0)
            ratings = (train_data - item_bias[np.newaxis, :]).copy()
            for j in xrange(train_data.shape[1]):
                top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
                for i in xrange(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(train_data[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
                    X_pred_inner += item_bias[np.newaxis, :]

        score = RMSE(X_pred_inner, test_data)

        if score < best:
            best = score
            best_k = k

    pred = np.zeros(train_matrix.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similiarty_matrix[:, i])[:-best_k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similiarty_matrix[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similiarty_matrix[i, :][top_k_users]))
        X_pred_inner += user_bias[:, np.newaxis]

    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-best_k - 1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
                X_pred_inner += item_bias[np.newaxis, :]


    return pred


def predict_topk_nobias1(train_matrix, kind='user', k=40,similarity_matrics='correlation'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)
    pred = np.zeros(train_matrix.shape)


    if kind == 'user':
        user_bias = train_matrix.mean(axis=1)
        ratings = (train_matrix - user_bias[:, np.newaxis]).copy()
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similiarty_matrix[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similiarty_matrix[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similiarty_matrix[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]

    if kind == 'item':
        item_bias = train_matrix.mean(axis=0)
        ratings = (train_matrix - item_bias[np.newaxis, :]).copy()
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similiarty_matrix[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similiarty_matrix[j, :][top_k_items]))
        pred += item_bias[np.newaxis, :]

    return pred


def predict_topk_nobias(train_matrix, kind='user', max_k=40, similarity_matrics='correlation'):
    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)
    pred = np.zeros(train_matrix.shape)

    train_data, test_data = train_test_split(train_matrix, test_size=0.25)
    best = 10
    best_k = 1
    pred = np.zeros(train_data.shape)
    X_pred_inner = np.zeros(train_data.shape)

    for k in range(1, max_k):
        if kind == 'user':
            user_bias = train_data.mean(axis=1)
            ratings = (train_data - user_bias[:, np.newaxis]).copy()
            for i in range(ratings.shape[0]):
                top_k_users = [np.argsort(similiarty_matrix[:, i])[:-k - 1:-1]]
                for j in range(ratings.shape[1]):
                    pred[i, j] = similiarty_matrix[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similiarty_matrix[i, :][top_k_users]))
            pred += user_bias[:, np.newaxis]

        if kind == 'item':
            item_bias = train_data.mean(axis=0)
            ratings = (train_data - item_bias[np.newaxis, :]).copy()
            for j in range(ratings.shape[1]):
                top_k_items = [np.argsort(similiarty_matrix[:, j])[:-k - 1:-1]]
                for i in range(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similiarty_matrix[j, :][top_k_items]))
            pred += item_bias[np.newaxis, :]

        score = RMSE(pred, test_data)

        if score < best:
            best = score
            best_k = k

    return pred


def RMSE(prediction, truth, model_name='', print_=False):

    # get the indices of test set with ratings, flatten to a list
    prediction = prediction[truth.nonzero()].flatten()
    ground_truth = truth[truth.nonzero()].flatten()
    rmse = sqrt(mean_squared_error(prediction, ground_truth))

    if print_:
        print('>> RMSE(%s): %s' %(model_name,rmse))

    return rmse

def CFPredict_topk_test(train_matrix, kind='user', k=40, similarity_matrics = 'cosine'):

    similiarty_matrix = similarity(train_matrix, kind=kind, metrics=similarity_matrics)

    def topk(matrix):
        x = matrix.argsort()[-3:][::-1]
        return x

    if kind == 'user':
        #print(train_matrix)
        mean_user_rating = train_matrix.mean(axis=1)
        #print(mean_user_rating)
        top_k_users = np.apply_along_axis(topk,axis=1,arr=similiarty_matrix)

        print('top_k_users', top_k_users)
        print('train_top_k',np.take(train_matrix,top_k_users))
        print('top similarity', np.take(similiarty_matrix, top_k_users))
        print(mean_user_rating[top_k_users])


        #ratings_diff = (train_matrix[top_k_users] - mean_user_rating[top_k_users, np.newaxis])
        print(np.take(train_matrix, top_k_users))
        print(mean_user_rating[top_k_users, np.newaxis])
        ratings_diff = (np.take(train_matrix,top_k_users) - mean_user_rating[top_k_users])
        #print('rating_diff', ratings_diff)

        print(np.take(similiarty_matrix,top_k_users))
        print(ratings_diff)

        pred = np.take(similiarty_matrix,top_k_users).dot(ratings_diff) / np.array([np.abs(np.take(similiarty_matrix,top_k_users)).sum(axis=1)]).T
        pred += mean_user_rating[:, np.newaxis]

    elif kind == 'item':
        mean_movie_rating = train_matrix.mean(axis=0)
        ratings_diff = (train_matrix - mean_movie_rating[np.newaxis, :])
        pred = ratings_diff.dot(similiarty_matrix) / np.array([np.abs(similiarty_matrix).sum(axis=1)])
        pred += mean_movie_rating[np.newaxis, :]

    return pred


if __name__ == '__main__':
    training, testing = loadinputData('../data/input/ml-100k/u.data')
    # prediction = CFPredict(training, kind='user')
    # RMSE(prediction, testing, print_=True,model_name='Collaborative Filtering')
    # prediction = SVDPredict(training)
    # RMSE(prediction,testing, print_=True,model_name='SVD')
    # pred = MeanPredict(training, kind='user')
    # RMSE(pred, testing,print_=True,model_name='Mean Model')
    pred_topk = predict_topk_nobias(training,max_k=10)
    RMSE(pred_topk, testing, print_=True, model_name='Collaborative Filtering - top K')


    # a = predict_slow_simple(training)
    # RMSE(a, testing, print_=True, model_name='Collaborative Filtering - top K')
    #prediction = CFPredict_topk_test(training, kind='user')
    #RMSE(prediction, testing, print_=True,model_name='Collaborative Filtering test')