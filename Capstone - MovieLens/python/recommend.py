import pandas as pd
import sys

def loadinputData(input_file):
    """
    
    :param input_file: CSV with input-rating.csv with columns: ['userId', 'movieId', 'rating', 'timestamp']
    :return: 
    
    """

    input_data = pd.read_csv(input_file)
    input_user = input_data['userId'].unique() # get unique input user
    input_movie_and_ratings = [tuple(x) for x in input_data[['movieId','rating']].values] # get all movie and ratings the user(s) have done

    print('>>> Moving-Rating Tuples: %s' % str(input_movie_and_ratings))
    print('>>> Input User(s): %s' % str(input_user))

    print('Accessing rating data...')
    ratings = pd.read_csv('../data/input/ml-ratings-100k-sample.csv')

    # extract all the movie and ratings that is the same as input users's movie rating
    # but not the same user

    count = 0
    total = len(ratings.index)
    columns=['userId','movieId','rating']
    new_rows = []

    for index, row in ratings.iterrows():
        count += 1 # Progress bar
        p = count / total * 100.0
        sys.stdout.write("\r>>> Ratings processed: %d (%i%%) " % (count, p))
        sys.stdout.flush()

        pair = (row[1],row[2])
        if pair in input_movie_and_ratings and row[2] not in input_user:
            new_rows.append(row)

    print('\r >>> Found %s matching movie rating' %len(new_rows))
    matching_movie_df = pd.DataFrame(new_rows,columns=columns).reset_index()
    matching_movie_df.to_csv('../data/output/v1-input-extract.csv')

def sortUserMovieAgreement():

    data = pd.read_csv('../data/output/v1-input-extract.csv')
    rating_count = 1
    rating_count_list = []
    rating_count_list_store =[]
    rating_count_list_uniques = []

    for i in range(0, len(data.index) - 1):

        row1 = data.ix[i]

        try:
            row2 = data.ix[i+1]

        except IndexError:
            print('>> Max Agreement: %s' %str(max(rating_count_list)))
            rating_count_list_uniques.sort()
            print('>> Freq Present: %s' %str(rating_count_list_uniques))

        user_id1 = row1['userId']
        user_id2 = row2['userId']
        if user_id1 == user_id2:
            rating_count += 1
        if user_id1 != user_id2:
            rating_count_list.append(rating_count)
            rating_count_list_store.append([row1['userId'], rating_count])
            if rating_count not in rating_count_list_uniques:
                rating_count_list_uniques.append(rating_count)
            rating_count = 1
    # get the number of exact rating we have from each user
    extract_counts_df = pd.DataFrame(rating_count_list_store, columns=['userId','counts'],index=None)
    extract_counts_df.to_csv('../data/output/a2-ratings-extract-counts-by-user.csv')

    print('>>> Output Rows: %s' %len(extract_counts_df.index))

def getHighestAgreementUserID(lower_limit=4, upper_limit=9):

    data = pd.read_csv('../data/output/a2-ratings-extract-counts-by-user.csv')
    agreeing_users = data[data['counts'].between(lower_limit, upper_limit, inclusive=True)]
    print('>>> %s agreeing users' %len(agreeing_users.index))
    agreeing_users.to_csv('../data/output/a3-extract-agreeing-users.csv')

def getMoviesFromHighAgreementUsers(input_file):

    agreeing_users = pd.read_csv('../data/output/a3-extract-agreeing-users.csv')
    ratings = pd.read_csv('../data/input/ml-ratings-100k-sample.csv')
    input_data = pd.read_csv(input_file)

    agreeing_users_recommend_movies = []
    input_movie_and_ratings = [tuple(x) for x in input_data[['movieId','rating']].values]
    agreeing_users = agreeing_users['userId']

    columns=['userId','movieId','rating']
    count = 0
    total = len(ratings.index)

    for index, row in ratings.iterrows():
        count += 1
        p = count / total * 100.0
        sys.stdout.write("\r>>> Ratings processsed: %d (%i%%) " % (count, p))
        sys.stdout.flush()

        pair = (row[1], row[2])
        if pair not in input_movie_and_ratings and row[2] in agreeing_users:
            agreeing_users_recommend_movies.append(row)


    agreeing_users_recommend_movies_df = pd.DataFrame(agreeing_users_recommend_movies, columns=columns,index=None)
    print('\n >> %s Movies recommended by agreeing users' %len(agreeing_users_recommend_movies_df.index))
    agreeing_users_recommend_movies_df.to_csv("../data/output/a4-ratings-extract-recommended.csv")

def sortMovies():
    data = pd.read_csv("../data/output/a4-ratings-extract-recommended.csv")
    data = data.sort_values(by='userId')
    grouped_df = data.groupby('movieId').count()
    df = pd.DataFrame(grouped_df).drop(['userId','rating'], axis=1).reset_index()
    df.to_csv('../data/output/a5-ratings-extract-recommended-sorted-counts.csv')

def extractTopkMovie(min_rating_count=3):
    data = pd.read_csv('../data/output/a5-ratings-extract-recommended-sorted-counts.csv')
    extracted_ratings = pd.read_csv("../data/output/a4-ratings-extract-recommended.csv")
    print('>>> %s of Agreeing Movies' % len(data.index))

    count = 0
    temp_movie_ratings_sum = 0
    temp_movie_counts = 0
    temp_movie_avg = 0
    movie_recommend_df = pd.DataFrame()

    for i in range(0, len(extracted_ratings.index) + 1 ):
        count += 1
        sys.stdout.write("\r>>> Ratings processed: %i" % count)
        sys.stdout.flush()

        row1 = data.ix[i]

        try:
            row2 = data.ix[i + 1]
        except IndexError:
            print('>> Max Agreement: %s' % str(max(rating_count_list)))
            rating_count_list_uniques.sort()
            print('>> Freq Present: %s' % str(rating_count_list_uniques))

        movie1 = row1['movieId']
        movie2 = row2['movieId']
        #temp_rating_delta = row1['rating']
        print(data['movieId'])
        #print(movie1)
        #print(data.str.contains(movie1))

        if  data['movieId'].str.contains(movie1) and movie2 == movie1:
            #temp_movie_ratings_sum += temp_rating_delta
            temp_movie_counts += 1

        if data['movieId'].str.contains(movie1) and movie2 != movie1:
            #temp_movie_ratings_sum += temp_rating_delta
            temp_movie_counts += 1
            #temp_movie_avg = temp_movie_ratings_sum / temp_movie_counts

            movie_recommend_df.append([movie1,temp_movie_counts])
            #temp_movie_ratings_sum = 0
            temp_movie_counts = 0

            print(movie_recommend_df)







if __name__ == '__main__':
    #loadinputData("../data/input/v1-input-ratings.csv")
    #sortUserMovieAgreement()
    #getHighestAgreementUserID()
    #getMoviesFromHighAgreementUsers("../data/input/v1-input-ratings.csv")
    #sortMovies()
    extractTopkMovie()