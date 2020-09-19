import csv
from collections import defaultdict
import random
from scipy import sparse
import os

from sklearn.preprocessing import MultiLabelBinarizer


def movielens(mode='hybrid'):
    # Open and read in the ratings file
    with open(os.path.join(os.getcwd(), 'ratings.csv'), 'r') as ratings_file:
        ratings_file_reader = csv.reader(ratings_file)
        raw_ratings = list(ratings_file_reader)
        raw_ratings_header = raw_ratings.pop(0)

    # Iterate through the input to map MovieLens IDs to new internal IDs
    # The new internal IDs will be created by the defaultdict on insertion
    movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
    movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))
    for row in raw_ratings:
        row[0] = movielens_to_internal_user_ids[int(row[0])]
        row[1] = movielens_to_internal_item_ids[int(row[1])]
        row[2] = float(row[2])
    n_users = len(movielens_to_internal_user_ids)
    n_items = len(movielens_to_internal_item_ids)

    # Shuffle the ratings and split them in to train/test sets 80%/20%
    random.shuffle(raw_ratings)  # Shuffles the list in-place
    cutoff = int(.8 * len(raw_ratings))
    train_ratings = raw_ratings[:cutoff]
    test_ratings = raw_ratings[cutoff:]


    # This method converts a list of (user, item, rating, time) to a sparse matrix
    def interactions_list_to_sparse_matrix(interactions):
        users_column, items_column, ratings_column, _ = zip(*interactions)
        return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                                 shape=(n_users, n_items))


    # Create sparse matrices of interaction data
    sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings)
    sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings)

    # Construct indicator features for users and items
    user_indicator_features = sparse.identity(n_users)
    item_indicator_features = sparse.identity(n_items)

    # Create sets of train/test interactions that are only ratings >= 4.0
    sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
    sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)

    # Open and read in the movies file
    with open(os.path.join(os.getcwd(), 'movies.csv'), 'r', encoding='UTF8') as genres_file:
        genres_file_reader = csv.reader(genres_file)
        raw_movie_metadata = list(genres_file_reader)
        raw_movie_metadata_header = raw_movie_metadata.pop(0)

    movie_genres_by_internal_id = {}
    movie_titles_by_internal_id = {}
    for row in raw_movie_metadata:
        row[0] = movielens_to_internal_item_ids[int(row[0])]  # Map to IDs
        row[2] = row[2].split('|')  # Split up the genres
        movie_genres_by_internal_id[row[0]] = row[2]
        movie_titles_by_internal_id[row[0]] = row[1]

    movie_genres = [movie_genres_by_internal_id[internal_id]
                    for internal_id in range(n_items)]

    # Transform the genres into binarized labels using scikit's MultiLabelBinarizer
    movie_genre_features = MultiLabelBinarizer().fit_transform(movie_genres)
    n_genres = movie_genre_features.shape[1]

    # Coerce the movie genre features to a sparse matrix, which TensorRec expects
    movie_genre_features = sparse.coo_matrix(movie_genre_features)

    if mode == 'MF':
        pass
    elif mode == 'CB':
        item_indicator_features = movie_genre_features
    elif mode == 'hybrid':
        from scipy.sparse import hstack
        item_indicator_features = hstack([item_indicator_features, movie_genre_features])

    return sparse_train_ratings_4plus, sparse_test_ratings_4plus, user_indicator_features, item_indicator_features, n_items