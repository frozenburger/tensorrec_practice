import tensorrec
from code_original.preprocessing import movielens as given_data
from data.fetch import movielens as my_data

#sparse_train_ratings_4plus, sparse_test_ratings_4plus, user_indicator_features, item_indicator_features, n_items = given_data(mode='hybrid')
sparse_train_ratings_4plus, sparse_test_ratings_4plus, user_indicator_features, item_indicator_features, n_items = my_data(to_sparse=True, mode='MF')

# This method consumes item ranks for each user and prints out recall@10 train/test metrics
def check_results(ranks):
    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_train_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_test_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    print("Recall at 10: Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
                                                            test_recall_at_10))
# Let's try a new loss function: WMRB
ranking_cf_model = tensorrec.TensorRec(n_components=5,
                                       loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_model.fit(interactions=sparse_train_ratings_4plus,
                     user_features=user_indicator_features,
                     item_features=item_indicator_features,
                     n_sampled_items=int(n_items * .01))

# Check the results of the WMRB MF CF model
predicted_ranks = ranking_cf_model.predict_rank(user_features=user_indicator_features,
                                                item_features=item_indicator_features)
check_results(predicted_ranks)