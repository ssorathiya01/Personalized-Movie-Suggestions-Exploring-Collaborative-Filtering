# Personalized Movie Suggestions: Exploring Collaborative Filtering
Introduction:
In today's era of personalized experiences, recommendation systems play an important role in guiding users to search for content according to their preferences. The collaborative filtering processes in these systems stand out for their ability to leverage the collective intelligence of users to deliver realistic and personalized recommendations. In this blog in this article we will dive into the world of collaborative research programs and explore how to make movies using the MovieLens dataset .
Loading the Dataset:
To start our search, we load the important features of the dataset. We initialize the variables X, W, b, Y, and R to store movie content, user parameters, ratings, and indicators. These variables provide important insights into the structure of the dataset and serve as input to our collaborative filtering model.Once the dataset is loaded, we analyze its statistics in detail to gain a deeper understanding. We calculate metrics like average ratings for a particular movie, user preferences, and shed light on movie popularity.
#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)
#From the matrix, we can compute statistics like average rating.
tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )
Understanding Collaborative Filtering:
Collaborative filtering is a method used to suggest items (like movies) to users by analyzing their previous interactions and the interactions of users with similar tastes. Essentially, collaborative filtering operates on the premise that users who have rated or engaged with similar items previously are likely to share similar preferences. Now, let's delve into the code to grasp how collaborative filtering functions in real-world scenarios.
def cofi_cost_func(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j,:]
        B = b[0,j]
        for i in range(nm):
            x = X[i,:]
            y = Y[i,j]
            r = R[i,j]
            J += r * np.square((np.dot(w,x) + B - y ))
    J += (lambda_) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    J = J/2
     
    return J
Implementing the Collaborative Filtering Cost Function:
To create an effective movie recommendation system, we need to develop a complex cost function that takes into account the difference between predicted and actual The above code snippet implements a collaborative filtering cost function, which calculates the cost of the model's predictions.
#Reduce the data set size so that this runs faster
num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

#Evaluate cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")
#Evaluate cost function with regularization 
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")
Training the Collaborative Filtering Model:
When we train a collaborative filtering model, we adjust the parameters to reduce a cost function. This involves using gradient descent and a personalized training loop. Through continuous updates to these parameters, the model gets better at predicting user movie ratings.
iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow's GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
Making Movie Recommendations:
With our trained collaborative filtering model in hand, we can now create personalized movie recommendations for users. We will determine how users are predicted to rate the movies and rank them to recommend top choices. By incorporating the user's preferences and past interactions, our recommendation system can guide users to discover new relevant movies according to their interests.
#Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

#sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')
filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)
Conclusion:
Collaborative filtering algorithms provide a great way to suggest movies, giving users personalized recommendations based on their likes and previous interactions. By understanding how collaborative filtering works and using the MovieLens dataset, we can create strong recommendation systems that help users find new and interesting movies that suit their preferences.
In this blog post, we explored the basics of collaborative filtering, implemented a cost function, trained a model, and created an individual movie proposal. Armed with this knowledge, readers can begin their journey to create and make a recommendation system that delivers a unique user experience.
