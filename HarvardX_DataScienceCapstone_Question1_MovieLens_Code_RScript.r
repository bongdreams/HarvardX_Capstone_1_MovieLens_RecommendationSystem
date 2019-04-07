# Note: this process could take a couple of minutes

# Installing required libraries
install.packages('tidyverse', dependencies = TRUE)
install.packages('caret', dependencies = TRUE)
install.packages('Rcpp', dependencies = TRUE)
install.packages('recosystem', dependencies = TRUE)
install.packages('stringr', dependencies = TRUE)
install.packages('dplyr', dependencies = TRUE)
install.packages('SimDesign', dependencies = TRUE)
install.packages('ggplot2', dependencies = TRUE)

# Loading the required libraries
library('stringr')
library('dplyr')
library('caret')
library('recosystem')
library("SimDesign")


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Downloading the data locally
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Reading the temporarily downloaded data into ratings
ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

#Describing the dataset
str(ratings)
head(ratings)
table(ratings$rating)
summary(ratings)


# Reading the data into movies
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3);

# Defining column names - movieId, title and genres
colnames(movies) <- c("movieId", "title", "genres");

# Populating the data in the respective column names
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres));

# Doing a left join between ratings and movies on the field "movieId"
movielens <- left_join(ratings, movies, by = "movieId")

# No Missing values in both the datasets
sum(is.na(ratings$rating))

sum(is.na(movielens$rating))

# No missing data
colSums(sapply(movielens, is.na))

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.10, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Trying to visualize the ratings data
ggplot(ratings, aes(x=rating)) + 
  geom_histogram(binwidth=0.1,color="black", fill="white",alpha=0.5)+
  labs(title="RATING HISTOGRAM",x="RATING", y = "COUNT")


#Creating the datasets in proper format 
#for running the recommendation using recosystem library
train_data <- data_memory(user_index = edx$userId, item_index = edx$movieId, 
                          rating = edx$rating, index1 = T);
test_data <- data_memory(user_index = validation$userId, item_index = validation$movieId, 
                         rating = validation$rating, index1 = T);

#Creating the recommender using recosystem library
recommender <- Reco()

#Training the recommender on train_data which is created from edx dataset
# opts argument is a list to supply the number of parameters
# dim is an integer which represents the number of latent factors
# costp_l2 is a numeric L2 regularization parameter for user factors
# costq_l2 is a numeric L2 regularization parameter for item factors
# lrate is numeric learning rate
# niter is an interger representing number of iterations
# nthread is an integer representing number of threads for parallel computing 
# nbin is an integer representing number of bins, must be greater than nthread - default is used in this case
# verbose is a logical parameter to show detailed information
# For more info on these tuning parameters try: ?recosystem::train

recommender$train(train_data, opts = c(dim = 30, costp_l2 = 0.1, costq_l2 = 0.1, 
                                       lrate = 0.1, niter = 100, nthread = 6, verbose = F)) 

#Predicting the ratings on test_data which is created from validation dataset
prediction <- recommender$predict(test_data, out_memory())

#Creating the dataframe for predicted ratings
pred <- data.frame("predicted ratings" = prediction)

#creating the dataframe with actual ratings from the validation set
test_data_df <- data.frame("actual ratings"=validation[,c(3)])

#Calculating the RMSE
RMSE <- RMSE(pred,test_data_df)

#Outputting the RMSE
#Please note the header is basically the name of the column of the pred dataset
print(RMSE)

#creating the dataframe which I will use to generate my submission file
pred_df <- cbind(validation,pred)

#writing the submission.csv file
#the file contains the userId,movieId and corresponding predicted ratings
write.csv(pred_df %>% select(userId, movieId, "predicted.ratings"),
          "submission.csv", na = "", row.names=FALSE)