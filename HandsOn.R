# Machine Learning R HandsOn --------------------------------------------------

# Libraries used
library(dplyr) # Data manipulation
library(lubridate) # Date and time manipulation
library(forcats) # Factorizing
library(caret) # Data manipulation
library(rsample) # Data sampling
library(tidyr) # Data manipulation
library(rpart) # Tree-based models
library(rpart.plot) # Plotting tree-based models
library(ipred) # Bagging
library(doParallel) # Parallel processing
library(ranger) # Random forests
library(gbm) # Generalized Boosted Models
library(e1071) # Support Vector Machines

# ──────────────────────────────────────────────────────────────────────────────
# --- FEATURE ENGINEERING ----
# ──────────────────────────────────────────────────────────────────────────────

# Loading the data -------------------------------------------------------------

dataset <-
  readxl::read_excel("/Users/pawelgach/Documents/Aarhus Uni/Machine Learning II/HandsOn/Data.xls")

# Missing Values ---------------------------------------------------------------

skimr::skim(dataset)
View(dataset)
# It's immediately obvious that reviewId is an Id column that can be deleted
dataset <- subset(dataset, select = -reviewId)

# Moreover, isCurrentJob, JobEndingYear, ratingCeo, ratingBusinessOutlook and
# ratingRecommendToFriend all have a big amount of missing values (~40%)

# However, since all values in isCurrentJob equal one, its pretty likely that
# the missing values are actually meant to be zeroes.
dataset$isCurrentJob[is.na(dataset$isCurrentJob)] <- 0

# Moreover, the missing values in job ending year probably signify employees
# that haven't quit yet

current_jobs <- dataset %>%
  filter(isCurrentJob == 1)

sum(current_jobs$jobEndingYear)
# Just NAs as expected
rm(current_jobs)

# Since jobEndingYear contributes pretty much the same amount of information as
# isCurrentJob, and is full of missing values, I decide to delete it along with
# other high NA columns
dataset <-
  dataset[, !(
    names(dataset) %in% c(
      "ratingBusinessOutlook",
      "ratingCeo",
      "jobEndingYear",
      "ratingRecommendToFriend"
    )
  )]

# For the NAs in jobTitle.text and location.name, I decide to replace them with
# "Other" as they are all categorical variables with a lot of classes that I
# plan to lump anyway
dataset <- dataset %>%
  mutate(jobTitle.text = ifelse(is.na(jobTitle.text), "Other", jobTitle.text)) %>%
  mutate(location.name = ifelse(is.na(location.name), "Other", location.name))

# Finally, I inspect the 66 rows in which employmentStatus is NA, since
# it only has 2 states and cannot be easily lumped
rows_with_na <- dataset %>%
  filter(is.na(employmentStatus))

View(rows_with_na)
# There doesn't appear to be anything extraordinary about them, so I decide to
# delete those rows
dataset <- dataset[!is.na(dataset$employmentStatus),]

# Transforming the Date column -------------------------------------------------

# Before continuing, I also transform the reviewDateTime variable into a Date
# column and then transform it into a numeric variable with the amounts
# of seconds that passed since the earliest date

dataset$reviewDateTime <- ymd_hms(dataset$reviewDateTime)

# Find the earliest date-time in the reviewDateTime column
reference_datetime <- min(dataset$reviewDateTime)

# Calculate the number of seconds since the reference date-time and create a
# new column with it
dataset$reviewDateNumeric <-
  as.numeric(difftime(dataset$reviewDateTime, reference_datetime, units = "secs"))

# Delete the old column
dataset <- subset(dataset, select = -reviewDateTime)

# Handling the dependent variable ----------------------------------------------

# Since it is not clear if the exam will concern classification or regression,
# I decide to create two versions of the dataset and work on them both.

# The first one will take the dependent variable as categorical, the other as
# continous

dataset_c <- dataset
dataset_r <- dataset

dataset_c$ratingOverall <- as.character(dataset_c$ratingOverall)
# (Will be factorized in the next step)


# Preparing the data for modeling ----------------------------------------------

# Now, for both datasets, I:
# - factorise, lump and dummy encode all character variables
# - scale all numeric variables

# Identify numeric columns
numeric_columns <- sapply(dataset_c, is.numeric)

# Scale numeric columns
data_scaled <- dataset_c
data_scaled[numeric_columns] <- scale(dataset_c[numeric_columns])

# Factorize and lump character variables
data_scaled <- data_scaled %>%
  mutate(across(where(is.character), ~ fct_lump(.x, n = 5)))

# Exclude ratingOverall from the dummy encoding process
non_dummy_vars <- c("ratingOverall")
dummy_vars <- setdiff(names(data_scaled), non_dummy_vars)

# Create dummy variables for the selected columns
dummy_formula <- as.formula(paste("~", paste(dummy_vars, collapse = " + ")))
dummy_model <- dummyVars(dummy_formula, data = data_scaled)
data_dummy <- predict(dummy_model, newdata = data_scaled)
data_dummy <- as.data.frame(data_dummy)

# Add "ratingOverall" back to the dataset
data_dummy <- cbind(data_dummy, ratingOverall = data_scaled$ratingOverall)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)
split <- initial_split(data_dummy, prop = 0.8)
train_c <- training(split)
test_c <- testing(split)

# use make.names to avoid future naming conflicts
levels(train_c$ratingOverall) <- make.names(levels(train_c$ratingOverall))
levels(test_c$ratingOverall) <- make.names(levels(test_c$ratingOverall))

# I do the same for the other dataset

numeric_columns <- sapply(dataset_r, is.numeric)

data_scaled <- dataset_r
data_scaled[numeric_columns] <- scale(dataset_r[numeric_columns])

data_scaled <- data_scaled %>%
  mutate(across(where(is.character), ~ fct_lump(.x, n = 5)))

dummy_vars <- dummyVars("~ .", data = data_scaled)
data_dummy <- predict(dummy_vars, newdata = data_scaled)
data_dummy <- as.data.frame(data_dummy)

set.seed(123)
split <- initial_split(data_dummy, prop = 0.8)
train_r <- training(split)
test_r <- testing(split)

# Naming conflicts -------------------------------------------------------------

# use make.names to avoid future naming conflicts in factor names
levels(train_c$ratingOverall) <- make.names(levels(train_c$ratingOverall))
levels(test_c$ratingOverall) <- make.names(levels(test_c$ratingOverall))

# Some of the variables contain special characters, so I make sure to transform
# them as well
names(train_c) <- make.names(names(train_c))
names(test_c) <- make.names(names(test_c))
names(train_r) <- make.names(names(train_r))
names(test_r) <- make.names(names(test_r))


# Cleanup ----------------------------------------------------------------------

rm(
  data_dummy,
  data_scaled,
  dummy_vars,
  rows_with_na,
  split,
  numeric_columns,
  reference_datetime,
  dummy_formula,
  dummy_model,
  non_dummy_vars
)

# Set up k-fold CV -------------------------------------------------------------

cv <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE
)

cv_r <- trainControl(
  method = "cv",
  number = 10
)


# ──────────────────────────────────────────────────────────────────────────────
# --- TREE-BASED MODELS ----
# ──────────────────────────────────────────────────────────────────────────────


# Classification Trees - CP ----------------------------------------------------

# Train the model (focus on CP)
tree_c_cp <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "rpart",
  metric = "Accuracy",
  trControl = cv,
  tuneLength = 20
)

# Visualise
rpart.plot(tree_c_cp$finalModel)

# Check metrics
tree_c_cp
plot(tree_c_cp)
# Accuracy only at 58%

# Check variable importance
varImp(tree_c_cp)
plot(varImp(tree_c_cp))
# Only 5 above 0.50

# Check test accuracy
predictions <- predict(tree_c_cp, newdata = test_c)

# Confusion Matrix
confusionMatrix(predictions, test_c$ratingOverall)
# Although the model makes mistakes, they are mostly close to their class
# E.g. - mismatches of 1 are usually 2, mismatches of 2 are usually 1 or 3 etc.

# Accuracy
(tree_c_depth_accuracy <- mean(predictions == test_c$ratingOverall))
# Test acc also at 58%

# Classification Trees - Depth -------------------------------------------------

# Train the model
tree_c_depth <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "rpart2", # change here
  metric = "Accuracy",
  trControl = cv,
  tuneLength = 20
)

# Visualise
rpart.plot(tree_c_depth$finalModel)

# Check metrics
tree_c_depth
plot(tree_c_depth)
# Accuracy slighlty worse at 54%

# Check variable importance
varImp(tree_c_depth)
plot(varImp(tree_c_depth))
# Only 4 above 0.50

# Check test accuracy
predictions <- predict(tree_c_depth, newdata = test_c)

# Confusion Matrix
confusionMatrix(predictions, test_c$ratingOverall)
# The model almost always predicts the higher classes 3, 4 and 5
# no predictions for 2

# Accuracy
(tree_c_cp_accuracy <- mean(predictions == test_c$ratingOverall))
# Test acc at 52%

# Regression Trees - CP --------------------------------------------------------

# Train the model (focus on CP)
tree_r_cp <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "rpart",
  metric = "RMSE",
  trControl = cv_r,
  tuneLength = 20
)

# Visualise
rpart.plot(tree_r_cp$finalModel)

# Check metrics
tree_r_cp
plot(tree_r_cp)
# RMSE at 0.6675, since data is scaled, this is a bad score

# Check variable importance
varImp(tree_r_cp)
plot(varImp(tree_r_cp))
# Only 5 above 0.50, similiar as before

# Predict on the test set
predictions <- predict(tree_r_cp, newdata = test_r)

# Calculate RMSE
(test_r_cp_rmse <- RMSE(test_r$ratingOverall, predictions))
# test RMSE at 0.6791, slighlty worse

# Regression Trees - Depth -----------------------------------------------------

# Train the model (focus on CP)
tree_r_depth <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "rpart2",
  metric = "RMSE",
  trControl = cv_r,
  tuneLength = 20
)

# Visualise
rpart.plot(tree_r_depth$finalModel)

# Check metrics
tree_r_depth
plot(tree_r_depth)
# RMSE at 0.7054, slighlty worse than CP

# Check variable importance
varImp(tree_r_depth)
plot(varImp(tree_r_depth))
# Only 5 above 0.50, similiar as before

# Predict on the test set
predictions <- predict(tree_r_depth, newdata = test_r)

# Calculate RMSE
(test_r_depth_rmse <- RMSE(test_r$ratingOverall, predictions))
# test RMSE at 0.7072, slighlty worse than train RMSE

# Bagging - Classification -----------------------------------------------------

# Training - takes a while, sped up with parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train the model
bag_c <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "treebag",
  trControl = cv, 
  nbagg = 100,
  control = rpart.control(minsplit = 100, cp = 0),
  verbose = TRUE
)

# Stop the cluster
stopCluster(cl)
# Reset to sequential processing
registerDoSEQ()

bag_c
# Accuracy 0.6038

# Test Accuracy
predictions <- predict(bag_c, test_c)
(bag_c_accuracy <- mean(predictions == test_c$ratingOverall))
# 0.5993

# Bagging - Regression ---------------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train the model
bag_r <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "treebag",
  trControl = cv_r, 
  nbagg = 100,
  control = rpart.control(minsplit = 100, cp = 0),
  verbose = TRUE
)

# Stop the cluster
stopCluster(cl)
# Reset to sequential processing
registerDoSEQ()

bag_r
# RMSE 0.6096

# Test RMSE
predictions <- predict(bag_r, newdata = test_r)
(bag_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.6210

# Random Forest - Classification -----------------------------------------------

# Define the grid of hyperparameters
tune_grid <- expand.grid(
  mtry = c(8, 10, 12, 14, 16), # Number of variables randomly sampled
  splitrule = c("gini", "extratrees"), # Splitting rule
  min.node.size = c(5, 10, 15) # Minimum size of terminal nodes
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train the model with tuning
set.seed(123) # For reproducibility
rf_c <- train(
  ratingOverall ~ ., 
  data = train_c, 
  method = "ranger", 
  trControl = cv, 
  tuneGrid = tune_grid
)

stopCluster(cl)
registerDoSEQ()

# Print the results
rf_c
# Accuracy 0.6945

# Test accuracy
predictions <- predict(rf_c, newdata = test_c)
(rf_c_accuracy <- confusionMatrix(predictions, test_c$ratingOverall))
# 0.701

# Random Forest - Regression ---------------------------------------------------

tune_grid <- expand.grid(
  mtry = c(8, 10, 12, 14, 16), # Number of variables randomly sampled
  splitrule = c("variance", "extratrees"), # Splitting rule
  min.node.size = c(5, 10, 15) # Minimum size of terminal nodes
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Train the model with tuning
set.seed(123) # For reproducibility
rf_r <- train(
  ratingOverall ~ ., 
  data = train_r, 
  method = "ranger", 
  trControl = cv_r, 
  tuneGrid = tune_grid
)

stopCluster(cl)
registerDoSEQ()

# Results
rf_r
# RMSE 0.5624

# Test RMSE
predictions <- predict(rf_r, newdata = test_r)
(rf_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.5623

# Gradient Boosting - Classification -------------------------------------------

tune_grid <- expand.grid(
  n.trees = c(100, 200, 300), # Number of trees to grow
  interaction.depth = c(1, 2, 4), # Number of splits in each tree
  shrinkage = c(0.001, 0.01, 0.1), # lambda (learning rate)
  n.minobsinnode = c(50, 100, 200) # Larger minimum terminal node sizes
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

gb_c <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "gbm",
  trControl = cv,
  tuneGrid = tune_grid
)

stopCluster(cl)
registerDoSEQ()

gb_c
# Accuracy 0.6250

# Test accuracy
predictions <- predict(gb_c, newdata = test_c)
(gb_c_accuracy <- confusionMatrix(predictions, test_c$ratingOverall))
# 0.6300 

# Gradient Boosting - Regression -----------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

gb_r <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "gbm",
  trControl = cv_r,
  tuneGrid = tune_grid
)

stopCluster(cl)
registerDoSEQ()

gb_r
# RMSE 0.6010

# Test RMSE
predictions <- predict(gb_r, newdata = test_r)
(gb_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.6136


# ──────────────────────────────────────────────────────────────────────────────
# ---- SUPPORT VECTOR MACHINES ----
# ──────────────────────────────────────────────────────────────────────────────


# Set up cheaper k-fold CV -----------------------------------------------------

cv_svm <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE
)

# Polynomial Kernel - Classification -------------------------------------------

# Define the tuning grid for polynomial kernel SVM
tune_grid_poly <- expand.grid(
  degree = c(2, 3, 4),
  scale = c(1),
  C = c(0.01, 0.1, 1)
)

# Take a sample to reduce computational cost of tuning
set.seed(123)
train_c_sample <- train_c[sample(1:nrow(train_c), 0.1 * nrow(train_c)), ]

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_poly_c_tune <- train(
  ratingOverall ~ .,
  data = train_c_sample,
  method = "svmPoly",
  trControl = cv_svm,
  tuneGrid = tune_grid_poly,
  metric = "Accuracy"
)

stopCluster(cl)
registerDoSEQ()

# Check and save best parameters
svm_poly_c_tune$bestTune

best_grid_poly <- expand.grid(
  degree = svm_poly_c_tune$bestTune$degree,
  scale = svm_poly_c_tune$bestTune$scale,
  C = svm_poly_c_tune$bestTune$C
)

# Train model using best parameters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_poly_c <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "svmPoly",
  trControl = cv_svm,
  tuneGrid = best_grid_poly,
  metric = "Accuracy"
)
stopCluster(cl)
registerDoSEQ()

svm_poly_c
# Accuracy 0.5433234

# Test Accuracy
predictions <- predict(svm_poly_c, newdata = test_c)
(svm_poly_c_accuracy <- confusionMatrix(predictions, test_c$ratingOverall))
# 0.5484

# Polynomial Kernel - Regression -----------------------------------------------

# Take a sample to reduce computational cost of tuning
set.seed(123)
train_r_sample <- train_r[sample(1:nrow(train_r), 0.1 * nrow(train_r)), ]

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_poly_r_tune <- train(
  ratingOverall ~ .,
  data = train_r_sample,
  method = "svmPoly",
  trControl = cv_svm,
  tuneGrid = tune_grid_poly,
  metric = "RMSE"
)

stopCluster(cl)
registerDoSEQ()

# Check and fetch best parameters
svm_poly_r_tune$bestTune

best_grid_poly <- expand.grid(
  degree = svm_poly_r_tune$bestTune$degree,
  scale = svm_poly_r_tune$bestTune$scale,
  C = svm_poly_r_tune$bestTune$C
)

# Train model using best parameters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_poly_r <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "svmPoly",
  trControl = cv_svm,
  tuneGrid = best_grid_poly,
  metric = "RMSE"
)

stopCluster(cl)
registerDoSEQ()

svm_poly_r
# RMSE 0.7248

# Test RMSE
predictions <- predict(svm_poly_r, newdata = test_r)
(svm_poly_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.7260

# Linear Kernel - Classification -----------------------------------------------

tune_grid_linear <- expand.grid(
  C = c(0.01, 0.1, 1)
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_line_c_tune <- train(
  ratingOverall ~ .,
  data = train_c_sample,
  method = "svmLinear",
  trControl = cv_svm,
  tuneGrid = tune_grid_linear,
  metric = "Accuracy"
)

stopCluster(cl)
registerDoSEQ()

svm_line_c_tune$bestTune

best_grid_line <- expand.grid(
  C = svm_line_c_tune$bestTune$C
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_line_c <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "svmLinear",
  trControl = cv_svm,
  tuneGrid = best_grid_line,
  metric = "Accuracy"
)

stopCluster(cl)
registerDoSEQ()

svm_line_c
# Accuracy 0.4346

# Test Accuracy
predictions <- predict(svm_line_c, newdata = test_c)
(svm_line_c_accuracy <- confusionMatrix(predictions, test_c$ratingOverall))
# 0.428

# Linear Kernel - Regression ---------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_line_r_tune <- train(
  ratingOverall ~ .,
  data = train_r_sample,
  method = "svmLinear",
  trControl = cv_svm,
  tuneGrid = tune_grid_linear,
  metric = "RMSE"
)

stopCluster(cl)
registerDoSEQ()

# Check and fetch best parameters
svm_line_r_tune$bestTune

best_grid_line <- expand.grid(
  C = svm_line_r_tune$bestTune$C
)

# Train model using best parameters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_line_r <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "svmLinear",
  trControl = cv_svm,
  tuneGrid = best_grid_line,
  metric = "RMSE"
)

stopCluster(cl)
registerDoSEQ()

svm_line_r
# RMSE 0.8679407

# Test RMSE
predictions <- predict(svm_line_r, newdata = test_r)
(svm_line_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.8756717

# Radial Kernel - Classification -----------------------------------------------

tune_grid_radial <- expand.grid(
  sigma = c(0.01, 0.05, 0.1, 0.5, 1),
  C = c(0.01, 0.05, 0.1, 0.5, 1, 5, 10)
)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_rad_c_tune <- train(
  ratingOverall ~ .,
  data = train_c_sample,
  method = "svmRadial",
  trControl = cv_svm,
  tuneGrid = tune_grid_radial,
  metric = "Accuracy"
)

stopCluster(cl)
registerDoSEQ()

# Check and save best parameters
svm_rad_c_tune$bestTune

best_grid_rad <- expand.grid(
  sigma = svm_rad_c_tune$bestTune$sigma,
  C = svm_rad_c_tune$bestTune$C
)

# Train model using best parameters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(1234)
svm_rad_c <- train(
  ratingOverall ~ .,
  data = train_c,
  method = "svmRadial",
  trControl = cv_svm,
  tuneGrid = best_grid_rad,
  metric = "Accuracy"
)
stopCluster(cl)
registerDoSEQ()

svm_rad_c
# Accuracy 0.5616596

# Test Accuracy
predictions <- predict(svm_rad_c, newdata = test_c)
(svm_rad_c_accuracy <- confusionMatrix(predictions, test_c$ratingOverall))
# 0.5705

# Radial Kernel - Regression ---------------------------------------------------

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_rad_r_tune <- train(
  ratingOverall ~ .,
  data = train_r_sample,
  method = "svmRadial",
  trControl = cv_svm,
  tuneGrid = tune_grid_radial,
  metric = "RMSE"
)

stopCluster(cl)
registerDoSEQ()

# Check and save best parameters
svm_rad_r_tune$bestTune

best_grid_rad <- expand.grid(
  sigma = svm_rad_r_tune$bestTune$sigma,
  C = svm_rad_r_tune$bestTune$C
)

# Train model using best parameters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

set.seed(123)
svm_rad_r <- train(
  ratingOverall ~ .,
  data = train_r,
  method = "svmRadial",
  trControl = cv_svm,
  tuneGrid = best_grid_rad,
  metric = "RMSE"
)
stopCluster(cl)
registerDoSEQ()

svm_rad_r
# RMSE 0.673304

# Test RMSE
predictions <- predict(svm_rad_r, newdata = test_r)
(svm_rad_r_rmse <- RMSE(test_r$ratingOverall, predictions))
# 0.6768312
