# Machine Learning II Exam Template --------------------------------------------

# Used libraries and datasets
library(rgl) # for 3D Plots - Requires external XQuartz software

library(ISLR) # Datasets and functions from the ISLR book
library(forecast) # for forecasting with 95% CI bands
library(boot) # bootstrap methods
library(splines) # for regression splines
library(gam) # for Generalized Additive Models
library(akima) #interpolation methods (e.g. spline fitting)
library(tree) # regression and classification tree-based models
library(MASS) # Functions and datasets from the MASS book
library(randomForest) # for tree-based random forest
library(dplyr) # for data manipulation
library(ggplot2) # plotting
library(rsample) # resampling
library(gbm) # for Generalized Boosted Regression Models (GBMs)
library(xgboost) # for eXtreme Gradient Boosting
library(caret) # for everything
library(e1071) # for Support Vector Machines and related concepts
library(ROCR) # For plotting ROC curves
library(vip) # For variable importance plots
library(pdp) # For partial dependence plots
library(recipes) # to prepare the data quickly
library(h2o) # for fitting stacked models REQUIRES JAVA INSTALLATION

# Datasets
attach(Wage)
attach(Carseats)
data(iris)
AmesHousing::make_ames()

# ──────────────────────────────────────────────────────────────────────────────
# POLYNOMIAL FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

# Choosing Degree of Polynomial ------------------------------------------------

fit.1 = lm(wage ~ age, data = Wage)
fit.2 = lm(wage ~ poly(age, 2), data = Wage)
fit.3 = lm(wage ~ poly(age, 3), data = Wage)
fit.4 = lm(wage ~ poly(age, 4), data = Wage)
fit.5 = lm(wage ~ poly(age, 5), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5)
# With ANOVA
# Choose the highest level that is significant and discard higher orders

cross_validate_poly <- function(data,
                                dependent_var,
                                independent_var,
                                max_degree = 5,
                                folds = 10) {
  # Prepare the environment
  set.seed(19) # For reproducibility
  cv.error = rep(0, max_degree) # Initialize the error vector
  
  # Perform cross-validation for models of degrees 1 through max_degree
  for (i in 1:max_degree) {
    # Construct the formula
    formula_i = as.formula(paste(dependent_var, "~ poly(", independent_var, ",", i, ")"))
    
    # Fit the model using glm
    fit.i = glm(formula_i, data = data)
    
    # Calculate cross-validation error
    cv.error[i] = boot::cv.glm(data, fit.i, K = folds)$delta[1]
  }
  
  # Return the cross-validation errors
  return(cv.error)
}

# Example of using the function
cv_error_results = cross_validate_poly(
  data = Wage,
  dependent_var = "wage",
  independent_var = "age",
  max_degree = 5
)
cv_error_results
# Choose the order that has the lowest CV error


# Fitting  ---------------------------------------------------------------------

fit = lm (wage ~ poly(age, 4, raw = T), data = Wage)
summary(fit)
# Generate a normal polynomial model

ortho_fit = lm (wage ~ poly(age, 4), data = Wage)
summary(ortho_fit)
# Generate an orthogonal polynomial model

# Alternate methods which return almost the same result as poly():

fit2a = lm(wage ~ age + I(age ^ 2) + I(age ^ 3) + I(age ^ 4), data = Wage)
# I() is a wrapper function for ^ to be correctly interpreted

fit2b = lm(wage ~ cbind(age, age ^ 2, age ^ 3, age ^ 4), data = Wage)


# Forecasting   ----------------------------------------------------------------

agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2])
# Generate range from predictions

preds = predict(fit, newdata = list(age = age.grid), se = TRUE)
se.bands = cbind(preds$fit + 2 * preds$se.fit, preds$fit - 2 * preds$se.fit)
# Calculate predictions

par(
  mfrow = c(1, 2),
  mar = c(4.5, 4.5, 1, 1),
  oma = c(0, 0, 4, 0)
)
# Defines plot paramaeters (grid, inner margin, outer margin)

plot(age,
     wage,
     xlim = agelims,
     cex = .5,
     col = "darkgrey")
title("Degree-4 Polynomial", outer = T)
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid,
         se.bands,
         lwd = 1,
         col = "blue",
         lty = 3)
# plot predictions


# Logistic Polynomial Regression -----------------------------------------------

fit = glm(I(wage > 250) ~ poly(age, 4), data = Wage, family = binomial)
# I() can create the binary variable automatically

preds = predict(fit, newdata = list(age = age.grid), se = T)

pfit = exp(preds$fit) / (1 + exp(preds$fit))
se.bands.logit = cbind(preds$fit + 2 * preds$se.fit, preds$fit - 2 * preds$se.fit)
se.bands = exp(se.bands.logit) / (1 + exp(se.bands.logit))
# Transform the logit results of predict() into probabilities and add a 95% CI

plot(
  age,
  I(wage > 250),
  xlim = agelims,
  type = "n",
  ylim = c(0, .2)
)
points(
  jitter(age),
  I((wage > 250) / 5),
  cex = .5,
  pch = "|",
  col = "darkgrey"
)
lines(age.grid, pfit, lwd = 2, col = "blue")
matlines(age.grid,
         se.bands,
         lwd = 1,
         col = "blue",
         lty = 3)
# plot the function (good idea to combine it with the non-logistic plot above)

par(mfrow = c(1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# STEP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

bins <- 4

fit = lm(wage ~ cut(age, bins), data = Wage)
# Fit the discretised variable

contrasts(cut(age, bins))
# inspects comparisons between the levels of the variable
summary(fit)

predicted_wages <- predict(fit)
# Use predict to get fitted values

agelims = range(age)
age.grid = seq(from = agelims[1], to = agelims[2])
# Fit 95% CI

preds = predict(fit, newdata = list(age = age.grid), se = TRUE)
se.bands = cbind(preds$fit + 2 * preds$se.fit, preds$fit - 2 * preds$se.fit)
# Generate 95% confidence range from predictions

plot(age,
     wage,
     xlim = agelims,
     cex = .5,
     col = "darkgrey")
title("Step Function", outer = T)
lines(sort(Wage$age),
      predicted_wages[order(Wage$age)],
      lwd = 2,
      col = "darkgreen")
matlines(
  age.grid,
  se.bands,
  lwd = 1,
  col = "darkgreen",
  lty = 5
)
# Plot the step function

# ──────────────────────────────────────────────────────────────────────────────
# SPLINES
# ──────────────────────────────────────────────────────────────────────────────

# Here we'll use library(splines)

# Regression Splines -----------------------------------------------------------

knots = c(25, 40, 60)

fit = lm(wage ~ bs(age, knots = knots), data = Wage)
pred = predict(fit, newdata = list(age = age.grid), se = T)
# Using the bs() function for polynomial splines (knots for predefined knots)

plot(age, wage, cex = .5, col = "gray")
title("Regression Splines with knots")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit + 2 * pred$se, lty = "dashed")
lines(age.grid, pred$fit - 2 * pred$se, lty = "dashed")
abline(
  v = knots,
  col = "darkgrey",
  lwd = 1,
  lty = 2
)
# Plotting


# Alternatively, the bs() function can find the appropriate knots
# if provided with the "df" argument

fit = lm(wage ~ bs(age, df = 6), data = Wage)
pred = predict(fit, newdata = list(age = age.grid), se = T)

knots <- attr(bs(age, df = 6), "knots")
# extracting knot locations

plot(age, wage, cex = .5, col = "gray")
title("Regression Splines with df")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit + 2 * pred$se, lty = "dashed")
lines(age.grid, pred$fit - 2 * pred$se, lty = "dashed")
abline(
  v = knots,
  col = "darkgrey",
  lwd = 1,
  lty = 2
)
# Plotting

# Natural Splines --------------------------------------------------------------

knots2 <- attr(ns(age, df = 4), "knots")
fit2 = lm(wage ~ ns(age, df = 4), data = Wage)
pred2 = predict(fit2, newdata = list(age = age.grid), se = T)
# this time using the ns() function for natural cubic splines

plot(age, wage, cex = .5, col = "gray")
title("Natural Splines")
lines(age.grid, pred2$fit, lwd = 2)
lines(age.grid, pred2$fit + 2 * pred2$se, lty = "dashed")
lines(age.grid, pred2$fit - 2 * pred2$se, lty = "dashed")
abline(
  v = knots2,
  col = "darkgrey",
  lwd = 1,
  lty = 2
)
# Plotting

# Smoothing Splines ------------------------------------------------------------

fit3 = smooth.spline(age, wage, df = 16)
# df can be passed on manually
fit3 = smooth.spline(age, wage, cv = TRUE)
# or calculated automatically with LOOCV

fit3
# to inspect

# Plot
plot(age, wage, cex = .5, col = "gray")
title("Smoothing Spline")
lines(fit3, lwd = 2)
# no knots and CI lines due to the nature of smoothing splines

# Local Linear Regression ------------------------------------------------------

fit4 = loess(wage ~ age, span = .2, data = Wage)
# span = 0.2 meaning the neighbourhood consists of 20% of the observations
# the larger the span, the smoother the fit

plot(age,
     wage,
     xlim = agelims,
     cex = .5,
     col = "darkgrey")
title("Local Regression")
lines(age.grid,
      predict(fit4, data.frame(age = age.grid)),
      col = "black",
      lwd = 2)
# Plotting

# ──────────────────────────────────────────────────────────────────────────────
# GENERALIZED ADDITIVE MODELS
# ──────────────────────────────────────────────────────────────────────────────

# GAM with natural splines -----------------------------------------------------
gam1 = lm(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)
summary(gam1)

par(mfrow = c(1, 3))
plot.Gam(gam1,
         se = TRUE,
         col = "darkgrey",
         main = "GAM  using natural splines")

# GAM wih smooth splines -------------------------------------------------------

# with the s() function from library(gam)
gam.m3 = gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage)
summary(gam.m3)

par(mfrow = c(1, 3))
plot(gam.m3,
     se = TRUE,
     col = "darkgrey",
     main = "GAM  using smooth splines")

# GAM with local regression ----------------------------------------------------

# with the lo() function from library(akima)
gam.lo = gam(wage ~ s(year, df = 4) + lo(age, span = 0.7) + education, data = Wage)
par(mfrow = c(1, 3))
plot.Gam(gam.lo, se = TRUE, col = "green")

gam.lo.i = gam(wage ~ lo(year, age, span = 0.5) + education, data = Wage)
par(mfrow = c(1, 2))
plot.Gam(gam.lo.i)

# Handling 3D Plots ------------------------------------------------------------

# The output of the above GAM plot will contain a 3D plot which is very hard
# to analyse using the base R plot window. To amend this, we can use
# library(rgl) (Requires installing XQuartz software)

x <- as.numeric(gam.lo.i$model$`lo(year, age, span = 0.5)`[, 1])
y <- as.numeric(gam.lo.i$model$`lo(year, age, span = 0.5)`[, 2])
z <- gam.lo.i$model$wage

# Create a 3D scatter plot
plot3d(
  x,
  y,
  z,
  type = "s",
  size = .5,
  xlab = "Year",
  ylab = "Age",
  zlab = "Wage"
)

# Customize the plot
grid3d(c("x", "y", "z")) # Add grid
title3d("3D Scatter Plot Example", line = 3)

# In this case, admittedly, it would be better to use multiple scatterplots
# for each year instead. However, this knowledge is still useful

# Model evaluation with ANOVA --------------------------------------------------

gam.m1 = gam(wage ~ s(age, 5) + education, data = Wage)
gam.m2 = gam(wage ~ year + s(age, 5) + education, data = Wage)
anova(gam.m1, gam.m2, gam.m3, test = "F")

# Logistic GAM -----------------------------------------------------------------

# Simply input the binomial variable at the beginning of the formula

gam.y.s = gam(
  I(wage > 250) ~ s(year, 4) + s(age, 5) + education,
  family = binomial,
  data = Wage,
  subset = (education != "1. < HS Grad")
)

plot(gam.y.s, se = T, col = "green")


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION TREES
# ──────────────────────────────────────────────────────────────────────────────

# Prepping example data
High = ifelse(Carseats$Sales <= 8, "0", "1")
Carseats = data.frame(Carseats, High)
Carseats$High = as.factor(Carseats$High)

# Functions from library(tree)

tree.carseats = tree(High ~ . - Sales, Carseats, split = c("deviance", "gini"))
# Declare the tree along the split decision metrics

summary(tree.carseats)
# View summary

tree.carseats
# View full tree in console

plot(tree.carseats)
text(tree.carseats, pretty = 0)
# plot with labels (hint: view in fullscreen)

# Evaluation -------------------------------------------------------------------

RNGkind("L'Ecuyer-CMRG")
set.seed(2)
# Declare method of RNG and seed for reproducibility

train = sample(1:nrow(Carseats), 200)
Carseats.test = Carseats[-train,]
High.test = High[-train]
# take random sample and split into train and test sets

tree.carseats = tree(High ~ . - Sales, Carseats, subset = train)
tree.pred = predict(tree.carseats, Carseats.test, type = "class")
tree.table <- table(tree.pred, High.test)
tree.table
# Generate new tree, predict and view prediction table

accuracy <- (tree.table[1, 1] + tree.table[2, 2]) / sum(tree.table)
accuracy # accuracy
1 - accuracy # test error

# Pruning ----------------------------------------------------------------------

set.seed(3)
# Reproducibility

cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
# FUN selects the metric by which to prune, here we choose misclassification

cv.carseats
# $size is the number of leaves in the tree
# $k is the alpha parameter
# $dev is the cross-validation error rate
# the size with the lowest CV error is the one to prune to

par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")
par(mfrow = c(1, 1))
# You can also check best size to prune visually

prune.carseats = prune.misclass(tree.carseats, best = 9)
# pruning to chosen parameter

plot(prune.carseats)
text(prune.carseats, pretty = 0)
# Examine pruning results

tree.pred = predict(prune.carseats, Carseats.test, type = "class")
tree.table <- table(tree.pred, High.test)
tree.table
accuracy <- (tree.table[1, 1] + tree.table[2, 2]) / sum(tree.table)
accuracy
1 - accuracy
# Examine test error

# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION TREES
# ──────────────────────────────────────────────────────────────────────────────

# Using the Boston dataset from library(MASS)

set.seed(1)
# Reproducibility

train = sample(1:nrow(Boston), nrow(Boston) / 2)
# Sample training set

tree.boston = tree(medv ~ ., Boston, subset = train)
# Fit regression tree

summary(tree.boston)

tree.boston

plot(tree.boston)
text(tree.boston, pretty = 0)
# Inspect the tree summary, console output and plot

# Pruning ----------------------------------------------------------------------

cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')

# select the size with the lowest $dev

prune.boston = prune.tree(tree.boston, best = 5)
# prune

plot(prune.boston)
text(prune.boston, pretty = 0)
# inspect

# Evaluation -------------------------------------------------------------------

# You can use this handy function:

evaluate_tree_model <- function(model, data, train_id, xvar) {
  # Make predictions
  yhat <- predict(model, newdata = data[-train_id, ])
  
  # Extract test labels
  test_labels <- data[-train_id, xvar]
  
  # Plot predictions vs. true values
  plot(yhat,
       test_labels,
       main = "Predicted vs. True Values",
       xlab = "Predicted Values",
       ylab = "True Values")
  
  # Add diagonal line
  abline(0, 1, col = "red")
  
  # Calculate mean squared error
  mse <- mean((yhat - test_labels) ^ 2)
  
  # Return MSE
  return(mse)
}

# Usage:
evaluate_tree_model(tree.boston, Boston, train, "medv")

# ──────────────────────────────────────────────────────────────────────────────
# BAGGING AND RANDOM FORESTS
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(randomForest)

set.seed(1)
# Reproducibility

# Bagging ----------------------------------------------------------------------

bag.boston = randomForest(
  medv ~ .,
  data = Boston,
  subset = train,
  mtry = 13,
  ntree = 1000
)
# Fit the bagging model. mtry HAS to be equal to the total amount of variables,
# otherwise the model will be a random forest. ntree = the number of grown trees

bag.boston
# inspect

yhat.bag = predict(bag.boston, newdata = Boston[-train,])
# make predictions

boston.test = Boston[-train, "medv"]
plot(yhat.bag, boston.test)
abline(0, 1)
# inspect accuracy visually

mean((yhat.bag - boston.test) ^ 2)
# Calculate test MSE

plot(bag.boston)
# NOTE: Cannot be plotted as a single chart, plot() will return the error
# as the amount of trees increases

# Random Forest ----------------------------------------------------------------

rf.boston = randomForest(
  medv ~ .,
  data = Boston,
  subset = train,
  mtry = 6,
  importance = TRUE
)

yhat.rf = predict(rf.boston, newdata = Boston[-train,])
mean((yhat.rf - boston.test) ^ 2)
# remember to check test mse to see if the model improves

importance(rf.boston)
varImpPlot(rf.boston)
# These will return the increase in metrics (here MSE and Node Purity - Gini)
# when the variable is randomly shuffled. The higher, the better, since if
# shuffling the variable has such a negative impact, it has to be important
# to the model

plot(rf.boston)
# plot() will result in a similiar chart as for the bagging approach

# ──────────────────────────────────────────────────────────────────────────────
# GRADIENT BOOSTING
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(gbm)

set.seed(1)
# Reproducibility

boost.boston = gbm(
  medv ~ .,
  data = Boston[train,],
  distribution = "gaussian",
  n.trees = 5000,
  interaction.depth = 4,
  shrinkage = 0.1,
  n.minobsinnode = 10,
  cv.folds = 10
)
# Fit gradient boost

summary(boost.boston)
# Check relative influence of predictors

plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")
# plot the predicted level of the dependent variable based on the level of the
# predictor

gbm.perf(boost.boston, method = "cv")
# plot training (black) and CV MSE (green) and best tree (blue)

yhat.boost = predict(boost.boston, newdata = Boston[-train,], n.trees = 5000)
mean((yhat.boost - boston.test) ^ 2)
# check MSE

# Tuning Model Parameters ------------------------------------------------------

# !!!WARNING!!! - Code below took ~3 mins to execute on an Apple M2 Chip

# create grid search to tune select the best parameter combination
hyper_grid <- expand.grid(
  n.trees = 6000,
  shrinkage = c(0.3, 0.1, 0.05, 0.01, 0.005),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for (i in seq_len(nrow(hyper_grid))) {
  # fit gbm for each parameter
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      formula = medv ~ .,
      data = Boston[train,],
      distribution = "gaussian",
      n.trees = hyper_grid$n.trees[i],
      shrinkage = hyper_grid$shrinkage[i],
      interaction.depth = hyper_grid$interaction.depth[i],
      n.minobsinnode = hyper_grid$n.minobsinnode[i],
      cv.folds = 10,
    )
  })
  
  # add SSE, best trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$time[i]  <- train_time[["elapsed"]]
  
}

# check results
arrange(hyper_grid, RMSE)

# ──────────────────────────────────────────────────────────────────────────────
# EXTREME GRADIENT BOOSTING
# ──────────────────────────────────────────────────────────────────────────────

# functions from library(xgboost) and library(recipes)

xgb_prep <- recipe(medv ~ ., data = Boston[train,]) %>%
  step_integer(all_nominal()) %>% # if data includes categorical columns
  prep(training = Boston[train,], retain = TRUE) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "medv")])
Y <- xgb_prep$medv

# Data has to be passed as a matrix of predictors and vector of predicted

set.seed(123)

ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  # iteration max ceiling
  objective = "reg:linear",
  early_stopping_rounds = 50,
  # model will stop if no improvement in N rounds
  nfold = 10,
  # number of cv folds
  params = list(
    eta = 0.1,
    # Learning rate
    max_depth = 3,
    min_child_weight = 3,
    # Minimum sum of instance weight
    subsample = 0.8,
    # Subsample ratio of the training instances
    colsample_bytree = 1.0,
    # Subsample ratio of columns
    alpha = 1,
    # L1 regularization term
    lamda = 1,
    # L2 regularization term
    gamma = 1 # minimum loss reduction
  ),
  verbose = 1
)
# run model

ames_xgb
# summary

# Tuning Model Parameters ------------------------------------------------------

# !!!WARNING!!! - Code below took ~9 mins to execute on an Apple M2 Chip

# create grid search to tune select the best parameter combination
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5,
  gamma = c(0, 1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = NA,
  trees = NA,
  time = NA
)

# grid search
for (i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  
  # Capture training time
  train_time <- system.time({
    m <- xgb.cv(
      data = X,
      label = Y,
      nrounds = 4000,
      objective = "reg:linear",
      early_stopping_rounds = 50,
      nfold = 10,
      verbose = 1,
      params = list(
        eta = hyper_grid$eta[i],
        max_depth = hyper_grid$max_depth[i],
        min_child_weight = hyper_grid$min_child_weight[i],
        subsample = hyper_grid$subsample[i],
        colsample_bytree = hyper_grid$colsample_bytree[i],
        gamma = hyper_grid$gamma[i],
        lambda = hyper_grid$lambda[i],
        alpha = hyper_grid$alpha[i]
      )
    )
  })
  
  # Store the RMSE, best iteration (trees), and training time in hyper_grid
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
  hyper_grid$time[i] <- train_time[["elapsed"]] # Store elapsed time
}

arrange(hyper_grid, rmse)

# ──────────────────────────────────────────────────────────────────────────────
# SUPPORT VECTOR MACHINES
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(e1071)

# Also possible with caret (e.g. method = "svmRadial"), see feature importance

# Using the iris dataset
# Only two classes for simplicity
iris_subset <- subset(iris, Species != "virginica")

# Only two features for visualization
iris_subset <-
  iris_subset[, c("Sepal.Length", "Sepal.Width", "Species")]
iris_subset <- droplevels(iris_subset)

# Train and test set
set.seed(123) # for reproducibility
indices <- sample(1:nrow(iris_subset), size = 0.7 * nrow(iris_subset))

train_iris <- iris_subset[indices, ]
test_iris <- iris_subset[-indices, ]

# Train the model
sv_model <-
  svm(Species ~ .,
      data = train_iris,
      type = "C-classification", # type of classification/regression
      kernel = "radial", # type of kernel function
      gamma = 1, # gamma parameter for latter two kernels
      cost = 1, # the c parameter
      decision.values = T, # return scores
      probability = T # return probabilities
  )

# Summary of the model
summary(sv_model)

# Tuning for best model
sv_tuning <-
  tune(
    svm, # function
    train.x = Species ~ .,
    data = train_iris,
    kernel = "radial",
    ranges = list( # values for testing
      cost = c(0.1, 1, 10, 100),
      gamma = c(0.1, 0.01, 0.001)
    ),
    decision.values = T,
    probability = T
  )

# check best parameters
sv_tuning

# extract and inspect best model
best_model <- sv_tuning$best.model
summary(best_model)

# Prediction
predictions <- predict(best_model, test_iris[-5])
confusionMatrix <- table(predictions, test_iris$Species)

# Check results
print(confusionMatrix)

# Calculate accuracy
sum(diag(confusionMatrix)) / sum(confusionMatrix)

# Plotting
plot(best_model, test_iris)
# if there are more than 2 dimensions, a formula is needed, e.g. x ~ y

# ──────────────────────────────────────────────────────────────────────────────
# SVM EXTRA 2-DIM PLOTTING APPROACH
# ──────────────────────────────────────────────────────────────────────────────

# Taken from the ESL book

# Function to create a grid over the data
make.grid = function(x, n = 75) {
  # Find the range of each feature in the 2-dimensional dataset
  grange = apply(x, 2, range)
  # Create a sequence of 'n' numbers over the ranges of both features
  x1 = seq(from = grange[1, 1], to = grange[2, 1], length = n)
  x2 = seq(from = grange[1, 2], to = grange[2, 2], length = n)
  # Generate a grid by creating all combinations of 'x1' and 'x2'
  expand.grid(x.1 = x1, x.2 = x2)
}

# Function to apply the grid, predict classes, and plot
svm_2d_plot = function(data,
                       feature_names,
                       model,
                       actual_classes) {
  # Extract features for grid creation
  x <- data[, feature_names]
  
  # Create a grid over the dataset
  xgrid = make.grid(x)
  names(xgrid) <- feature_names
  
  # Predict the class for each point in the grid using the trained model
  ygrid = predict(model, xgrid)
  
  # Plot the grid points colored by their predicted class
  plot(xgrid,
       col = c("red", "blue")[as.numeric(ygrid)],
       pch = 20,
       cex = .2)
  points(x, col = actual_classes, pch = 19) # Data points colored by class
  
  # Highlight the support vectors, if model has an 'index' attribute
  if (!is.null(model$index)) {
    points(x[model$index,], pch = 5, cex = 2)
  }
}

# Using the function -----------------------------------------------------------

svm_2d_plot(
  train_iris, # dataset
  c("Sepal.Width", "Sepal.Length"), # parameters to plot
  best_model, # svm model
  as.numeric(train_iris$Species) # classes
)

# ──────────────────────────────────────────────────────────────────────────────
# SVM ROC CURVES
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(ROCR)

# Predicting on the test set
predictions <- predict(best_model, test_iris, decision.values = TRUE)

# Extracting decision values
# The attributes of the predictions contain the decision values
decision.values <- attributes(predictions)$decision.values

# Actual binary class labels
# Convert factor to numeric binary class labels (setosa = 1, versicolor = 2)
actual <- as.numeric(test_iris$Species) - 1

# Creating the prediction object for ROCR
pred <- prediction(decision.values, actual)

# Calculating performance to plot
perf <- performance(pred, "tpr", "fpr")

# Plotting the ROC curve
plot(perf, main = "ROC Curve")
abline(a = 0, b = 1, lty = 2) # Adding a diagonal line
# The diagonal represents random chance - the curve should be above it

# Checking AUC
performance(pred, measure = "auc")@y.values[[1]]
# 0.5 is equal to random guessing, 1 means the classifier is perfect

# ──────────────────────────────────────────────────────────────────────────────
# SVM FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

# library(caret) approach required to work with other libraries
set.seed(5628)  
caret_sv <- train(
  Species ~ ., 
  data = train_iris,
  method = "svmRadial",               
  metric = "ROC",
  trControl = trainControl(
    method = "cv", 
    number = 10, 
    classProbs = TRUE,             
    summaryFunction = twoClassSummary
  ),
  tuneLength = 10
)

# Create a wrapper function to extract predicted probabilities
prob_iris <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "versicolor"]
}

# Variable importance plot from library(vip)
set.seed(2827)  # for reproducibility
vip(
  caret_sv,
  method = "permute",
  nsim = 5,
  train = train_iris,
  target = "Species",
  metric = "roc_auc",
  reference_class = "versicolor",
  pred_wrapper = prob_iris
)
# How much each variable impacts the classification

# Partial Dependence Plot from library(pdp)
features <- c("Sepal.Width", "Sepal.Length")
pdps <- lapply(features, function(x) {
  partial(
    caret_sv,
    pred.var = x,
    which.class = 2,
    prob = TRUE,
    plot = TRUE,
    plot.engine = "ggplot2"
  ) +
    coord_flip()
})

gridExtra::grid.arrange(grobs = pdps, ncol = 2)
# How much each value impacts the classification (higher yhat = more)

# ──────────────────────────────────────────────────────────────────────────────
# STACKED MODELS
# ──────────────────────────────────────────────────────────────────────────────

# prep dataset
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Obligatory first step: Initialise a JVM via h2o
h2o.init()


# Transform dataset
train_h2o <-
  prep(blueprint, training = ames_train, retain = TRUE) %>%
  juice() %>%
  # juice is bake equivalent that automatically uses the dataset from prep()
  as.h2o() # transform into h2o compatible object

test_h2o <- prep(blueprint, training = ames_train) %>%
  bake(new_data = ames_test) %>%
  as.h2o()

# Time Tracking ----------------------------------------------------------------

# Insert any function (e.g. like one for fitting the computationally heavy 
# models below) between these two blocks of code to measure the time it takes
# to execute

start.time <- proc.time()

# >>>Insert code here<<<

stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time[3])

# GLM Model --------------------------------------------------------------------

h2o_glm <- h2o.glm(
  x = setdiff(names(ames_train), "Sale_Price"), # response variable
  y = "Sale_Price", # predictor variables
  training_frame = train_h2o, # h2o dataframe input
  alpha = 0.1, # The elastic net mixing parameter
  remove_collinear_columns = TRUE,
  nfolds = 10, # Number of folds for cv
  fold_assignment = "Modulo", # Method for assigning observations to cv folds
  keep_cross_validation_predictions = TRUE,
  seed = 123 # reproducibility
)

# RF Model ---------------------------------------------------------------------

h2o_rf <- h2o.randomForest(
  x = setdiff(names(ames_train), "Sale_Price"),
  y = "Sale_Price",
  training_frame = train_h2o,
  ntrees = 100, # Number of trees to grow
  mtries = 20, # Number of variables randomly sampled at each split
  max_depth = 30, # Maximum depth of the trees
  min_rows = 1, # Minimum number of observations for a leaf
  sample_rate = 0.8, # Fraction of the train set used for growing each tree
  nfolds = 10, 
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123,
  stopping_rounds = 50, # rounds without improvement before early stopping
  stopping_metric = "RMSE", # Metric used for early stopping
  stopping_tolerance = 0 # Tolerance for the metric to be considered as improved
)

# GBM Model --------------------------------------------------------------------

h2o_gbm <- h2o.gbm(
  x = setdiff(names(ames_train), "Sale_Price"),
  y = "Sale_Price",
  training_frame = train_h2o, 
  ntrees = 500,
  learn_rate = 0.01, # Learning rate (shrinkage rate)
  max_depth = 7,
  min_rows = 5,
  sample_rate = 0.8, 
  nfolds = 10, 
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123, 
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Stacked Existing Models  -----------------------------------------------------

ensemble_tree <- h2o.stackedEnsemble(
  x = setdiff(names(ames_train), "Sale_Price"),
  y = "Sale_Price",
  training_frame = train_h2o,
  model_id = "my_tree_ensemble", # desired name of the model (aesthetic only)
  base_models = list(h2o_glm, h2o_rf, h2o_gbm),
  metalearner_algorithm = "drf" # chosen meta-learning algorithm (RF here)
)

# Evaluation:

# define function to extract RMSE
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}

# extract RMSE of individual models
list(h2o_glm, h2o_rf, h2o_gbm) %>%
  purrr::map_dbl(get_rmse)

# extract RMSE of the stacked model
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE

# Stacked Models from Grid  ----------------------------------------------------

# Create parameter grid
hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1),
  learn_rate_annealing = c(0.99, 1),
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1)
)

# Set the search criteria to randomly choose parameters from the grid 25 times
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 25)

h2o_grid <- h2o.grid(
  algorithm = "gbm", # train as GBM models
  grid_id = "h2o_grid", # aesthetic, name of grid search
  x = setdiff(names(ames_train), "Sale_Price"),
  y = "Sale_Price",
  training_frame = train_h2o,
  hyper_params = hyper_grid, # pass on the grid
  search_criteria = search_criteria, # pass on the search criteria
  ntrees = 50,
  stopping_metric = "RMSE",
  stopping_rounds = 10,
  stopping_tolerance = 0,
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123
)

# To get results:

h2o_grid # stored locally in R
#or
h2o.getGrid(grid_id = "h2o_grid", # stored "externally" in the JVM
            sort_by = "rmse")

# To get more info about a specific (here, the best) model:
best_model_id <- h2o_grid@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
h2o.performance(best_model, newdata = test_h2o)

# Automated Stacked Model Search -----------------------------------------------

# The function below will automatically parse through many different models and
# hyperparameters until the models converge (no improvement after x tries) or
# until it reaches the max runtime. The default is !!!ONE HOUR!!! so make sure
# you run it a background job or just have enough time for it. Also keep in
# mind that it will try to tie up as much of your computer's resources as
# possible, so it's use will be limited in the meantime.

# Use AutoML to generate a list of candidate models
auto_ml <- h2o.automl(
  x = setdiff(names(ames_train), "Sale_Price"),
  y = "Sale_Price",
  training_frame = train_h2o,
  nfolds = 5,
  max_runtime_secs = 60 * 120, # set max runtime, here 2 hours (120 mins)
  max_models = 20, # function will stop at 20 models
  keep_cross_validation_predictions = TRUE,
  sort_metric = "RMSE",
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0.02
)

# Show the top 25 models
auto_ml@leaderboard %>%
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)

# Get the top model with auto_ml@leader
auto_ml@leader

# Extract best model hyperparameters
auto_ml@leader@parameters

# To extract parameters for the model in the nth (here 4th) position
leaderboard_df <- as.data.frame(auto_ml@leaderboard)
model_id_fourth <- leaderboard_df$model_id[4]

# Extract the parameters of the model from H2O using its model_id
h2o.getModel(model_id_fourth)@parameters