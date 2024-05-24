# Machine Learning R HandsOn --------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

dataset <-
  readxl::read_excel("/Users/pawelgach/Documents/Aarhus Uni/Machine Learning II/HandsOn/Data.xls")

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
library(dplyr)
current_jobs <- dataset %>%
  filter(isCurrentJob == 1)

sum(current_jobs$jobEndingYear)
# Just NAs as expected
rm(current_jobs)

# Since jobEndingYear contributes pretty much the same amount of information as
# isCurrentJob, and is full of missing values, I decide to delete it along with
# other high NA columns
dataset <-
  dataset[,!(
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
dataset <- dataset[!is.na(dataset$employmentStatus), ]

# Before continuing, I also transform the reviewDateTime variable into a Date
# column and then transform it into a numeric variable with the amounts
# of seconds that passed since the earliest date

library(lubridate)

dataset$reviewDateTime <- ymd_hms(dataset$reviewDateTime)

# Find the earliest date-time in the reviewDateTime column
reference_datetime <- min(dataset$reviewDateTime)

# Calculate the number of seconds since the reference date-time and create a
# new column with it
dataset$reviewDateNumeric <-
  as.numeric(difftime(dataset$reviewDateTime, reference_datetime, units = "secs"))

# Delete the old column
dataset <- subset(dataset, select = -reviewDateTime)

# Now, I:
# - factorise, lump and dummy encode all character variables
# - scale all numeric variables
library(forcats)
library(caret)

# Factorize and lump character variables
dataset <- dataset %>%
  mutate(across(where(is.character), ~ fct_lump(factor(.), n = 5)))

# Create dummy variables for factorized columns
dummy_vars <- dummyVars("~ .", data = dataset)
data_dummy <- predict(dummy_vars, newdata = dataset)
data_dummy <- as.data.frame(data_dummy)

# Identify numeric columns
numeric_columns <- sapply(data_dummy, is.numeric)

# Scale numeric columns
data_scaled <- data_dummy
data_scaled[numeric_columns] <- scale(data_dummy[numeric_columns])

# The last step is splitting the data into test and train sets. I choose a 0.8
# split
library(rsample)

split <- initial_split(data_scaled, prop = 0.8)
train <- training(split)
test <- testing(split)

# ──────────────────────────────────────────────────────────────────────────────
# NON-LINEAR MODELS
# ──────────────────────────────────────────────────────────────────────────────

# Polynomials ------------------------------------------------------------------

# Step Functions ---------------------------------------------------------------

# Splines ----------------------------------------------------------------------

# GAMs -------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# TREE-BASED MODELS
# ──────────────────────────────────────────────────────────────────────────────

# Ordinary Trees ---------------------------------------------------------------

# Bagging ----------------------------------------------------------------------

# Random Forest ----------------------------------------------------------------

# Gradient Boosting ------------------------------------------------------------

# Extreme Gradient Boosting ----------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# SUPPORT VECTOR MACHINES
# ──────────────────────────────────────────────────────────────────────────────

