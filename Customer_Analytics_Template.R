# Customer Analytics Exam Template ---------------------------------------------

# Used libraries

library(foreign) # reading and writing unusual formats (SPSS, Stata, and SAS)
library(lavaan) # SEM and latent variable analysis
library(psych) # factor analysis and others
library(nFactors) # find number of factors to retain in EFA
library(semTools) # additional tools for SEM beyond lavaan
library(lavaanPlot) # visualizing lavaan SEM models
library(semPlot) # general SEM plotting
library(seminr) # PLS-SEM
library(bnlearn) # Bayesian Networks
library(Rgraphviz) # Visualising Bayesian Networks
library(dplyr) # Utilities
library(gRain)  # Probabilistic inference in Bayesian Networks
library(gRbase) # Alternative for Visualising Bayesian Networks
library(caTools) # Utilities
library(clickstream) # For clickstream analysis
library(dfidx) # For logit data
library(mlogit) # For mlogit models
library(readstata13) # For reading stata datasets
library(NbClust) # For hierarchical and non-hierarchical clustering
library(mclust) # For model-based clustering
library(flexclust) # For flexible clustering algorithms
library(partykit) # Visualising flexclust
library(flexmix) # For mixture distribution models
library(arules) # For association rules mining
library(arulesViz) # For arules visualisation
library(recommenderlab) # For product recommendation
library(tidyverse) # For data preprocessing

# ──────────────────────────────────────────────────────────────────────────────
# WORKING WITH SAV FILES
# ──────────────────────────────────────────────────────────────────────────────

# Loading data -----------------------------------------------------------------

data <-
  read.spss(
    "/Users/pawelgach/Documents/Aarhus Uni/Customer Analytics/HBAT.sav",
    to.data.frame = TRUE
  )
View(data)
skimr::skim(data)

# Retrieving variable names ----------------------------------------------------

VariableLabels <- unname(attr(data, "variable.labels"))

# ──────────────────────────────────────────────────────────────────────────────
# EXPLORATORY FACTOR ANALYSIS (EFA)
# ──────────────────────────────────────────────────────────────────────────────

# Checking for assumptions -----------------------------------------------------

# Correlation:
corrplot::corrplot(cor(data[, c(7:19)]))
#Plot, or
round(cor(data[, c(7:19)]), 2)
# in console
# The variables should have a good correlation for FA

# Bartlett Sphericity Test from library(psych)
cortest.bartlett(cor(data[, c(7:19)]), nrow(data[, c(7:19)]))
# If p-value < 0.05, the data cannot be reduced further, which is good

# Kaiser-Meyer-Oklin Test (KMO) also from library(psych)
KMO(data[, c(7:19)])
# KMO helps determine if the correlations between variables can be explained by
# other variables. The MSA should be >.60

# Applying EFA -----------------------------------------------------------------

# Using functions from library(nFactors)

ev <- eigen(cor(data[, c(7:19)]))
ev$values
# in console, or

plot(ev$values, type = "line")
abline(h = 1, lty = "dotted", lwd = 2)
grid()
# graphically

# First, we extract eigenvalues to check for the appropriate number of factors
# We do that by selecting the number of eigenvalues above 1


fit1 = factanal(
  ~ x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18,
  factors = 4,
  data = data,
  lower = 0.1,
  rotation = "varimax"
)

# You can pass raw data, a correlation matrix or a covariance matrix
# the function applies Maximum Likelihood by default.

print(fit1, sort = TRUE, cutoff = 0.2)
# print only the most relevant functions (with loadings >.2)

# When a variable has high loading in two factors, it means it is
# CROSS LOADED, and it should be deleted

# When a variable has no loading above .2, we might consider analysing it
# further or deleting it

# The test at the bottom checks whether the model is sufficient. Since the
# p-values is BELOW 0.05, the model is INSUFFICIENT

# Check Communalities (1 - Uniquness)
sort(apply(fit1$loadings ^ 2, 1, sum))

# Variables with low communality are generally unexplained by the model and
# are candidates for deletion

# ──────────────────────────────────────────────────────────────────────────────
# CONFIRMATORY FACTOR ANALYSIS (CFA)
# ──────────────────────────────────────────────────────────────────────────────

CFA.model <- 'CS =~ x18 + x9 + x16
              PV =~ x11 + x6 + x13
              MK =~ x12 + x7 + x10
              TS =~ x8 + x14

              CS ~~ PV
              CS ~~ MK
              CS ~~ TS
              PV ~~ MK
              PV ~~ TS
              MK ~~ TS'

# The CFA model has to be declared as a string. First, it contains the predicted
# factors and their variables ("=~") and then the correlated factors ("~~").
# By default, all factors are correlated, so there is no need to include the
# last 6 rows (they are here just to provide an example)

fit <- cfa(CFA.model, data = data)
# pass the string and data to the function

summary(
  fit,
  fit.measures = TRUE,
  standardized = TRUE,
  modindices = FALSE
)
# print a summary of the model

# Evaluation -------------------------------------------------------------------

# Guidelines for a good model:
# CFI >.90, TLI>.90, RMSEA< 0.08, SRMR <.0.08.
# + High and significant loadings

# Then, check Reliability with library(semTools)

semTools::reliability(fit)
# alpha =  Cronbach alpha - preferably > .5
# omega = similar to composite reliability index (CR) - should be > .7
# avevar = average variance extracted (AVE) - should be > .5

# check Discriminant Validity with the same library
# ensure that constructs (factors) do not overlap too much with each other
# (measure distinct concepts)

dV <- discriminantValidity(fit, merge = TRUE)
dV[c("lhs", "op", "rhs", "est")] # look at the "est" column
sqrt(reliability(fit)[5, ]) # sqrt of AVEs from previous function

# sqrt of AVE for a factor should be HIGHER than its correlations with other
# factors for it to be discriminantly valid

# Plotting ---------------------------------------------------------------------

# Using library(lavaanPlot)

lavaanPlot(
  model = fit,
  node_options = list(shape = "box", fontname = "Helvetica"),
  edge_options = list(color = "grey"),
  coefs = TRUE,
  covs = TRUE,
  stand = TRUE,
  sig = .05,
  stars = "covs"
)

# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURAL EQUATION MODELING (SEM)
# ──────────────────────────────────────────────────────────────────────────────

SEM.model1 <- '
# Measurement model
        CS =~ x18 + x9 + x16
        PV =~ x6 + x13
        MK =~ x12 + x7 + x10
        TS =~ x8 + x14
# Structural model
        x19 ~ CS + PV + MK + TS'

# SEM is declared as a string, similar to CFA
# The structual model is optional, here it is used to measure the impact of
# the factors on the x19 variable as an example

# fit and inspect the model
fitSEM1 <-
  sem(SEM.model1,
      data = data,
      se = "robust",
      estimator = "ML")

summary(
  fitSEM1,
  fit.measures = TRUE,
  standardized = TRUE,
  rsquare = TRUE
)

# Check MIs to see if any variable/factor would improve the model if it were
# freely estimated. Rule of thumb - it should be at least >10
modificationindices(fitSEM1,
                    sort = T,
                    minimum.value = 10,
                    op = "~~")

# to calculate the parameters with bootstrap, use:
bootstrapLavaan(fitSEM1,
                R = 10,
                type = "ordinary",
                FUN = "coef")

# Plotting ---------------------------------------------------------------------

lavaanPlot(
  model = fitSEM1,
  node_options = list(shape = "box", fontname = "Helvetica"),
  edge_options = list(color = "grey"),
  coefs = TRUE,
  covs = TRUE,
  stand = TRUE,
  sig = .05,
  stars = "regress",
  labels = list(x19 = "SATISFACTION")
)

# alternatively, from library(semPlot)
semPaths(
  fitSEM1,
  "std",
  intercepts = FALSE,
  style = "lisrel",
  layout = "tree2"
)

# ──────────────────────────────────────────────────────────────────────────────
# PARTIAL LEAST SQUARES REGRESSION SEM (PLS-SEM)
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(seminR)

satisfaction <- cSEM::satisfaction # dataset

# reduce dimensions to only those that will be used below to improve performance
sat_reduc <- satisfaction[, c("imag1", "imag2",
                              "expe1", "expe2",
                              "loy1", "loy2", "loy3",
                              "val1", "val2", "val3")]

# Measurement model
simple_mm <- constructs(
  composite("IMAG", # name of the latent variable (construct)
            multi_items("imag", 1:2), # indicators (items)
            weights = mode_B # method for calculating weights
  ),
  composite("EXPE", multi_items("expe", 1:2), weights = mode_B),
  composite("LOY", multi_items("loy", 1:3), weights = mode_A),
  composite("VAL", multi_items("val", 1:3), weights = mode_B)
)

# Structural model
simple_sm <- relationships(
  paths(
  from = c("IMAG", "EXPE"), # specifies intra-model paths
  to = c("LOY")
  ),
  paths(
    from = c("LOY"),
    to = c("VAL")
  )
)

# Run the model
sat_model <- estimate_pls(
  data = sat_reduc,
  measurement_model = simple_mm,
  structural_model = simple_sm,
  inner_weights = path_weighting, # weights algorithm, also try: path_factorial
  missing = mean_replacement, # how to handle NAs
  missing_value = "-99" # how are NAs indicated
)

# Summarize the model results
summary(sat_model)

# Plotting
plot(sat_model)

# Reflective evaluation --------------------------------------------------------

# Inspect the indicator loadings
summary(sat_model)$loadings
# Higher loadings = higher impact on the indicators from the latent variable

# Inspect the indicator reliability
summary(sat_model)$loadings ^ 2
# Should be higher than 0.708

# Inspect the Cronbachs alpha and composite reliability
summary(sat_model)$reliability
# alpha - Cronbachs alpha, should be higher than 0.7
# rhoC - composite reliability, should be higher than 0.7

# Plot the reliabilities of constructs
plot(summary(sat_model)$reliability)

# Table of the FL criteria
summary(sat_model)$validity$fl_criteria
# LR Criterions on the diagonal should be higher than the all the correlations 
# on the lower triangle

# HTMT criterion
summary(sat_model)$validity$htmt
# All values should fall BELOW 0.85 (more conservative) or 0.90

# Bootstrap the model
boot_seminr <- bootstrap_model(
  seminr_model = sat_model,
  nboot = 1000, # Could be more, less here for performance
  cores = NULL,
  seed = 123
)

# Extract the bootstrapped HTMT, all values should fall BELOW 0.85 or 0.90
summary(boot_seminr, alpha = 0.05)$bootstrapped_HTMT

# Check bootstraped loadings. Closer to 1 is better
summary(boot_seminr, alpha = 0.05)$bootstrapped_loadings

# Formative evaluation ---------------------------------------------------------

# Check base model weights, higher is better
summary(sat_model)$weights

# Redundancy analysis
# Note - this code doesn't make sense with this dataset, it's just an example
redundancy_mm <- constructs(
  composite("FORM", multi_items("imag", 1:2), weights = mode_B),
  composite("REFL", single_item("loy1"))
)
# Assuming imag is formative, val1 is reflective, both measure the same
# phenomenon

# Create structural model
redundancy_sm <- relationships(paths(from = c("FORM"), to = c("REFL")))

# Estimate the model
redundancy_model <- estimate_pls(
  data = sat_reduc,
  measurement_model = redundancy_mm,
  structural_model = redundancy_sm,
  missing = mean_replacement,
  missing_value = "-99"
)

# Summarize the model
summary(redundancy_model)

# Check path coefficients
summary(redundancy_model)$paths
# Path coefficien between REFL and FORM should be above 0.7

# Check bootstraped weights from before. Closer to 1 is better
summary(boot_seminr, alpha = 0.05)$bootstrapped_weights

# Structural evaluation --------------------------------------------------------

# Inspect the structural model collinearity VIF
summary(sat_model)$vif_antecedents
# Shouldn't be higher than 3 (more conservative) or 5

# Inspect the structural paths
summary(boot_seminr, alpha = 0.05)$bootstrapped_paths
# Inspect the total effects
summary(boot_seminr, alpha = 0.05)$bootstrapped_total_paths
# Both CI's should not include 0 in both of the above outputs

# Inspect the model RSquares
summary(sat_model)$paths
# Should be AT LEAST 0.25, preferably 0.5

# Inspect the effect sizes
summary(sat_model)$fSquare
# Should be AT LEAST 0.15, higher is better 

# Generate the model predictions
sat_predict <- predict_pls(model = sat_model,
                                    technique = predict_DA, 
                                    noFolds = 10, 
                                    reps = 10)

# Summarize the prediction results
summary(sat_predict)

# Analyze the distribution of prediction error of certain indicators
plot(summary(sat_predict), indicator = "val1")
# The more is at 0, the better

# Alternative models -----------------------------------------------------------

# Compare this for any two models:
summary(sat_model)$it_criteria["BIC", ]
# Lower values mean the model is better

# Mediation analysis -----------------------------------------------------------

specific_effect_significance(
  boot_seminr,
  from = "IMAG",
  through = "LOY",
  to = "VAL",
  alpha = 0.05
)
# Look at Original Est and Bootstrap Mean to inspect the strength and sign of 
# the mediation effect, compare to direct effect

# Moderation analysis -----------------------------------------------------------

# Create a new model with:
simple_mm <- constructs(
  composite("IMAG",multi_items("imag", 1:2), weights = mode_B),
  composite("EXPE", multi_items("expe", 1:2), weights = mode_B),
  composite("LOY", multi_items("loy", 1:3), weights = mode_A),
  composite("VAL", multi_items("val", 1:3), weights = mode_B),
  interaction_term(iv = "LOY", moderator = "EXPE", method = two_stage)
  # this line for a moderation effect
)

simple_sm <- relationships(
  paths(
    from = c("IMAG"),
    to = c("LOY")
  ),
  paths(
    from = c("LOY", "EXPE", "LOY*EXPE"),
    # The moderation variable, independent variable and interaction term here
    to = c("VAL")
    # the dependent variable here
  )
)

# Run the model as normal
mod_model <- estimate_pls(
  data = sat_reduc,
  measurement_model = simple_mm,
  structural_model = simple_sm,
  inner_weights = path_weighting,
  missing = mean_replacement,
  missing_value = "-99"
)

# Summarize the model results and check for significance
summary(mod_model)

# Run the rest of the model analysis as normal

# Slope analysis to check IV/DV relationship at different moderator values
slope_analysis(
  moderated_model = mod_model,
  dv = "VAL", # dependent variable
  moderator = "EXPE", # moderator
  iv = "LOY", # independent variable
   leg_place = "bottomright" # legend corner
)


# ──────────────────────────────────────────────────────────────────────────────
# BAYESIAN NETWORKS
# ──────────────────────────────────────────────────────────────────────────────

# Building the structure manually ----------------------------------------------

# Functions from library(bnlearn)

# Create an empty graph specifying the nodes (variables) of the network
dag <- empty.graph(nodes = c("Fuse", "Plea", "Atti", "Comm"))

# Adding arcs between nodes to define direct dependencies
# Arcs imply a directional influence from the 'from' node to the 'to' node
dag <- set.arc(dag, from = "Fuse", to = "Atti")
dag <- set.arc(dag, from = "Plea", to = "Atti")
dag <- set.arc(dag, from = "Fuse", to = "Comm")
dag <- set.arc(dag, from = "Plea", to = "Comm")
dag <- set.arc(dag, from = "Atti", to = "Comm")

# Display the DAG summary to check the structure details
print(dag)

# List direct dependencies for each variable in the network
modelstring(dag)
nodes(dag)  # Lists all the nodes in the graph
arcs(dag)   # Lists all the arcs in the graph

# Plot the graph to visualize the network
plot(dag)

# Optional: Advanced visualization with library(Rgraphviz)
graphviz.plot(dag)

# Alternative method using a matrix to define all arcs at once
dag2 <- empty.graph(nodes = c("Fuse", "Plea", "Atti", "Comm"))
arcs(dag2) <- matrix(
  c("Fuse", "Atti",
    "Plea", "Atti",
    "Fuse", "Comm",
    "Plea", "Comm",
    "Atti", "Comm"),
  byrow = TRUE, ncol = 2, dimnames = list(NULL, c("from", "to"))
)

plot(dag2)

# Alternative method using a model string
dag3 <- model2network("[Fuse][Plea][Atti|Fuse:Plea][Comm|Fuse:Plea:Atti]")

plot(dag3)


# Defining Conditional Probability Tables --------------------------------------
# A.K.A. CPTs

# Levels (Outcomes) for each node
Fuse.lv <- c("Low", "Med", "High")
Plea.lv <- c("Low", "Med", "High")
Atti.lv <- c("Low", "Med", "High")
Comm.lv <- c("Low", "Med", "High")

# CPT for 'Fuse'
Fuse.prob <-
  array(c(0.02, 0.26, 0.72),
        dim = 3,
        dimnames = list(Fuse = Fuse.lv))

# CPT for 'Plea'
Plea.prob <-
  array(c(0.01, 0.55, 0.44),
        dim = 3,
        dimnames = list(Plea = Plea.lv))

# CPT for 'Atti', depending on 'Fuse' and 'Plea'
Atti.prob <- array(
  c(0.99, 0.01, 0.00, ..., 0.09, 0.91),
  # Not going to type it all in. The amount needs to be a multiplication of the
  # dimensions, e.g. for (3,3,3) it would be 3*3*3 = 27 values
  dim = c(3, 3, 3),
  dimnames = list(Atti = Atti.lv, Plea = Plea.lv, Fuse = Fuse.lv)
)

# CPT for 'Comm', depending on 'Fuse', 'Plea', and 'Atti'
Comm.prob <- array(
  c(0.00, 1.00, 0.00, ..., 0.10, 0.90),
  # Just like above, here the code would contain 3*3*3*3 = 81 values
  dim = c(3, 3, 3, 3),
  dimnames = list(
    Comm = Comm.lv,
    Atti = Atti.lv,
    Plea = Plea.lv,
    Fuse = Fuse.lv
  )
)

# Combining the CPTs into a list
cpt <-
  list(
    Fuse = Fuse.prob,
    Plea = Plea.prob,
    Atti = Atti.prob,
    Comm = Comm.prob
  )

# Relating the DAG and CPT to define a fully-specified Bayesian Network
bn <- custom.fit(dag, cpt)

# Creating a DAG from existing data --------------------------------------------

# Dataset
retention <-
  read.csv(
    "retention.csv",
    header = T,
    colClasses = "factor"
  )

# There are four main constraint-based algorithms here:
par(mfrow = c(2,2))

bn.gs <- gs(retention,
            alpha = 0.05, # Significance level
            test = "x2") # Or test = "mi"
graphviz.plot(bn.gs, main = "Grow-Shrink Chi-Squared")

bn.iamb <- iamb(retention, alpha = 0.05, test = "mi")
graphviz.plot(bn.iamb, main = "IAMB Mutual Information")

bn.fast_iamb <- fast.iamb(retention, alpha = 0.05, test = "mi")
graphviz.plot(bn.fast_iamb, main = "Fast-IAMB Mutual Information")

bn.inter_iamb <- inter.iamb(retention, alpha = 0.05, test = "mi")
graphviz.plot(bn.inter_iamb, main = "Inter-IAMB Mutual Information")

par(mfrow = c(1,1))

# Note: they may leave some links undirected
# To check for them:

# Approach 1
undirected.arcs(bn.gs)

# Approach 2
# Look for lines without arrows on charts

# To direct an undirected link (or change existing)
bn.gs <- set.arc(bn.gs, from = "Atti", to = "Comm")
# also check out reverse.arc() and drop.arc()

# There is also a score-based algorithm
bn.hc <- hc(retention, score = "bic")
graphviz.plot(bn.hc, main = "Hill Climbing_BIC")

# To learn the parameters or a DAG, use:
bn.mle <- bn.fit(bn.gs, data = retention, method = "mle")
bn.mle # Summary
# Check individually
bn.mle$Fuse
bn.mle$Plea
bn.mle$Atti
bn.mle$Comm

# Model Evaluation -------------------------------------------------------------

# Metrics of model complexity
nodes(bn.mle)
arcs(bn.mle)
bn.mle

# Metrics of model sensitivity
# Test if any two nodes are d-separated
dsep(bn.mle, x = "Plea", y = "Fuse")
dsep(bn.mle, x = "Plea", y = "Comm")

# Metrics of relationship strength

options(scipen = 0) 
# Make R automatically choose between scientific and regular notation

arc.strength(bn.gs, retention, criterion = "x2") %>% .[order(.$strength), ]
# "x2" outputs the p-value of the relationships. The lower, the better

arc.strength(bn.gs, retention, criterion = "bic") %>% .[order(.$strength), ]
# "bic" gives the change in BIC caused by arc removal. Should be below 0
# The lower, the better. If above 0, removal improves the model

# Comparison of different DAG models
bnlearn::score(bn.gs, retention, type = "aic")
bnlearn::score(bn.gs, retention, type = "bic")
bnlearn::score(bn.gs, retention, type = "bde")
# Choose model with all three closest to zero (metrics might not agree)

# Metrics of predictive accuracy

# Functions from library(gRain) and library(gRbase)

# With k-fold CV
netcv = bn.cv (
  retention,
  bn.gs,
  loss = "pred", # Chosen loss function
  k = 5, # number of folds
  loss.args = list(target = "Comm"), # Specifies prediction (target) variable
  debug = TRUE # Addiitonal error/warning output
)
netcv

# Using a testing sample
retention_test <-
  read.csv(
    "retention_test.csv",
    header = T,
    colClasses = "factor"
  )

# Convert to library(gRain) class
net = as.grain(bn.mle)
net 

# Get the probability distribution for the 'Comm' node given new data
predComm = predict(
  net,
  response = c("Comm"), # Specifies prediction (target) variable
  newdata = retention_test, # Testing data
  predictors = names(retention_test)[-4], # variables used for predictions
  type = "distribution" # Output type selection
)

# Extract the predicted probabilities for 'Comm'
predComm = predComm$pred$Comm
predComm

# Predict the class label for 'Comm'
# Same setup as before, different output
predComm_class = predict(
  net,
  response = c("Comm"),
  newdata = retention_test,
  predictors = names(retention_test)[-4],
  type = "class"
)

# Extract the predicted class labels for 'Comm'
predCommclass = predComm_class$pred$Comm
predCommclass

# Create a confusion matrix
table(predComm_class$pred$Comm, retention_test$Comm)
# first argument (prediction) is left, second (actual) is top

# Plot the AUC with library(caTools)
colAUC(predComm, retention_test[, 4], plotROC = TRUE)

# Making Queries ---------------------------------------------------------------

# Create a library(gRain) object
junction <- compile(as.grain(bn.mle))

# marginal distribution (probabilities) of specific nodes (no set evidence)
querygrain(junction, nodes = "Atti")

# Setting evidence to predict conditional probabilities based on new information

# Set evidence where 'Fuse' is 'Low'.
jLow <- setEvidence(junction, nodes = "Fuse", states = "Low")
# Query conditional probabilities for 'Atti' and 'Comm' given 'Fuse' = Low.
A1 = querygrain(jLow, nodes = "Atti")
C1 = querygrain(jLow, nodes = "Comm")

# Do the same for Med and High
jMed <- setEvidence(junction, nodes = "Fuse", states = "Med")
A2 = querygrain(jMed, nodes = "Atti")
C2 = querygrain(jMed, nodes = "Comm")

jHigh <- setEvidence(junction, nodes = "Fuse", states = "High")
A3 = querygrain(jHigh, nodes = "Atti")
C3 = querygrain(jHigh, nodes = "Comm")

# Summary for 'Atti' across different 'Fuse' states (Low, Med, High).
# Aggregates the conditional probabilities for easier comparison.
AttiHigh <- c(A1$Atti[[1]], A2$Atti[[1]], A3$Atti[[1]])
AttiLow <- c(A1$Atti[[2]], A2$Atti[[2]], A3$Atti[[2]])
AttiMed <- c(A1$Atti[[3]], A2$Atti[[3]], A3$Atti[[3]])

# Create a data frame to store the summary information.
df1 <- data.frame(Fuse = c("Low", "Med", "High"), AttiLow, AttiMed, AttiHigh)

# Plot the conditional probabilities of 'Atti' for different 'Fuse' states
matplot(
  rownames(df1), # X-axis: Fuse states
  df1, # Y-axis: Conditional probabilities of 'Atti'
  type = 'l', # Line plot
  xlab = 'Fuse', # X-axis label
  ylab = '', # Y-axis label (empty)
  ylim = c(0, 1) # Set Y-axis limits to [0,1] for probabilities
)

# Add a legend to the plot for clarity.
legend(
  'topright', # Position
  inset = .01, # Slight inset from the top right corner
  legend = colnames(df1[, 2:4]), # Labels for the lines
  pch = 1, # Type of point
  horiz = TRUE, # Horizontal legend
  col = 2:4 # Colors
)

# ──────────────────────────────────────────────────────────────────────────────
# CLICKSTREAM ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(clickstream)

# Read a clickstream file
cls <-
  readClickstreams(file = "/Users/pawelgach/Desktop/Clickpath analysis code/sample.csv",
                   sep = ",", 
                   header = TRUE)

# Display
cls
summary(cls)

# Fit a Markov Chain model
mc <-
  fitMarkovChain(
    clickstreamList = cls,
    order = 2, # Set order of Markov Chain
    control = list(optimizer = "quadratic") # also try "linear"
  )

# Display
mc
summary(mc)
# Stats to compare different models
# LogLikelihood - the higher, the better
# AIC/BIC - the lower, the better
plot(mc, order = 2)
# Probability of trasitioning to one state (page) from another based on 2
# previous states

# Prediction -------------------------------------------------------------------

pattern <- new("Pattern", sequence = c("P9", "P2"))
# Declare the users current click pattern
resultPattern <- predict(mc, startPattern = pattern, dist = 1)
resultPattern
# predict and display likelihood of next page clicked

pattern <- new(
  "Pattern",
  sequence = c("P9", "P2"),
  absorbingProbabilities = data.frame(Buy = 0.333, Defer = 0.667)
)
# Declare the users pattern and total probabilities of concrete actions
resultPattern <- predict(mc, startPattern = pattern, dist = 2)
resultPattern
# Probability of the user clicking on the next page and taking an action

# Calculate user specific action probabilities based on past stats and current
# sequence
absorbingProbabilities <- c(0.5, 0.5) 
sequence <- c("P9", "P2")

# Loop through each state in the sequence
for (s in sequence) {
  # Update the probabilities with the probabilities of from the markov chain
  absorbingProbabilities <- absorbingProbabilities *
    data.matrix(subset(
      mc@absorbingProbabilities,
      state == s,
      select = c("Buy", "Defer")
    ))
}

# Normalize the probabilities to 1 and display
absorbingProbabilities <- absorbingProbabilities / sum(absorbingProbabilities)
absorbingProbabilities

# Clustering Clickstreams ------------------------------------------------------

# Seed for reproducibility
set.seed(12345)

# Build a set amount (3 here) of clusters
clusters <- clusterClickstreams(clickstreamList = cls, order = 1, centers = 3)

# Display
clusters
clusters$clusters[[1]]
clusters$clusters[[2]]
clusters$clusters[[3]]

# You can then fit Markov Chains to each cluster individually

# Finding the right order ------------------------------------------------------

maxOrder <- 5 
result <- data.frame()
for (k in 1:maxOrder) {
  mc <- fitMarkovChain(clickstreamList = cls, order = k)
  result <- rbind(result, c(k, summary(mc)$aic, summary(mc)$bic))
}

names(result) <- c("Order", "AIC", "BIC")
result

# Pick the one with lowest BIC and AIC


# ──────────────────────────────────────────────────────────────────────────────
# CHOICE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

# Load example dataset

cbc.df <- read.csv(
  "http://goo.gl/5xQObB",
  colClasses = c(
    seat = "factor",
    cargo = "factor",
    price = "factor",
    choice = "integer"
  )
)

cbc.df$eng <- factor(cbc.df$eng, levels = c("gas", "hyb", "elec"))
cbc.df$carpool <- factor(cbc.df$carpool, levels = c("yes", "no"))
summary(cbc.df)

# Display the distribution of the binary choice column across different columns
xtabs(choice ~ price, data = cbc.df)
xtabs(choice ~ cargo, data = cbc.df)
xtabs(choice ~ carpool, data = cbc.df)
xtabs(choice ~ seat, data = cbc.df)
xtabs(choice ~ eng, data = cbc.df)


# convert the data to an mlogit.data with library(dfidx)

# add a column with unique question numbers (3 for 3 alternatives per question)
cbc.df$chid <- rep(1:(nrow(cbc.df) / 3), each = 3)

# shape the data for mlogit
cbc.mlogit <- dfidx(cbc.df, choice = "choice",
                    idx = list(c("chid", "resp.id"), "alt"))

# fitting a nmlogit model with library(mlogit)
m1 <- mlogit(choice ~ 0 + # 0 for no intercept
               seat + 
               cargo + 
               eng + 
               price,
             data = cbc.mlogit)
summary(m1)

# Fitting a model with continuous price
m2 <- mlogit(choice ~ 0 + 
               seat + 
               cargo + 
               eng + 
               as.numeric(as.character(price)),
             data=cbc.mlogit)
summary(m2)

# Comparing two models with a likelihood ratio test
lrtest(m1,m2)
# Higher LogLik is better
# Higher Chisq is better
# Most important - insignificant p-value means models are essentially the same

# Prediction -------------------------------------------------------------------

# How much more are customers willing to pay for feature?
# Requires price as continuous
coef(m2)["cargo3ft"]/(-coef(m2)["as.numeric(as.character(price))"]/1000)

# Predict likelihood of users choosing a product with given attributes

predict.mnl <- function(model, data) {
  data.model <- model.matrix(update(model$formula, 0 ~ .), data = data)[, -1]
  # Build model based on given attributes
  utility <- data.model %*% model$coef
  share <- exp(utility) / sum(exp(utility))
  # Report purchase probability
  cbind(share, data)
}

# example data
attrib <- list(
  seat = c("6", "7", "8"),
  cargo = c("2ft", "3ft"),
  eng = c("gas", "hyb", "elec"),
  price = c("30", "35", "40")
)

new.data <- expand.grid(attrib)[c(8, 1, 3, 41, 49, 26), ]

# predict and display by rank
scores <- predict.mnl(m3, new.data)
scores[order(scores$share, decreasing = T), ]


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

# Preparing the external dataset -----------------------------------------------

# Load the dataset using library(readstata13)
HBAT <- read.dta13("/Users/pawelgach/Desktop/Customer Segmentation Code/hbat.dta")

# Create column with unique IDs
HBAT$id <- seq_along(HBAT[,1])

# Define variables of interest
hbat <- c("x6", "x8", "x12", "x15", "x18")

# Remove identified outliers
HBAT_cleaned <- HBAT[!HBAT$id %in% c(6, 87), ]
nobs_clean <- nrow(HBAT_cleaned)
HBAT_cleaned$id <- seq(1, nobs_clean)


# Hierarchical clustering using Ward's method ----------------------------------

# Create a squared Euclidean distance matrix
dist2 <- dist(HBAT[, hbat], method = "euclidean") ^ 2

# Perform hierarchical clustering
H.fit <- hclust(dist2, method = "ward.D")

# Plot the dendrogram
plot(H.fit)

# Determine the point with the highest percentage increase in clustering
denominator <- cumsum(H.fit[[2]])
length(denominator) <- dim(HBAT)[1]-2
denominator <- c(1,denominator)
pct <- H.fit[[2]]/denominator
cbind(tail(order(pct, decreasing = F), n = 10),
      pct[tail(order(pct, decreasing = F), n = 10)])
# Look for significant jumps - here we could use 4 or 5 for example, as 3 or 2
# clusters might not be enough

# Cut the dendrogram based on the chosen number of clusters
grp <- cutree(H.fit, k = 4)

# Highlight the optimal cluster solution on the dendrogram
plot(H.fit)
rect.hclust(H.fit, k = 4, border = "red")

# Summary of averages per cluster
aggregate(HBAT[, hbat], by = list(cluster = grp), FUN = mean)

# ANOVA
lapply(HBAT[, hbat], function(x)
  summary(aov(x ~ grp, data = HBAT)))
# If the p-value is significant, the variable is distinct in different clusters

# Hierarchical clustering using Complete linkage -------------------------------

# Perform clustering using complete linkage
H.fit_complete <-
  hclust(dist(HBAT[, hbat], method = "euclidean"), method = "complete")

# Plot the dendrogram
plot(H.fit_complete)

# Use NbClust to determine the best number of clusters
res <-
  NbClust(
    HBAT[, hbat],
    distance = "euclidean",
    min.nc = 2,
    max.nc = 8,
    method = "ward.D",
    index = "all"
  )

# Print the results from NbClust
res$Best.nc
res$All.index

# Non-Hierarchical Clustering --------------------------------------------------

# Seed for reproducibility
set.seed(4118)

# Perform k-means clustering with 4 clusters and 25 random starts
NH.fit <- kmeans(HBAT[, hbat], 4, nstart = 25)
print(NH.fit) # Print clustering results

# Convert cluster assignments to a factor for further analysis
grp <- as.factor(NH.fit[[1]])

# Display the size of each cluster
table(grp)

# Calculate and display the mean of each variable within each cluster
aggregate(HBAT[, hbat], list(grp), mean)

# Perform ANOVA
lapply(HBAT[, hbat], function(x)
  summary(aov(x ~ grp, data = HBAT)))

# Plot cluster centers using a line plot (commonly known as a snake plot)
matplot(
  t(NH.fit$centers),
  type = "l",
  col = 1:4,
  lty = 1:4,
  xlab = "Variable",
  ylab = "Center Value"
)
legend(
  "topright",
  legend = paste("Cluster", 1:4),
  col = 1:4,
  lty = 1:4
)

# Calculate means for additional variables to assess criterion validity
aggregate(HBAT[, c("x19", "x20", "x21", "x22")], list(grp), mean)

# Perform ANOVA for these additional variables
lapply(HBAT[, c("x19","x20","x21","x22")], function(x)
  summary(aov(x ~ grp, data = HBAT)))

# Perform profiling of categorical variables
profile_categorical <- function(var) {
  tbl <- table(HBAT[[var]], grp)
  cat("Profiling for:", var, "\n")
  print(round(100 * prop.table(tbl, 2), 1)) # Print percentage distribution
  print(chisq.test(tbl)) # Perform and print chi-squared test results
}

# Apply profiling function to categorical variables
profile_categorical("x1")
# Distribution of variable (factor) levels across clusters, as well as
# the p-value of its significance in constructing the cluster

# Model-based clustering -------------------------------------------------------

# Function from library(mclust)
(mc <- Mclust(HBAT[,hbat]))
# Note: With parentheses, the value is both assigned and printed to the console

# Visualise different solutions
plot(mc, HBAT[,hbat], what = "BIC", col = "black")

# Summarize
summary(mc)
# BIC and ICL - the lower, the better

# Look at a 4-cluster solution
mc4 <- Mclust(HBAT[,hbat],G=4)
summary(mc4)

# Flexible clustering algorithms -----------------------------------------------

# Loading and preparing external dataset
mcdonalds <-
  read.csv("/Users/pawelgach/Desktop/Customer Segmentation Code/mcdonalds.csv")
# Convert Yes/No responses to binary (1/0)
MD.x <- as.matrix(mcdonalds[, 1:11] == "Yes") * 1
# Principal Component Analysis (PCA) to reduce dimensions (and to be used later)
(MD.pca <- prcomp(MD.x))

# Reproducibility
set.seed(28)

# k-means clustering
MD.km28 <- stepFlexclust(MD.x, 2:8, nrep = 10, verbose = FALSE)
# Plot to determine optimal number of segments
plot(MD.km28, xlab = "number of segments")

# Stability of clustering solutions using bootstrap method
# (Takes like 30 seconds)
MD.b28 <- bootFlexclust(MD.x, 2:8, nrep = 10, nboot = 100)
# Plot to assess segment stability
plot(MD.b28, xlab = "number of segments", ylab = "adjusted Rand index")
# Higher ARI = Better

# Save and plot the four-segment solution
MD.k4 <- relabel(MD.km28)[["4"]]
plot(
  slswFlexclust(MD.x, MD.k4),
  ylim = 0:1,
  xlab = "segment number",
  ylab = "segment stability"
)

# Gorge plot
MD.km28 <- relabel(MD.km28)
histogram(MD.km28[["4"]], data = MD.x, xlim = 0:1)

# Plot of segment level stability across solutions
slsaplot(MD.km28)

# Hierarchical clustering to visualize segment profiles
MD.vclust <- hclust(dist(t(MD.x)))

# Barchart of segment profiles
barchart(MD.k4, shade = TRUE, which = rev(MD.vclust$order))

# Segment separation using principal components
plot(
  MD.k4,
  project = MD.pca,
  data = MD.x,
  hull = FALSE,
  simlines = FALSE,
  xlab = "principal component 1",
  ylab = "principal component 2"
)

# Visualising flexible clusters ------------------------------------------------

# Extract the segment membership for each consumer
mcdonalds$k4 <- clusters(MD.k4)

# Convert the Like scale into a factor with ordered levels
mcdonalds$like <-
  factor(
    mcdonalds$Like,
    levels = c(
      "I love it!+5",
      "+4",
      "+3",
      "+2",
      "+1",
      "0",
      "-1",
      "-2",
      "-3",
      "-4",
      "I hate it!-5"
    )
  )

# Mosaic plot - Visualize like/hate distribution across segments
mosaicplot(
  table(mcdonalds$k4, mcdonalds$like),
  shade = TRUE,
  main = "",
  xlab = "Segment Number",
  col = rainbow(10)
)

# Visualize gender distribution across segments
# mosaicplot(
#   table(mcdonalds$k4, mcdonalds$Gender),
#   shade = TRUE,
#   col = c("pink", "lightblue")
# )

# Age distribution across segments with a boxplot
boxplot(
  mcdonalds$Age ~ mcdonalds$k4,
  varwidth = TRUE,
  notch = TRUE,
  main = "Age Distribution by Segment",
  xlab = "Segment",
  ylab = "Age"
)

# Predict segment 3 membership using a classification tree
# Prepare the VisitFrequency variable as a factor with ordered levels
mcdonalds$visitfrequency <-
  factor(
    mcdonalds$VisitFrequency,
    levels = c(
      "Never",
      "Once a year",
      "Every three months",
      "Once a month",
      "Once a week",
      "More than once a week"
    )
  )

# Build and plot a classification tree
tree <-
  ctree(factor(k4 == 3) ~ like + Age + visitfrequency + factor(Gender),
        data = mcdonalds)
plot(tree, main = "Classification Tree for Segment 3 Membership")

# Selecting target segments
# Calculate average visit frequency per segment
visit <-
  tapply(as.numeric(mcdonalds$visitfrequency), mcdonalds$k4, mean)

# Recenter and invert the like scale
like <- tapply(-as.numeric(mcdonalds$like) + 6, mcdonalds$k4, mean)

# Calculate the proportion of females in each segment
female <-
  tapply((mcdonalds$Gender == "Female") + 0, mcdonalds$k4, mean)

# Plot segment evaluation focusing on visit frequency and like scale with size 
# indicating female proportion
plot(
  visit,
  like,
  cex = 10 * female,
  xlim = c(2, 4.5),
  ylim = c(-3, 3),
  xlab = "Average Visit Frequency",
  ylab = "Adjusted Like Scale"
)
text(visit, like, labels = 1:4, pos = 3)  # Add segment labels

# Mixture Distribution ---------------------------------------------------------

# Fit mixture models over a range of components (from 2 to 8)
MD.m28 <-
  stepFlexmix(
    MD.x ~ 1,
    k = 2:8,
    nrep = 10,
    model = FLXMCmvbinary(),
    verbose = FALSE
  )

MD.m28

# Plot AIC, BIC, ICL for each model
plot(MD.m28, ylab = "Value of Information Criteria (AIC, BIC, ICL)")

# Select and extract the model with 4 clusters
MD.m4 <- getModel(MD.m28, which = "4")

# Compare cluster assignments
comparison_table <-
  table(kmeans = clusters(MD.k4), mixture = clusters(MD.m4))
print(comparison_table)

# Refit the mixture model using the k-means clusters as initial assignments
MD.m4a <-
  flexmix(MD.x ~ 1, cluster = clusters(MD.k4), model = FLXMCmvbinary())

# Compare the cluster assignments
adjusted_comparison_table <-
  table(kmeans = clusters(MD.k4), mixture = clusters(MD.m4a))
print(adjusted_comparison_table)

# Log-likelihood for the adjusted mixture model
log_likelihood_adjusted <- logLik(MD.m4a)
print(log_likelihood_adjusted)

# Log-likelihood for the original mixture model
log_likelihood_original <- logLik(MD.m4)
print(log_likelihood_original)
# Higher loglik = better

# ──────────────────────────────────────────────────────────────────────────────
# PRODUCT RECOMMENDATION
# ──────────────────────────────────────────────────────────────────────────────

# Functions from library(arules)

data(Groceries)
# Dataset

# To transform into "transactions" class

# trans_list <- data %>%
#   group_by(transaction) %>%
#   summarise(items = list(product), .groups = 'drop') %>%
#   pull(items)
# 
# trans_list <- lapply(trans_list, unlist)
# transactions <- as(trans_list, "transactions")

# Inspect transactions object
summary(Groceries)

inspect(Groceries) # Display all. works with head() and tail()
inspect(head(Groceries, 3))

# Display most frequent
tail(sort(itemFrequency(Groceries)), 30)
itemFrequencyPlot(Groceries, support = 0.1, cex.names = 1.0)

# Apriori algorithm ------------------------------------------------------------

rules <-
  apriori(Groceries,
          parameter = list(
            supp = 0.01, # minimal support
            conf = 0.8, # minimal confidence
            target = "rules" # what to proritise
            # ?APparameter
          ))

# Inspect results
inspect(subset(groc.rules, lift > 3))

# Display with library(arulesViz)
plot(groc.rules,
     method = "graph",
     engine = "htmlwidget",
     max = 1000)

# Select rules with highest lift
groc.hi <- head(sort(groc.rules, by = "lift"), 50)

# Prep for Collaborative filtering ---------------------------------------------

# Functions from library(recommenderlab)

# Dataset
data(MovieLense)

# First few ratings of the first user
head(as(MovieLense[1, ], "list")[[1]])

# Number of ratings per user
hist(rowCounts(MovieLense))

# Number of ratings per movie (Item)
hist(colCounts(MovieLense))

# Top 10 movies (items) by frequency
movie_watched <- data.frame(movie_name = names(colCounts(MovieLense)),
                            watched_times = colCounts(MovieLense))
movie_watched[order(movie_watched$watched_times, decreasing = TRUE), ][1:10, ]

# Selecting non-sparse data:
# At least 100 users who evaluated at least 30 items for each item
rates <- MovieLense[rowCounts(MovieLense) > 30, colCounts(MovieLense) > 100]
rates <- rates[rowCounts(rates) > 30, ]

# Data should be split into training and testing first
which_train <-
  sample(
    x = c(TRUE, FALSE),
    size = nrow(rates),
    replace = TRUE,
    prob = c(0.8, 0.2)
  )

train <- rates[which_train, ]
test <- rates[!which_train, ]

# Choosing the right model -----------------------------------------------------

# Load the registry of available recommender model algorithms for data type
(recommender_models <-
  recommenderRegistry$get_entries(dataType = "realRatingMatrix"))

# See what models are available for this type of data without the details
names(recommender_models)

# Describe each approach
lapply(recommender_models, "[[", "description")

# Access parameters of specific model
recommender_models$IBCF_realRatingMatrix$parameters

# Item-based Collaborative Filtering -------------------------------------------

# Create a recommender model using the 'Recommender' function
rec_model <-
  Recommender(data = train,
              method = "IBCF",
              parameter = list( # You can also leave this all as default
                k = 30, # Number of neighbors to consider
                method = "Cosine", # Similarity measure to use
                normalize = "Z-score", # Method for normalizing item ratings
                na_as_zero = TRUE # Treat NA values as zero (for sparse data)
              ))

# Generate predictions for test set
rec_predicted <- predict(object = rec_model, 
                          newdata = test, 
                          n = 5) # predictions to make per user


# Extract the recommendations for the first user
rec_predicted@items[[1]]

# Create a matrix of recommended items for each user
rec_matrix <-
  lapply(rec_predicted@items, function(x)
    colnames(rates)[x])

# Display the recommendations for the first four users
rec_matrix[1:4]

# User-based Collaborative Filtering -------------------------------------------

# Build a recommender model using UBCF
rec_model_2 <- Recommender(data = train, method = "UBCF")

# Generate predictions
rec_predicted_2 <-
  predict(object = rec_model_2,
          newdata = test,
          n = 5)

# Create a matrix of recommended items for each user
rec_matrix_2 <-
  sapply(rec_predicted_2@items, function(x)
    colnames(rates)[x])

# Display the recommendations for the first four users
rec_matrix_2[,1:4]

# Evaluation -------------------------------------------------------------------

# Cross-validation

eval_sets <-
  evaluationScheme(
    data = rates,
    method = "cross-validation",
    k = 4, # number of folds
    given = 20, # use only 20 randomly picked ratings per user to predict
    goodRating = 4 # threshold to consider an item as "good"
  )

# Make predictions with CV
eval_prediction <-
  predict(
    object = rec_model,
    newdata = getData(eval_sets, "known"),
    n = 5,
    type = "ratings"
  )

# Calculate accuracy metrics such as RMSE, MSE, and MAE
eval_accuracy <- calcPredictionAccuracy(x = eval_prediction,
                                        data = getData(eval_sets, "unknown"),
                                        byUser = TRUE)

# Show a sample of prediction accuracy results by user
head(eval_accuracy)

# Plot the distribution of RMSE for each user
ggplot(data = as.data.frame(eval_accuracy), aes(x = RMSE)) + 
  geom_histogram(binwidth = 0.1) +
  ggtitle("Distribution of the RMSE by user")

# Evaluate the model as a whole
calcPredictionAccuracy(x = eval_prediction,
                       data = getData(eval_sets, "unknown"),
                       byUser = FALSE)

## Evaluation of IBCF top-N recommendations
# Construct a confusion matrix to evaluate recommendation accuracy
results <-
  evaluate(x = eval_sets,
           method = "IBCF",
           n = seq(10, 100, 10))

# Display the confusion matrix for the first fold of cross-validation
head(getConfusionMatrix(results)[[1]])

# Sum up the confusion matrix indices across all splits
columns_to_sum <- c("TP", "FP", "FN", "TN")
indices_summed <- Reduce("+", getConfusionMatrix(results))[, columns_to_sum]
head(indices_summed)

# Plot ROC curve for visual evaluation of model performance
plot(results, annotate = TRUE, main = "ROC curve")
# Ideally, the ROC curve should be close to the upper left corner of the plot, 
# indicating a high TPR (sensitivity) and a low FPR (specificity).

# Plot precision-recall curve to assess the accuracy and recall of the model
plot(results, "prec/rec", annotate = TRUE, main = "Precision-Recall")
# Ideally, the precision-recall curve should approach the top-right corner, 
# indicating both high precision (few FP among predicted positives) and high 
# recall (high percentage of actual positives correctly identified)

# Comparing different models ---------------------------------------------------

# Define a list of models with different settings for evaluation
models_to_evaluate <- list(
  IBCF_cos = list(name = "IBCF", param = list(method = "cosine")),
  IBCF_cor = list(name = "IBCF", param = list(method = "pearson")),
  UBCF_cos = list(name = "UBCF", param = list(method = "cosine")),
  UBCF_cor = list(name = "UBCF", param = list(method = "pearson")),
  random = list(name = "RANDOM", param = NULL)
)

# Evaluate models using a range of top-n recommendations
n_recommendations <- c(1, 5, seq(10, 100, 10))
list_results <-
  evaluate(x = eval_sets, method = models_to_evaluate, n = n_recommendations)

# Plot ROC curve to compare models
plot(list_results, annotate = 1, legend = "topleft")
title("ROC curve")

# Plot precision-recall for different models
plot(
  list_results,
  "prec/rec",
  annotate = 1,
  legend = "bottomright",
  ylim = c(0, 0.4)
)
title("Precision-recall")
