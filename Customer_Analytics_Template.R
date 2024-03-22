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
