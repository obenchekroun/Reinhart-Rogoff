###################################################################################
### Analyse de la corrélation entre dette et croissance - Reinhardt and Rogoff  ###
###################################################################################

rm(list=ls())
cat("\014")
Sys.setlocale("LC_ALL", 'en_GB.UTF-8')

library(tidyquant)  # Loads tidyverse and several other pkgs 
library(tidyverse)

library(tidyverse)
library(caret) # machine learning // Classification And REgression Training
theme_set(theme_classic()) # pour définir le thème des graphs de ggplot
library(splines) # model en splines
library(mgcv) #Generalized additive models

library(readxl) # Manipulation des excels
library(ggstatsplot) #visualisation statistique


if(Sys.getenv("RSTUDIO") == "1"){
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #Definition du répertoire de travail
}

##############################
### Préparation de la data ###
##############################

RR_data <- read_excel("RR.xlsx",sheet = "Mean") %>%
  select(-"Debt category")
names(RR_data) <- c("Country", "Year", "Debt","Growth")

# Split the data into training and test set
set.seed(123)
training.samples <- RR_data$Growth %>%
  createDataPartition(p = 0.8, list = FALSE)
RR.train.data  <- RR_data[training.samples, ]
RR.test.data <- RR_data[-training.samples, ]

################################
### Recherche de corrélation ###
################################

RR_data %>%
  ggplot(aes(x = Debt, y = Growth)) +
  geom_point(colour = "blue")

RR_data %>%
  ggplot(aes(x = Debt, y = Growth)) +
  xlim(0,120) + ylim(-5,10) +
  geom_point(colour = "blue")

qqplot(RR_data$Debt,RR_data$Growth)

# calculate Pearson's r
#cor(RR_data$Debt, RR_data$Growth) #corrélation de pearson pour vérifier si corrélé
cor.test(RR_data$Debt, RR_data$Growth, method = "pearson")
ggscatterstats(data = RR_data, x = Debt, y = Growth)

lmodel <- lm(Growth ~ Debt, data = RR_data)
lmodel$coefficients
summary(lmodel)

RR_data %>%
  ggplot(aes(x = Debt, y = Growth)) +
  geom_point(colour = "blue") +
  stat_smooth(colour ="green") +
  geom_smooth(method = "lm", fill = NA, colour = "red")


##########Test de différents types de corrélation
##### régression linéaire
##### Growth = b0 + b1*Debt

# Build the model
model <- lm(Growth ~ Debt, data = RR.train.data)

# Make predictions
predictions <- model %>% predict(RR.test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, RR.test.data$Growth),
  R2 = R2(predictions, RR.test.data$Growth)
)

ggplot(RR.train.data, aes(x = Debt, y = Growth) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ x)

##### Régression polynomial
##### Growth=b0+b1∗Debt+b2∗Debt^2
lm(Growth ~ Debt + I(Debt^2), data = RR.train.data)
lm(Growth ~ poly(Debt, 2, raw = TRUE), data = RR.train.data)  #formule alternative

## Rang 6
lm(Growth ~ poly(Debt, 6, raw = TRUE), data = RR.train.data) %>%
  summary() ## On voit que les termes de rang sup à 5 ne sont pas significatifs

# Build the model
model <- lm(Growth ~ poly(Debt, 5, raw = TRUE), data = RR.train.data)
# Make predictions
predictions <- model %>% predict(RR.test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, RR.test.data$Growth),
  R2 = R2(predictions, RR.test.data$Growth)
)

ggplot(RR.train.data, aes(x = Debt, y = Growth) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x, 5, raw = TRUE))

##### Régression logarithmique
##### 

# Build the model
model <- lm(Growth ~ log(Debt), data = RR.train.data)
# Make predictions
predictions <- model %>% predict(RR.test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, RR.test.data$Growth),
  R2 = R2(predictions, RR.test.data$Growth)
)

ggplot(RR.train.data, aes(x = Debt, y = Growth) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ log(x))


##### Régression en spline
#####
#You need to specify two parameters: the degree of the polynomial and the location of the knots. In our example, we’ll place the knots at the lower quartile, the median quartile, and the upper quartile:

# Build the model (cubic splines i.e degree 3)
knots <- quantile(RR.train.data$Debt, p = c(0.25, 0.5, 0.75))
model <- lm (Growth ~ bs(Debt, knots = knots), data = RR.train.data)
# Make predictions
predictions <- model %>% predict(RR.test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, RR.test.data$Growth),
  R2 = R2(predictions, RR.test.data$Growth)
)

#Note that, the coefficients for a spline term are not interpretable.

ggplot(RR.train.data, aes(x = Debt, y = Growth) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ splines::bs(x, df = 3))

##### Generalized additive models
#####

# Build the model
model <- gam(Growth ~ s(Debt), data = RR.train.data)
#The term s(lstat) tells the gam() function to find the “best” knots for a spline term.

# Make predictions
predictions <- model %>% predict(RR.test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, RR.test.data$Growth),
  R2 = R2(predictions, RR.test.data$Growth)
)

ggplot(RR.train.data, aes(x = Debt, y = Growth) ) +
  geom_point() +
  stat_smooth(method = gam, formula = y ~ s(x))


#################################
### Tutorial : ##################
#http://www.sthda.com/english/articles/40-regression-analysis/162-nonlinear-regression-essentials-in-r-polynomial-and-spline-regression-models/
#################################

# Load the data
#We’ll use the Boston data set [in MASS package], introduced in Chapter @ref(regression-analysis), for predicting the median house value (mdev), in Boston Suburbs, based on the predictor variable lstat (percentage of lower status of the population).

data("Boston", package = "MASS")

#We’ll randomly split the data into training set (80% for building a predictive model) and test set (20% for evaluating the model). Make sure to set seed for reproducibility.
# Split the data into training and test set
set.seed(123)
training.samples <- Boston$medv %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- Boston[training.samples, ]
test.data <- Boston[-training.samples, ]

ggplot(train.data, aes(x = lstat, y = medv) ) +
  geom_point() +
  stat_smooth()

##### régression linéaire
##### The standard linear regression model equation can be written as medv = b0 + b1*lstat.

# Build the model
model <- lm(medv ~ lstat, data = train.data)

# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)

ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ x)

##### Régression polynomial
##### The polynomial regression adds polynomial or quadratic terms to the regression equation as follow: 
##### medv=b0+b1∗lstat+b2∗lstat2
## Rang 2 In R, to create a predictor x^2 you should use the function I(), as follow: I(x^2). This raise x to the power 2.
lm(medv ~ lstat + I(lstat^2), data = train.data)
lm(medv ~ poly(lstat, 2, raw = TRUE), data = train.data)  #formule alternative

#The output contains two coefficients associated with lstat : one for the linear term (lstat^1) and one for the quadratic term (lstat^2).

## Rang 6
lm(medv ~ poly(lstat, 6, raw = TRUE), data = train.data) %>%
  summary() ## On voit que les termes de rang sup à 5 ne sont pas significatifs

# Build the model
model <- lm(medv ~ poly(lstat, 5, raw = TRUE), data = train.data)
# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)

ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x, 5, raw = TRUE))

##### Régression logarithmique
##### When you have a non-linear relationship, you can also try a logarithm transformation of the predictor variables:

# Build the model
model <- lm(medv ~ log(lstat), data = train.data)
# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)

ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ log(x))


##### Régression en spline
#####
#Polynomial regression only captures a certain amount of curvature in a nonlinear relationship. An alternative, and often superior, approach to modeling nonlinear relationships is to use splines (P. Bruce and Bruce 2017).
#Splines provide a way to smoothly interpolate between fixed points, called knots. Polynomial regression is computed between knots. In other words, splines are series of polynomial segments strung together, joining at knots (P. Bruce and Bruce 2017).

#You need to specify two parameters: the degree of the polynomial and the location of the knots. In our example, we’ll place the knots at the lower quartile, the median quartile, and the upper quartile:

# Build the model (cubic splines i.e degree 3)
knots <- quantile(train.data$lstat, p = c(0.25, 0.5, 0.75))
model <- lm (medv ~ bs(lstat, knots = knots), data = train.data)
# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)

#Note that, the coefficients for a spline term are not interpretable.

ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ splines::bs(x, df = 3))

##### Generalized additive models
#####
#Once you have detected a non-linear relationship in your data, the polynomial terms may not be flexible enough to capture the relationship, and spline terms require specifying the knots.
#Generalized additive models, or GAM, are a technique to automatically fit a spline regression. This can be done using the mgcv R package:

# Build the model
model <- gam(medv ~ s(lstat), data = train.data)
#The term s(lstat) tells the gam() function to find the “best” knots for a spline term.

# Make predictions
predictions <- model %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$medv),
  R2 = R2(predictions, test.data$medv)
)

ggplot(train.data, aes(lstat, medv) ) +
  geom_point() +
  stat_smooth(method = gam, formula = y ~ s(x))


###########
### END ###
###########
