library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)  

data <- read.csv("crimes.csv")
colnames(data)

# Find multicollinearity by applying VIF
target <- 'ViolentCrimesPerPop'
features <- data %>% select(-ViolentCrimesPerPop) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = data)

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = data)

glm %>% summary()

# VIF (Variance Inflation Factor) ----
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = data)
}

glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 

data <- data %>% select(ViolentCrimesPerPop,features)


# Standardize features ----
data %>% glimpse()

data[,-1] <- data[,-1] %>% scale() %>% as.data.frame()

#  Split data into train and test sets using seed=123
h2o.init()
h2o_data <- data %>% as.h2o()
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'ViolentCrimesPerPop'
features <- data %>% select(-ViolentCrimesPerPop) %>% names()

# Build linear regression model. p value of variables should be max 0.05
model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict

test_set <- test %>% as.data.frame()
residuals = test_set$ViolentCrimesPerPop - y_pred$predict

# Calculate RMSE and Adjusted R-squared
RMSE = sqrt(mean(residuals^2))
y_test_mean = mean(test_set$ViolentCrimesPerPop)

tss = sum((test_set$ViolentCrimesPerPop - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)

# Check overfitting ----
y_pred_train <- model %>% h2o.predict(newdata = train) %>% as.data.frame()

train_set <- train %>% as.data.frame()
residuals = train_set$ViolentCrimesPerPop - y_pred_train$predict

RMSE_train = sqrt(mean(residuals^2))
y_train_mean = mean(train_set$ViolentCrimesPerPop)

tss = sum((train_set$ViolentCrimesPerPop - y_train_mean)^2)
rss = sum(residuals^2)

R2_train = 1 - (rss/tss); R2_train

n <- train_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2_train = 1-(1-R2_train)*((n-1)/(n-k-1))