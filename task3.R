rm(list=ls())
library(ggplot2)
library(data.table)
library(MASS)
library(GGally)
library(Metrics)

# read data ---------------------------------------------------------------
dt <- fread('df.csv')
dt$V1 <- NULL

colnames(dt)

#look at the data 
dt <- dt[order(Date)]
summary(dt)

#EDA

ggplot(dt)+geom_point(aes(distance_fire_stations,distance_rivers,color = ignition))

#ANOVA
str(dt)

#vegetation_class fixed to 
uniqueN(dt$vegetation_class);unique(dt$vegetation_class)

#fix error

dt[vegetation_class == 'herbaceous_vegetation']

dt[vegetation_class == 'Forestt']$vegetation_class <- 'forest'
dt[vegetation_class == '$herb$aceous_vegetation']$vegetation_class <- 'herbaceous_vegetation'

dt[vegetation_class == '']

veg_calss <-  dt[,'vegetation_class']

for (i in unique(veg_calss$vegetation_class)){
  print(i)
  veg_calss[,(i):=0]
  veg_calss[,(i):= as.integer(vegetation_class == i)]
  setnames(veg_calss,i,paste0(i,'_ign'))
}

summary(veg_calss)

dt <- cbind(dt[,-'vegetation_class'],veg_calss[,-1])
setnames(dt,'_ign','empty_ign')

#check collinearity 

cor(na.omit(dt[,-'Date']))

#delete some columns which do not have any data
unique(dt$forest_deciduous_needle)
dt <- dt[,-'forest_deciduous_needle']

unique(dt$forest_mixed)
dt <- dt[,-'forest_mixed']

unique(dt$moss_lichen)
dt <- dt[,-'moss_lichen']

#Now check the collinearities
# cor(na.omit(dt[,-'Date']))
# 
# dt_na0 <- na.omit(dt[,-'Date'])
# dt_na0[ignition==1]
# 
# 2288/15199
# 
# nrow(dt[ignition==1])/nrow(dt)
# nrow(dt[ignition==1])

#lets check how nas exist per predictor
summary(dt)

#yearly_avg_temp:6831 ,forest_evergreen_broad
#forest_evergreen_needle, forest_unknown,herbaceous_vegetation,shrubland        
#sprarse_vegetation,urban,waterwetland:5, 

dt <- dt[!is.na(distance_fire_stations)]

#yearly_avg_temp:6831
#dt[,year:= year(Date)]

range(dt$yearly_avg_temp,na.rm = T)
summary(dt)


# train test dataset ------------------------------------------------------
dt <- na.omit(dt)

#stratified
set.seed(100)
p <- 0.75
idx <- dt[, .I[sample(.N, ceiling(p*.N))], by = ignition]$V1

train <- dt[,-'Date'][idx]
test  <- dt[,-'Date'][-idx]

nrow(train[ignition ==1])/nrow(train)
nrow(test[ignition ==1])/nrow(test)

#think if you want to delete yearly_avg_temp

model_anova <- aov(ignition~ ., train)
summary(model_anova)

linear_model <- lm(ignition~ ., na.omit(dt[,-'Date']))
summary(linear_model)

linear_ignition <- round(as.numeric(predict(linear_model,test[,-'ignition'], type = "response")))
linear_ignition
rmse(linear_ignition,test$ignition)

#logit GLM

y_name <- "ignition"

# Keep your data in a data.frame named df
predictors <- setdiff(names(train), y_name)

# Separate numeric vs non-numeric (only square the numeric ones)
all_vars  <- predictors

# Build: y ~ (all_vars)^2 + I(x1^2) + I(x2^2) + ...
f_interact <- paste(all_vars, collapse = " + ")
f_squares  <- if (length(predictors)) paste(sprintf("I(%s^2)", predictors), collapse = " + ") else "1"

f_str <- sprintf("%s ~ (%s)^2 + %s", y_name, f_interact, f_squares)
form  <- as.formula(f_str)

fit_logit <- glm(ignition~ . , data = train, 
                 family = binomial(link = "logit"))
summary(fit_logit)
plot(fit_logit)

glm_ignition <- round(as.numeric(predict(fit_logit,test[,-'ignition'], type = "response")))
glm_ignition

rmse(glm_ignition,test$ignition)

table(Predicted = glm_ignition, Actual = test$ignition)

library(caret)

confusionMatrix(
  factor(glm_ignition, levels = c(0,1)),
  factor(test$ignition, levels = c(0,1))
)

#random forest
library(randomForest)

train$ignition <- factor(train$ignition, levels = c(0,1))
test$ignition  <- factor(test$ignition,  levels = c(0,1))

rf_model <- randomForest(ignition ~ ., 
                         data = train, 
                         ntree = 500, 
             importance = TRUE)

#plot(rf_model)  # eyeball the plateau

rf_ignition <- round(as.numeric( predict(rf_model, test[,-'ignition'], type = "prob")[, "1"]))
rf_ignition
rmse(rf_ignition,as.numeric(test$ignition))
table(Predicted = rf_ignition, Actual = test$ignition)

confusionMatrix(
  factor(rf_ignition, levels = c(0,1)),
  factor(test$ignition, levels = c(0,1))
)


#lasso and ridge regression 
library(glmnet)

cvfit <- cv.glmnet(model.matrix(ignition ~ ., data = train)[, -1] , 
                   train$ignition, 
                   family = "binomial",      # logistic regression
                   alpha =0,                # 1 = LASSO, 0 = Ridge, between = Elastic Net
                   nfolds = 10)

#plot(cvfit)
p1 <- predict(cvfit, newx = model.matrix(ignition ~ ., data = test)[, -1], 
              s = "lambda.min", type = "response")
cvfit_predict <- round(as.numeric(p1))
rmse(cvfit_predict,as.numeric(test$ignition))
table(Predicted = cvfit_predict, Actual = test$ignition)

confusionMatrix(
  factor(cvfit_predict, levels = c(0,1)),
  factor(test$ignition, levels = c(0,1))
)


#xboost
library(xgboost)
str(train)
set.seed(100)
p <- 0.75
idx <- dt[, .I[sample(.N, ceiling(p*.N))], by = ignition]$V1

train <- dt[,-'Date'][idx]
test  <- dt[,-'Date'][-idx]

# --- Prep matrices ---
X_train <- model.matrix(ignition ~ . , data = train)[, -1]
y_train <- train$ignition

X_test  <- model.matrix(ignition ~ . , data = test)[, -1]
y_test  <- test$ignition

dtrain <- xgb.DMatrix(data = X_train, label = as.numeric(y_train))
dtest  <- xgb.DMatrix(data = X_test,  label = as.numeric(y_test))

# --- Train ---
params <- list(
  objective = "binary:logistic",  # logistic output
  eval_metric = "auc",            # AUC metric
  eta = 0.1,                      # learning rate
  max_depth = 6,                  # tree depth
  subsample = 0.8,                # row sampling
  colsample_bytree = 0.8          # feature sampling
)

set.seed(42)
fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 20,
  print_every_n = 50
)

# --- Predict ---
p1 <- predict(fit_xgb, dtest)   # probabilities in [0,1]
pred_class <- ifelse(p1 >= 0.5, 1, 0)

table(truth = y_test, pred = pred_class)


confusionMatrix(
  factor(pred_class, levels = c(0,1)),
  factor(test$ignition, levels = c(0,1))
)

# --- Feature importance ---
importance <- xgb.importance(model = fit_xgb)
head(importance)
xgb.plot.importance(importance)

# GAM
library(mgcv)

num_vars <- names(train)[sapply(train, is.numeric)]
num_vars <- setdiff(num_vars, "ignition")
fac_vars <- setdiff(names(train), c(num_vars, "ignition"))

gam_form <- as.formula(
  paste(
    "ignition ~",
    paste(c(paste0("s(", num_vars, ")"), fac_vars), collapse = " + ")
  )
)

gam_fit <- gam(
  gam_form,
  data = train,
  family = binomial(link = "logit"),
  method = "REML"          # good default for smoothing selection
)


summary(gam_fit)
plot(gam_fit, pages = 1, se = TRUE)   # visualize smooth terms

p_gam <- predict(gam_fit, newdata = test, type = "response")
pred_class <- ifelse(p_gam >= 0.5, 1, 0)

confusionMatrix(
  factor(pred_class, levels = c(0,1)),
  factor(test$ignition, levels = c(0,1))
)

#auc, brier, accuracy 