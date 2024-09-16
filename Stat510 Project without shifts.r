
library(pROC)
library(psych)
library(purrr)
library(readr)
library(tidyverse)
library(GGally)
library(corrplot)
library(psych)
library(pscl)
library(car)
library(MASS)
library(readr)
library(olsrr)
library(caret)
library(ROCit)
library(lmtest)
library(caret)
library(glmnet)
library(cutpointr)
library(pander)
library(tree)
library(broom)



##### Cleaning and creating the data set

df = as_tibble(read.csv("### Baseball Savant pitch-by-pitch data ###"))

#Subseting the data to only batted balls put in play
df = subset(df, df$description == 'hit_into_play')

#Getting only variables that have to do with a batted ball(launch angle, exit speed, etc.)
df = df[,c(9,10,38,39,53,54,55)]

#Removing the description variable
df = df[,-2]

#Adding a factor variable to the data of the true outcome of the batted ball where 1=hit and 0=not a hit.
df = mutate(df, hit.or.out= as.factor(ifelse(df$events %in% c('single', 'double', 'triple', 'home_run') , 1,0)))


#Remove N/A values
df = na.omit(df)


#End up with this size data set
#dim(df)
#str(df)



##### Splitting the data into test and training sets
n = nrow(df)
prop = .8
set.seed(123)
train_id = sample(1:n, size = n*prop, replace = FALSE)
test_id = (1:n)[-which(1:n %in% train_id)]
train_set = df[train_id, ]
test_set = df[test_id, ]

train_set= train_set[,-1]
test_set= test_set[,-1]


## Initial logistic regression model with all variables. 

#3.)

m1 = glm(hit.or.out~ hc_x+hc_y+hit_distance_sc+launch_speed+launch_angle, train_set, family="binomial")

summary(m1)

#From initial model it does not look like the defensive alignments have an effect.


#Attempting to predict the train set
pred = predict(m1, train_set, type="response")
#optimal = optimalCutoff(train_set$hit.or.out, pred)[1]
pred = as.factor(ifelse(predict(m1, train_set, type="response")>.5,1,0))
tb = table(pred = pred, truth = train_set$hit.or.out)
tb
(tb[1,1] + tb[2,2])/sum(tb)





## Checking for multicollinearity

#a.)
cor(df[,c(2:6)])
corPlot(df[,c(2:6)])
#There looks to be multicollineraity between hit_distance_sc and hc_y

#The VIF matrix
solve(cor(df[,c(2:6)]))
#Have to decide to get rid of one or the other or create an interaction term. Most likely drop hit_distance_sc
vif(m1)
varImp(m1)


# I originally took every single variable that had to do with batted balls from the data set 
# to make sure I was going to have enough to satisfy the requirements of the data. 
# Even before I got to this step I realized this would have led to a lot of 
# multicollinearity because a lot of the variables were related to one another. 
# For example the launch_speed_angle was based off of the two variables. 
# I still though that hc_y and hit_distance might provide a little different values 
# and influence on the model but the correlation plots say otherwise.  


## Decided to drop hit_distance_sc due to multicollinearity. This is glm without hit_distcance_sc.

f = glm(formula = hit.or.out ~ hc_x + hc_y  + launch_speed + launch_angle, 
        family = "binomial", data = train_set)
summary(f)

vif(f)
varImp(f)

residualPlots(f)

#Diagnostics of this model

summary(f)

pi_hat<-f$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve

ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:

ROCit_obj$AUC





## Checking if complex model is appropriate.(Polynomial and interaction terms)


#Check for inclusion of polynomial terms for hc_x
j=glm(hit.or.out~poly(hc_x,4),train_set,family=binomial)
summary(j)

hc_x_scale<-scale(train_set$hc_x,scale=F)
hc_x_2<-hc_x_scale^2
train_set<-cbind(train_set,hc_x_scale,hc_x_2)

k = glm(hit.or.out~hc_x+hc_x_2+hc_y+launch_speed+launch_angle, train_set, family=binomial)

summary(k)

#Looks like I could add hc_x^2 into the model. 




############### Doesnt look like either of these polynomial terms need to be added ##########
#Check for inclusion of polynomial terms for launch_speed
j=glm(hit.or.out~poly(launch_speed,4),train_set,family=binomial)
summary(j)


############## Dont think I should actually do this ################
#Check for inclusion of polynomial terms for launch_angle
j=glm(hit.or.out~poly(launch_angle,4),train_set,family=binomial)
summary(j)





## Check for interaction term of hc_y*hit_distance_sc and hc_y*launch_angle
m5 = glm(hit.or.out~ hc_x+hc_y*hit_distance_sc+launch_speed+launch_angle, train_set, family="binomial")
summary(m5)

m7 = glm(hit.or.out~ hc_x+hc_y*launch_angle+launch_speed, train_set, family="binomial")
summary(m7)
varImp(m7)

m8 = glm(hit.or.out~ hc_y*launch_angle, train_set, family="binomial")
summary(m8)

########### The interaction term of hc_y*launch_angle looks to be very significant. #############




## GLM with complex terms

r = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family=binomial)
summary(r)


pi_hat<-r$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC






## Between checking if a complex model is appropriate and running the lasso model.
# Looks as though I should attempt to see what a model of hc_x, hc_x_2, hc_y*launch_angle, launch_speed.

####### Make sure to scale the hc_x term in the test_set before analysis #####




## Residual Diagnostics


####### Don't worry about heteroskadasticity for logistic regression ########

#Check for outliers
residualPlots(r)

marginalModelPlots(r)

influenceIndexPlot(r)

influencePlot(r)

train_set[c(82,129,216,316),]

r_update = update(r, subset=-c(82,129,216,316))

compareCoefs(r,r_update)

#When fitting the data without the outiers the model doesnt change significanty. 









## Model Selection 1

mj = glm(hit.or.out~ hc_y*launch_angle, train_set, family="binomial")

# Logistic diagnostics

summary(mj)

pi_hat<-mj$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC




## Model Selection 2

p = glm(hit.or.out~ hc_x+hc_x_2, train_set, family="binomial")

# Logistic diagnostics

summary(p)

pi_hat<-p$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC




## Model Selection 3

mk = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle, train_set, family="binomial")

# Logistic diagnostics

summary(mk)

pi_hat<-mk$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC




## Model Selection 4

ml = glm(hit.or.out~ hc_x+hc_x_2+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC







## Model Selection 5
ml = glm(hit.or.out~ hc_y*launch_angle+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC







## Model Selection 6

ml = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC






## Model Selection 7

ml = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC





## Model Selection 8

ml = glm(hit.or.out~ hc_x+hc_x_2+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC





## Model Selection 9

ml = glm(hit.or.out~ hc_y*launch_angle+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC




## Model Selection 10

ml = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family="binomial")

# Logistic diagnostics

summary(ml)

pi_hat<-ml$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC




## Stepwise regression

############## Stepwise regression #################
#This is stepwise regression before implementing any of the previous work of complex & interacton models.
mj = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family="binomial")


#backward stepwise regression using step() function
final<-step(mj)
summary(final)

#define model with only the intercept
intercept_only <- glm(hit.or.out ~ 1, train_set, family= binomial)


#perform forward stepwise regression
forward <- step(intercept_only, direction='forward', scope=formula(mj), trace=0)
summary(forward)

#perform both direction stepwise regression
both <- step(intercept_only, direction='both', scope=formula(mj), trace=0)
summary(both)





## Trying to Predict test set


########### Still predicting the training set #######################

p = glm(hit.or.out~ hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family="binomial")


#Attempting to predict the train set
pred = predict(p, train_set, type="response")
pred = as.factor(ifelse(predict(p, train_set, type="response")>.5,1,0))
tb = table(pred = pred, truth = train_set$hit.or.out)
tb
(tb[1,1] + tb[2,2])/sum(tb)

# Logistic diagnostics

summary(p)

pi_hat<-p$fitted.values

cp_yhat<-as.factor(1*(pi_hat>.5))  #Class predictions
yf<-as.factor(train_set$hit.or.out)  #Actual data


#Confusion matrix
confusionMatrix(cp_yhat,yf,positive="1")


#ROC curve
ROCit_obj <- rocit(score=as.numeric(cp_yhat),class=train_set$hit.or.out)
plot(ROCit_obj)

#To obtain the AUC and optimal cutpoint:
ROCit_obj$AUC


##################### Predicting Test Set ##########################



###################### Remember to scale or do whatever is needed to the test set to include the polynomial term of hc_x ######################

#Check for inclusion of polynomial terms for hc_x
j=glm(hit.or.out~poly(hc_x,4),test_set,family=binomial)
summary(j)

hc_x_scale<-scale(test_set$hc_x,scale=F)
hc_x_2<-hc_x_scale^2
test_set<-cbind(test_set,hc_x_scale,hc_x_2)

k = glm(hit.or.out~hc_x+hc_x_2+hc_y*launch_angle+launch_speed, train_set, family=binomial)

summary(k)


#Attempting to predict the test set. 
pred = predict(k, test_set, type="response")
pred = as.factor(ifelse(predict(k, test_set, type="response")>.5,1,0))
tb = table(pred = pred, truth = test_set$hit.or.out)
tb
(tb[1,1] + tb[2,2])/sum(tb)
