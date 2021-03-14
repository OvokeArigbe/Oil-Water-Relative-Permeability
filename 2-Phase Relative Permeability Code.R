# Import Packages
install.packages("mass")
install.packages("h2o")
install.packages("gridExtra")
install.packages("MXNetR")

library(MASS)
library(h2o)
library(gridExtra)

# Data import for Krw anmd Kro
set.seed(123)

relperm_orig=read.table("C:/Users/ovkoe/Documents/relPermeability/relpermpredkrw.csv",header=T, sep=",")
relperm_krw=as.data.frame(relperm_orig)
plot(relperm_krw)

relperm_orig=read.table("C:/Users/ovkoe/Documents/relPermeability/relpermpredkro.csv",header=T, sep=",")
relperm_kro=as.data.frame(relperm_orig)
plot(relperm_kro)

# h2o initialisation
h2o.init(ip = "localhost",port = 54321)

# kro - oil relative permeability
# krw - water relative permeability

# Create training and testing datasets for kro
Indo <- sample(1:nrow(relPerm_kro), 107)
trainkro <- relperm_kro[Indo,]
testkro <- relperm_kro[-Indo,]

# Create training and testing datasets for krw
Indo <- sample(1:nrow(relPerm_krw), 107)
trainkrw <- relperm_krw[Indo,]
testkrw <- relperm_krw[-Indo,]

# Organise the formular names for kro
allVars <- colnames(relperm_kro)
predictorsvars <- allVars[!allVars%in%"Kro"]
predictorsvars <- paste(predictorsvars,collapse = "+")
form <- as.formula(paste("kro~", predictorsvars, collapse = "+"))

# Organise the formular names for krw
allVars <- colnames(relperm_krw)
predictorsvars <- allVars[!allVars%in%"Krw"]
predictorsvars <- paste(predictorsvars,collapse = "+")
form <- as.formula(paste("kro~", predictorsvars, collapse = "+"))

# Scaling for krw and kro
MaxValue <- apply(relperm_krw,2,max)
MinValue <- apply(relperm_krw,2,min)
traindlw_df <- as.data.frame(scale(traindlw,center = MinValue,scale = MaxValue-MinValue))
traindlw_h2o <- as.h2o(traindlw_df,destination_frame = "traindlw_h2o")

Maxvalue <- apply(relperm_kro,2,max)
Minvalue <- apply(relperm_kro,2,min)
traindlo_df <- as.data.frame(scale(traindlo,center = Minvalue,scale = Maxvalue-Minvalue))
traindlo_h2o <- as.h2o(traindlo_df,destination_frame = "traindlo_h2o")

# convert to data frame
testdlw_df <- as.data.frame(scale(testdlw,center = MinValue,scale = MaxValue-MinValue))
testdlw_h2o <- as.h2o(testdlw_df,destination_frame = "testdlw_h2o")
valDlw_df <- as.data.frame(scale(valDlw,center = MinValue,scale = MaxValue-MinValue))
valDlw_h2o <- as.h2o(valDlw_df,destination_frame = "valDlw_h2o")

testdlo_df <- as.data.frame(scale(testdlo,center = Minvalue,scale = Maxvalue-Minvalue))
testdlo_h2o <- as.h2o(testdlo_df,destination_frame = "testdlo_h2o")
valdlo_df <- as.data.frame(scale(valDlo,center = Minvalue,scale = Maxvalue-Minvalue))
valdlo_h2o <- as.h2o(valdlo_df,destination_frame = "valdlo_h2o")


# Defining x and y
yw = "krw"
xw = setdiff(colnames(traindlw_h2o),yw)

yo = "kro"
xo = setdiff(colnames(traindlo_h2o),yo)


# Hyper_parameter tuning with grid search
hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout"),
  hidden=list(c(20,20,20,20,20),c(50,50,50,50,50),c(30,30,30,30),c(200,200,200,200,200)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varied less than 1%)
help("h2o.grid")
search_criteria = list(strategy="RandomDiscrete", stopping_rounds=10,  seed=1234567, stopping_metric="AUTO", stopping_tolerance=1e-3)

# Grid search
dl_random_grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid",
  training_frame=traindlw_h2o,
  validation_frame=valDlw_h2o,
  x=xw,
  y=yw,
  epochs=10,
  stopping_tolerance=1e-2, ## stop when logloss does not improve by >=1% for two scoring events
  # score_validatiion_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025, ## dont score 2.5% f the score time
  max_w2=10, ## can help improve stability for rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)
summary(dl_random_grid)

library(jsonlite)
grid <- h2o.getGrid("dl_random_grid",sort_by = "err",decreasing = FALSE)

# Model fitting
# Regularization with l1 and l2 to further solve the problem of overfitting
modeldlw = h2o.deeplearning(x=xw,
                            y=yw,
                            seed = 1234,
                            training_frame = as.h2o(traindlw_df),
                            nfolds = 5,
                            standardize = FALSE,  # since it has already been normalized
                            stopping_rounds = 5,
                            epochs = 400,
                            overwrite_with_best_model = TRUE,
                            ignore_const_cols = FALSE,
                            activation = "Rectifier",
                            hidden = c(100,100),
                            l2=6e-5,
                            diagnostics = TRUE,
                            variable_importances = TRUE,
                            loss = "Automatic",
                            distribution = "AUTO",
                            stopping_metric = "RMSE")

help("h2o.deeplearning")
plot(as.data.frame(h2o.varimp(modeldlw))) # variable importance of the model

# modeldlw = h2o.deeplearning(x=xw, y=yw, training_frame = Traindlw_dfi, hidden = c(200,200), epochs = 5)

# modeling oil relative permeability

modeldlo = h2o.deeplearning(x=xo,
                            y=yo,
                            seed = 1234,
                            training_frame = as.h2o(traindlo_df),
                            nfolds = 5,
                            # standardize =FALSE
                            stopping_rounds = 5,
                            epochs = 400,
                            overwrite_with_best_model = TRUE,
                            ignore_const_cols=FALSE,
                            activation = "Rectifier",
                            diagnostics = TRUE,
                            variable_importances = TRUE,
                            hidden = c(100,100,100,100),
                            l2=6e-5,
                            loss = "Automatic",
                            distribution = "AUTO",
                            stopping_metric = "RMSE")

as.data.frame(h2o.varimp(modeldlo))
# predictions
predictiondlw = as.data.frame(predict(modeldlw,as.h2o(testdlw_df)))
predictiondlw_v = as.data.frame(predict(modeldlw,as.h2o(valDlw_df)))
g=cbind(predictiondlw,testdlw_df$krw)
View(g)
h=cbind(predictiondlw_v,valDlw_df$krw)
View(h)
h2o.varimp_plot(modeldlo)


predictiondlo = as.data.frame(predict(modeldlo,as.h2o(testdlo_df)))
predictiondlo_v = as.data.frame(predict(modeldlo,as.h2o(valdlo_df)))
i=cbind(predictiondlo,testdlo_df$kro)
View(i)
j=cbind(predictiondlo_v,valdlo_df$kro)
View(j)
predictiondlw$predict
h2o.sensitivity(modeldlw)

# plotting predicted values vs actual values
par(mfrow=c(2,2))

plot(testdlw_df$krw,predictiondlw$predict,col = 'black',main = 'dnn validation krw',
     pch=1,cex=1,type = "p",xlab = "Actual",ylab = "Predicted")

plot(valdlw_dfi$Krw,predictiondlw_v$predict,col = 'black',main = 'dnn test Krw',
     pch=1,cex=1,type = "p",xlab = "Actual",ylab = "Predicted")

plot(Testdlo_dfi$Kro,predictiondlo$predict,col = 'black',main = 'dnn validation Kro',
     pch=1,cex=1,type = "p",xlab = "actual",ylab = "predicted")

plot(valdlo_dfi$Kro,predictiondlo_v$predict,col = 'black',main = 'dnn test Kro',
     pch=1,cex=1,type = "p",xlab = "actual",ylab = "predicted")

# MSE determination
MSEdlw <- sum((predictiondlw$predict-Testdlw_dfi$Krw)^2)/nrow(Testdlw_dfi)
MSEdlw_v<- sum((predictiondlw_v$predict-valdlw_dfi$Krw)^2)/nrow(valdlw_dfi)

MSEdlo <- sum((predictiondlo$predict-Testdlo_dfi$Kro)^2)/nrow(Testdlo_dfi)
MSEdlo_v<- sum((predictiondlo_v$predict-valdlo_dfi$Kro)^2)/nrow(valdlo_dfi)

