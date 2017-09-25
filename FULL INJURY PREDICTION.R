################################################ FULL INJURY PREDICTION ################################################

setwd('C:\\Users\\Ignacio S-P\\Desktop\\R Data processing\\Houston Texans\\')
my_files <- list.files(pattern = "\\.csv$")
my_data <- lapply(my_files, read.csv, stringsAsFactors = FALSE)
Tableinstructions = read.csv("Tableinstructions.csv", sep = ",", stringsAsFactors = FALSE)
## Number of tables to analyze
my_data = my_data[6]

## Creating the results table (rows for every table, method and iteration)
Methods = c("RF", "RF2", "RF3", "GLM", "SVM", "LDA", "GBM", "RIDGE", "ACGLM")
Niterations = 50
Ntables = 1
Nmethods = length(Methods)
Runs = c(sapply(1:Niterations, function(x) rep(x, times = Ntables*Nmethods)))
Runs.Method = c(sapply(Methods, function(x) rep(x, times = Ntables)))
Tableinstructions$Name = paste(Tableinstructions[,2],
                               Tableinstructions[,3],
                               Tableinstructions[,4],
                               Tableinstructions[,5])
Results.TB = data.frame(Runs, Runs.Method, Tableinstructions, stringsAsFactors = FALSE)
Results.TB$LogLoss = NA
Results.TB$Auc = NA

## Runs the loop with the input table (row per run-table combination, all methods in each)
setwd('C:\\Users\\Ignacio S-P\\Desktop\\R Data processing\\Houston Texans\\ROCiteration\\')

Run.TB = data.frame(c(sapply(1:Niterations, function(x) rep(x, times = Ntables))), Tableinstructions$Name)
colnames(Run.TB) = c("Run", "Table")
Run.TB$TablePosition = as.numeric(as.factor(Run.TB$Table))

output <- file("Results_TB.txt", "w")
write.table(Results.TB, file = output)
close(output)

################### EVALUATION: LOGLOSS FUNCTION  ####

LogLoss = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
  }

################### PREANALYSIS: VARIABLE REMOVING  ####

RemoveColumns.list = c("X", "PlayerName", "RoundDate", "Date", "Latest.Position", "Training.Load.7.day.Total", "Training.Load.7.day.SD,
                       Training.Load.28.day.Total", "Acute.7.day.Average", "Chronic.28.day.average", "NextDate", "Key", "TypeInjury",
                       "WeightEWMA.0.05", "WeightEWMA.0.5", "WeightEWMA.0.95")

for (i in 1:length(my_data)) {
  for (z in RemoveColumns.list) {
    if (all(z %in% colnames(my_data[[i]]))) {
      my_data[[i]] = my_data[[i]][, -which(colnames(my_data[[i]]) == z)]
    }
  }
}

colnames(my_data[[1]])

################### PREANALYSIS: PARAMETER TUNNING  ####

F.MethTuning = function (i) {
## GBM ####
  GBM.Train = my_data[[i]]
  GBM.Train$Injury = as.factor(GBM.Train$Injury)
  trainTask <- makeClassifTask(data = GBM.Train, target = "Injury", positive = "1")
  #load GBM
  getParamSet("classif.gbm")
  g.gbm <- makeLearner("classif.gbm", predict.type = "prob")
  #specify tuning method
  set_cv <- makeResampleDesc("Bootstrap",iters = 30L, stratify = TRUE)
  rancontrol <- makeTuneControlRandom(maxit = 20L)
  gbm_par<- makeParamSet(
    makeDiscreteParam("distribution", values = "bernoulli"),
    makeIntegerParam("n.trees", lower = 100, upper = 2000),
    makeIntegerParam("interaction.depth", lower = 2, upper = 30),
    makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
    makeNumericParam("shrinkage",lower = 0.01, upper = 1)
  )
  tune_gbm <- tuneParams(learner = g.gbm, task = trainTask, resampling = set_cv, measures = auc,
                         par.set = gbm_par, control = rancontrol)
  final_gbm <- setHyperPars(learner = g.gbm, par.vals = tune_gbm$x)
  
## RF ####
  rf <- makeLearner("classif.randomForest", predict.type = "prob")
  set_cv <- makeResampleDesc("Bootstrap",iters = 30L, stratify = TRUE)
  rancontrol <- makeTuneControlRandom(maxit = 20L)
  rf_par <- makeParamSet(
    makeIntegerParam("ntree",lower = 100, upper = 1200),
    makeIntegerParam("mtry", lower = 3, upper = 10),
    makeIntegerParam("nodesize", lower = 10, upper = 50)
  )
  rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = trainTask, par.set = rf_par, control = rancontrol, measures = auc)
  final_rf <- setHyperPars(rf, par.vals = rf_tune$x)
  
## SVM ####  
  ksvm <- makeLearner("classif.ksvm", predict.type = "prob")
  set_cv <- makeResampleDesc("Bootstrap",iters = 30L, stratify = TRUE)
  rancontrol <- makeTuneControlRandom(maxit = 20L)
  svm_par <- makeParamSet(
    makeNumericParam("C", lower = 0, upper = 1), #cost parameters
    makeNumericParam("sigma",lower = 0, upper = 1) #RBF Kernel Parameter
  )
  svm_tune <- tuneParams(ksvm, task = trainTask, resampling = set_cv, par.set = svm_par, control = rancontrol,measures = auc)
  final_svm <- setHyperPars(ksvm, par.vals = svm_tune$x)
  
## RF2 ####
  rf2 <- makeLearner("classif.cforest", predict.type = "prob")
  set_cv <- makeResampleDesc("Bootstrap",iters = 30L, stratify = TRUE)
  rancontrol <- makeTuneControlRandom(maxit = 20L)
  rf2_par <- makeParamSet(
    makeIntegerParam("mtry", lower = 3, upper = 12),
    makeIntegerParam("ntree", lower = 100, upper = 1200),
    makeIntegerParam("maxdepth", lower = 2, upper = 30),
    makeIntegerParam("minsplit", lower = 10, upper = 60)
  )
  rf2_tune <- tuneParams(learner = rf2, resampling = set_cv, task = trainTask, par.set = rf2_par, control = rancontrol, measures = auc)
  #using hyperparameters for modeling
  final_rf2 <- setHyperPars(rf2, par.vals = rf2_tune$x)
  return(list(final_rf, final_rf2, final_svm, final_gbm))
}

final_all <- sapply(1:length(my_data), FUN = F.MethTuning)

################### LOOP START ####

F.Methiteration = function (x) {
  CurrentRun = x$Run
  CurrentTable = x$Table
  Datx = my_data[[x$TablePosition]]
  
################### TRAIN TEST ####
Dat1 = Datx[Datx$Injury == 1,]
Dat2 = Datx[Datx$Injury == 0,]
Train1.vec = sample(1:nrow(Dat1), size = round(nrow(Dat1)*0.66))
Train1 = Dat1[Train1.vec,]
Test1 = Dat1[-Train1.vec,]
Train2.vec = sample(1:nrow(Dat2), size = round(nrow(Dat2)*0.66))
Train2 = Dat2[Train2.vec,]
Test2 = Dat2[-Train2.vec,]
Comp.TrainACratio = rbind(Train1, Train2)
Comp.TestACratio = rbind(Test1, Test2)
rm(Datx, Dat1, Dat2, Train1, Test1, Train2, Test2)
  
## Adjusting to give ACratio only to ACglm when there are components so building the model and the comparison is always possible.
if (colnames(Comp.TrainACratio)[1] == "Comp.1") {
  Comp.Train = subset(Comp.TrainACratio, select = -c(AC.ratio))
  Comp.Test = subset(Comp.TestACratio, select = -c(AC.ratio))
} else {
  Comp.Train = Comp.TrainACratio
  Comp.Test = Comp.TestACratio
}

################### RANDOM FOREST  ####

trainTask <- makeClassifTask(data = Comp.Train, target = "Injury", positive = "1")
testTask <- makeClassifTask(data = Comp.Test, target = "Injury", positive = "1")

rf_model <- train(learner = makeLearner("classif.cforest", predict.type = "prob"), trainTask)
#rf_model <- train(final_all[[1,x$TablePosition]], trainTask)
rf_pred <- predict(rf_model, testTask)
rf_perf = prediction(rf_pred$data$prob.1, Comp.Test$Injury)
auc_rf = ROCR::performance(rf_perf, "tpr","fpr")
auc2_rf = ROCR::performance(rf_perf, "auc")
rm(rf_model)

## Conditional inference trees
rf2_model = cforest(Injury~.,data = Comp.Train, ntree = 1000)
rf2_pred = predict(rf2_model, Comp.Test, type = "prob")
rf2_perf = prediction(rf2_pred[,2], Comp.Test$Injury)
auc_rf2 = ROCR::performance(rf2_perf, "tpr","fpr")
auc2_rf2 = ROCR::performance(rf2_perf, "auc")
rm(rf2_model)

## RandomForest SRC
rf3_model = rfsrc(Injury~.,data = Comp.Train, ntree = 1000, family = class)
rf3_pred = predict(rf3_model, Comp.Test, type = "prob")
rf3_pred2 = data.frame(rf3_pred$predicted)
rf3_perf = prediction(rf3_pred2[,2], Comp.Test$Injury)
auc_rf3 = ROCR::performance(rf3_perf, "tpr","fpr")
auc2_rf3 = ROCR::performance(rf3_perf, "auc")

################### GLM LOGISTIC REGRESSION  ####
glm_model = glm(Injury~., data = Comp.Train, family = binomial)
glm_pred = predict(glm_model, Comp.Test,  type = "response")
glm_perf = prediction(glm_pred, Comp.Test$Injury)
auc_glm = ROCR::performance(glm_perf, "tpr","fpr")
auc2_glm = ROCR::performance(glm_perf, "auc")

################### GLM AIC STEP  ####
#stepAIC(object, scope, scale = 0,
#        direction = c("both", "backward", "forward"),
#        trace = 1, keep = NULL, steps = 1000, use.start = FALSE,
#        k = 2, …)

################### SVM  ####
trainTask2 = trainTask <- makeClassifTask(data = Comp.Train, target = "Injury", positive = "1")
svm_model = train(makeLearner("classif.ksvm", predict.type = "prob"), trainTask2)
svm_pred = predict(svm_model, testTask)
svm_perf = prediction(svm_pred$data$prob.1, Comp.Test$Injury)
auc_svm = ROCR::performance(svm_perf, "tpr","fpr")
auc2_svm = ROCR::performance(svm_perf, "auc")
rm(svm_model)

################### LDA LINEAR DISCRIMINANT ANALISIS  ####
if (!colnames(Comp.Train)[1] == "Comp.1") {
  lda_model = lda(Injury~., data = subset(Comp.Train, select = -c(EWMALoad.0.25, EWMALoad.0.45, EWMALoad.0.65,
                                                                  EWMALoad.0.85, EWMA.PlayerCentered.Load.0.05B)))
  lda_pred = predict(lda_model, type = "response" , newdata = Comp.Test)
  res = lda_pred$posterior[,2]
  lda_pred2 = prediction(res, Comp.Test$Injury)
  auc_lda = ROCR::performance(lda_pred2, "tpr","fpr")
  auc2_lda = ROCR::performance(lda_pred2, "auc")
}
################### GBM GRADIENT BOOSTING MODEL ####

gbm_model <- train(makeLearner("classif.gbm", predict.type = "prob"), trainTask)
gbm_pred <- predict(gbm_model, testTask)
gbm_perf = prediction(gbm_pred$data$prob.1, Comp.Test$Injury)
auc_gbm = ROCR::performance(gbm_perf, "tpr","fpr")
auc2_gbm = ROCR::performance(gbm_perf, "auc")
rm(gbm_model)

################### RIDGE REGRESSION ####
Comp.Train$Injury = as.numeric(Comp.Train$Injury)
Comp.Test$Injury = as.numeric(Comp.Test$Injury)

ridge_model = glmnet(as.matrix(subset(Comp.Train, select = -c(Injury))), Comp.Train[,"Injury"], family = "binomial", alpha=0,
                     lambda = cv.glmnet(as.matrix(subset(Comp.Train, select = -c(Injury))), Comp.Train[,"Injury"])$lambda.1se)
ridge_pred <- predict(ridge_model, as.matrix(subset(Comp.Test, select = -c(Injury))), type="response")
ridge_perf = prediction(ridge_pred, Comp.Test$Injury)
auc_ridge = ROCR::performance(ridge_perf, "tpr","fpr")
auc2_ridge = ROCR::performance(ridge_perf, "auc")

################### ACGLM ACUTE-CHRONIC RATIO ####

ACglm_model = glm(Injury~ AC.ratio, data = Comp.TrainACratio, family = binomial)
ACglm_pred = predict(ACglm_model, Comp.TestACratio,  type = "response")
ACglm_perf = prediction(ACglm_pred, Comp.TestACratio$Injury)
auc_ACglm = ROCR::performance(ACglm_perf, "tpr","fpr")
auc2_ACglm = ROCR::performance(ACglm_perf, "auc")

################### Saving results in Results.TB ####
TableRunRows = intersect(which(Results.TB$Runs == CurrentRun), which(Results.TB$Name == CurrentTable))
Results.TB = read.table("Results_TB.txt")
print(paste((sum(is.na(Results.TB))/16), "of", Ntables*Niterations, "runs remaining"))

Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF")), "LogLoss"] = LogLoss(Comp.Test$Injury, rf_pred$data$prob.1)
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF2")), "LogLoss"] = LogLoss(Comp.Test$Injury, rf2_pred)
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF3")), "LogLoss"] = LogLoss(Comp.Test$Injury, rf3_pred2[1])
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GLM")), "LogLoss"] = LogLoss(Comp.Test$Injury, glm_pred[2])
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "SVM")), "LogLoss"] = LogLoss(Comp.Test$Injury, svm_pred$data$prob.1)
if (!colnames(Comp.Train)[1] == "Comp.1") {
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "LDA")), "LogLoss"] = LogLoss(Comp.Test$Injury, lda_pred[2]$posterior)
}
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GBM")), "LogLoss"] = LogLoss(Comp.Test$Injury, gbm_pred$data$prob.1)
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RIDGE")), "LogLoss"] = LogLoss(Comp.Test$Injury, ridge_pred[2])
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "ACGLM")), "LogLoss"] = LogLoss(Comp.Test$Injury, ACglm_pred[2])

Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF")), "Auc"] = auc2_rf@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF2")), "Auc"] = auc2_rf2@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF3")), "Auc"] = auc2_rf3@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GLM")), "Auc"] = auc2_glm@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "SVM")), "Auc"] = auc2_svm@y.values
if (!colnames(Comp.Train)[1] == "Comp.1") {
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "LDA")), "Auc"] = auc2_lda@y.values
}
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GBM")), "Auc"] = auc2_gbm@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RIDGE")), "Auc"] = auc2_ridge@y.values
Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "ACGLM")), "Auc"] = auc2_ACglm@y.values

## Precision
# Precision.DF = data.frame(Comp.TestACratio$Injury, Comp.TestACratio$AC.ratio ,rf_pred$data$prob.1, rf2_pred[2], rf3_pred2[1],
#                      glm_pred[2], svm_pred$data$prob.1, gbm_pred$data$prob.1, ridge_pred[2], ACglm_pred[2])
# NumberofAlarms = nrow(Precision.DF[which(Precision.DF[,2] > 1.5 ), ])
# 
# Precision.DF = Precision.DF[order(Precision.DF[, 3], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[ ,4], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF2")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 5], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RF3")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 6], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GLM")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 7], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "SVM")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 8], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# if (!colnames(Comp.Train)[1] == "Comp.1") {
#   Precision.DF2 = data.frame(Comp.TestACratio$Injury, lda_pred[2]$posterior)
#   Precision.DF2 = Precision.DF2[order(Precision.DF2[, 2], decreasing = TRUE),]
#   Model.prediction = Precision.DF2[1:NumberofAlarms, 1]
#   Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "LDA")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# }  
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "GBM")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 9], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "RIDGE")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms
# Precision.DF = Precision.DF[order(Precision.DF[, 2], decreasing = TRUE),]
# Model.prediction = Precision.DF[1:NumberofAlarms, 1]
# Results.TB[intersect(TableRunRows, which(Results.TB$Runs.Method == "ACGLM")), "Precision"] = 100*length(Model.prediction[which(Model.prediction == 1)])/NumberofAlarms

## Saving Results.TB

output <- file("Results_TB.txt", "w")
write.table(Results.TB, file = output)
close(output)

## Saving image
ROC.curve = paste("ROC", CurrentTable, CurrentRun, ".png", sep = "")
png(filename = ROC.curve)
if (colnames(Comp.Train)[1] == "Comp.1") {
  plot(auc_rf, main = "ROC Curve", col = "#5AC8FA", lwd = 2) +
    plot(auc_rf2, add = TRUE, col = "#FFCC00", lwd = 2) +
    plot(auc_rf3, add = TRUE, col = "#FF9500", lwd = 2) +
    plot(auc_glm, add = TRUE, col = "#FF2D55", lwd = 2) +
    plot(auc_svm, add = TRUE, col = "#007AFF", lwd = 2) +
    plot(auc_gbm, add = TRUE, col = "#4CD964", lwd = 2) +
    plot(auc_ridge, add = TRUE, col = "#FF3B30", lwd = 2) +
    plot(auc_ACglm, add = TRUE, col = "#8E8E93", lwd = 2) +
  abline(a=0,b=1,lwd=2,lty=2, col="#EFEFF4")
} else {
  plot(auc_rf, main = "ROC Curve", col = "#5AC8FA", lwd = 2) +
    plot(auc_rf2, add = TRUE, col = "#FFCC00", lwd = 2) +
    plot(auc_rf3, add = TRUE, col = "#FF9500", lwd = 2) +
    plot(auc_glm, add = TRUE, col = "#FF2D55", lwd = 2) +
    plot(auc_svm, add = TRUE, col = "#007AFF", lwd = 2) +
    plot(auc_gbm, add = TRUE, col = "#4CD964", lwd = 2) +
    plot(auc_ridge, add = TRUE, col = "#FF3B30", lwd = 2) +
    plot(auc_ACglm, add = TRUE, col = "#8E8E93", lwd = 2) +
    plot(auc_lda, add = TRUE, col = "#117733", lwd = 2) +
    abline(a=0,b=1,lwd=2,lty=2, col="#EFEFF4")
}
dev.off()

## RF Light blue
## RF2 Yellow
## RF3 Orange
## GLM Pink
## SVM Blue
## LDA Dark green
## GBM Light green
## RIDGE Red
## ACglm Grey
## Abline bone
}

start.time = Sys.time()
Methods.log = lapply(split(Run.TB, 1:nrow(Run.TB)), FUN = F.Methiteration)
end.time = Sys.time()
time.taken = end.time - start.time
time.taken
Sys.time()

################### VISUALIZING RESULTS ####

Results.TB = data.table(read.table("Results_TB.txt", stringsAsFactors = TRUE))
Results.TB$Runs.Method = as.character(Results.TB$Runs.Method)

ResultsResume = Results.TB[, j = list(LogLoss = mean(LogLoss, na.rm=TRUE), Auc = mean(Auc, na.rm=TRUE)), by = list(Runs.Method)]
ResultsResume2 = Results.TB[, j =list(LogLoss = mean(LogLoss, na.rm=TRUE), Auc = mean(Auc, na.rm=TRUE), Games = Gamessessions, Comp = Components, Well = Wellness), by = list(Runs.Method, Name)]

Results.TB %>% group_by(Gamessessions) %>% 
  summarise(Logloss = mean(LogLoss, na.rm = TRUE), Auc = mean(Auc, na.rm = TRUE), Prec = mean(Precision, na.rm = TRUE))

Results.TB %>% group_by(Wellness) %>% 
  summarise(Logloss = mean(LogLoss, na.rm = TRUE), Auc = mean(Auc, na.rm = TRUE))

Results.TB %>% group_by(Components) %>% 
  summarise(Logloss = mean(LogLoss, na.rm = TRUE), Auc = mean(Auc, na.rm = TRUE))

Results.TB %>% group_by(Runs.Method, Name) %>% 
  summarise(Logloss = mean(LogLoss, na.rm = TRUE), Auc = mean(Auc, na.rm = TRUE))

p1 <- ggplot(ResultsResume, aes(x = LogLoss, y = Auc))
p1 + geom_point(aes(color=Runs.Method), size = 4)

p2 <- ggplot(ResultsResume2, aes(x = LogLoss, y = Auc))
p2 + geom_point(aes(color=Runs.Method), size = 4)

p3 <- ggplot(ResultsResume2, aes(x = LogLoss, y = Auc))
p3 + geom_point(aes(color=Runs.Method, shape = Games), size = 4)

p4 <- ggplot(ResultsResume2, aes(x = LogLoss, y = Auc))
p4 + geom_point(aes(color=Runs.Method, shape = Comp), size = 4)

p5 <- ggplot(ResultsResume2, aes(x = LogLoss, y = Auc))
p5 + geom_point(aes(color=Runs.Method, shape = Well), size = 4)

p7 <- ggplot(Results.TB, aes(x = LogLoss, y = Auc))
p7 + geom_point(aes(color=Runs.Method), position = position_jitter(width = 0.25, height = 0))

## Save model
saveRDS(model, "model.rds")

## Recover model
my_model <- readRDS("C:\\Users\\Ignacio S-P\\Desktop\\R Data processing\\Houston Texans\\ROCiteration\\Ridge_model.rds")


## Communication graphs
library(ggthemes)
library(scales)

theme_new <- theme_bw() +
  theme(plot.background = element_rect(size = 1, color = "blue", fill = "black"),
        text=element_text(size = 12, family = "Serif", color = "ivory"),
        axis.text.y = element_text(colour = "purple"),
        axis.text.x = element_text(colour = "red"),
        panel.background = element_rect(fill = "pink"),
        strip.background = element_rect(fill = muted("orange")))

p5 + theme_new

Datplots1 = Datx
Datplots1$Count = 1
Datplots2 = Datplots1 %>% group_by(Date) %>% 
  summarise(Injury = sum(Injury, na.rm = TRUE), Sessions = sum(Count, na.rm = TRUE))

G.Evo1 = ggplot(Datplots2, aes(x = Date)) +
  geom_line(data = Datplots2, aes(y = Sessions)) +
  geom_point(data = Datplots2, aes(y = Injury)) +
  scale_y_continuous(limits = c(0, 50), name = "Player's sessions") +
  theme_stata()
G.Evo1

G.Evo1 + stat_identity(data = Datplots2, aes(y = Injury))

G.Evo1 + theme_new() +  scale_fill_stata() + theme(legend.position = "right", axis.text.x  = element_text(angle=45, vjust=0.5, size=8))
G.Evo1 + theme_stata() + scale_fill_stata() + theme(legend.position = "right", axis.text.x  = element_text(angle=45, vjust=0.5, size=8))

## Total bars comparison

Bartable = data.frame(c(2215, 63, 450, 50, 459, 41, 487, 13), c("Total", "Total", "Prediction", "Prediction", "EWMA-ACratio", "EWMA-ACratio","ACratio", "ACratio"), c(".No Injury", "Injury", ".No Injury", "Injury", ".No Injury", "Injury", ".No Injury", "Injury"))
colnames(Bartable) = c("Value", "Type", "Injury")

G1 = ggplot(Bartable, aes(x = Type, y = Value, fill = Injury)) +
  geom_bar(data = Bartable[Bartable$Injury], stat = "identity", colour = "white", aes(fill = Injury)) +
  coord_flip() +
  scale_fill_discrete("#5AD25A", "#D25A5A")
G1

## Decay graphs
simpleset = data.frame(seq(0,40,1), rep(1,41))
colnames(simpleset) = c("Day", "Load")
EWMAdecay = seq(0.05, 0.95, 0.05)

spec2colours = c("#3d52a1", "#4d60a5", "#5d6ea9", "#6e7cad", "#7e8ab1", "#8e98b5", "#9ea6ba", "#aeb4be", "#f8e7c6",
                 "#f1d5b9", "#ebc3ad", "#e4b0a1", "#dd9e94", "#d78b88", "#c9666f", "#c25463", "#bb4157", "#b52e4a", "#ae1c3e")
Decay.graph = ggplot(simpleset, aes(x = Day, y = Load))

for (z in 1:length(EWMAdecay)) {
  temp = simpleset
  temp$Load = ((EWMAdecay[z])^temp$Day)^0.5
  Decay.graph = Decay.graph + geom_line(data = temp, col = spec2colours[z])
  Decay.graph = Decay.graph + geom_point(data = temp, col = spec2colours[z])
}
Decay.graph + scale_y_continuous(limits = c(0, 0.10))

## Timeline and EWMA graph
prueba2 = prueba
prueba2 = prueba2[prueba2$EWMARatio < 0.0015,]
prueba2$Injury = as.factor(prueba2$Injury)
prueba2 = prueba2[!is.na(prueba2$EWMARatio),]
prueba2$Dangerzone = 0.00015
prueba2$Outofdangerzone = 0.0015
G2 = ggplot() +
  geom_point(data = prueba2, aes(x = prueba2$Date, y = prueba2$EWMARatio, col=Injury)) +
  geom_line(data = prueba2, aes(x = prueba2$Date, y = prueba2$Dangerzone), col="red") +
  geom_line(data = prueba2, aes(x = prueba2$Date, y = prueba2$Outofdangerzone), col="green")
G2

## Timeline and EWMA graph2
prueba2 = prueba
prueba2$Injury = as.factor(prueba2$Injury)
prueba2 = prueba2[!is.na(prueba2$EWMARatio),]
prueba2$Dangerzone = 0.00015
prueba2$Outofdangerzone = 0.0015
tab = table(prueba2$PlayerName, prueba2$Injury)
tab[order(tab[,2]),]
prueba3 = prueba2[prueba2$PlayerName == "William Fuller",]

nrow(prueba3)
G3 = ggplot() +
  geom_bar(data = prueba3, stat= "identity", aes(x = prueba3$Date, y = prueba3$Estimated.Training.Load, fill = Injury)) +
  geom_line(data = prueba3, aes(x = prueba3$Date, y = prueba3$AC.ratio*400), col="purple", size = 1) +
  # geom_line(data = prueba3, aes(x = prueba3$Date, y = prueba3$Dangerzone), col = "red", size = 1) +
  # geom_line(data = prueba3, aes(x = prueba3$Date, y = prueba3$Outofdangerzone), col = "green", size = 1) +
  # geom_point(data = prueba3, aes(x = prueba3$Date, y = prueba3$Estimated.Training.Load/400000), size = 2) +
  # geom_line(data = prueba3, aes(x = prueba3$Date, y = prueba3$Estimated.Training.Load/400000), col="black", size = 1) +
  geom_line(data = prueba3, aes(x = prueba3$Date, y = (prueba3$EMWALoad.0.85*1000)^0.5), col="blue", size = 1) +
  geom_line(data = prueba3, aes(x = prueba3$Date, y = (prueba3$EWMARatio*9000000)^0.5), col="green", size = 1) +
  geom_line(data = prueba3, aes(x = prueba3$Date, y = (prueba3$EMWALoad.0.05 *80000000)^0.3), col="yellow", size = 1)
G3


prueba4 = prueba3[order(prueba3$Estimated.Training.Load, decreasing = TRUE),]
prueba4 = prueba4[order(prueba4$DaystoNext),]
View(prueba4[,c(3,28,41,59, 70)])

pruebaX = prueba
pruebaX = prueba[!is.na(prueba$EWMARatio), ]
pruebaX[pruebaX$EWMARatio > 0.0015, "Set"] = 1
pruebaX[pruebaX$EWMARatio < 0.0015, "Set"] = 2
pruebaX[pruebaX$EWMARatio < 0.00015, "Set"] = 3
pruebaY <- pruebaX[,sapply(pruebaX, is.numeric)]
pruebaY = pruebaY[,-c(40:58, 62:66)]
Bip = CanonicalBiplot(pruebaY, as.factor(pruebaY$Set), SUP = NULL, InitialTransform = 5)
plot(Bip, mode="s")

plot3d(pruebaX[,c(70,59,43)], col=pruebaX$Set)
