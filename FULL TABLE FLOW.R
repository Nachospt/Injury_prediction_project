################################################ FULL DATA FLOW ################################################

## 0.1 Step: libraries load
## 0.2 Step: table load
## 1.Step: First changes and filtering
## 2.Step: RPE and GPS data from 1 session is in 2 rows. Correcting this in 'Full.table'
## 3.Step: Merging Full.table to injury table
## 4.Step: Feature engineering (creating additional variables that models can use)
## 5.Step: Game sessions NA imputing (adding last value within the last 5 days)
## 6.Step: Adding last features: EWMA and manually adding age
## 7.Step: Config table: table versions for analysis
## 8.Step: Final table creations and saving

################### 0.1 Step: libraries load ####
library(data.table)
library(dplyr)
library(RODBC)
library(sqldf)
library(MultBiplotR)
library(ggplot2)
library(scales)
library(mice)
library(lubridate)
library(lattice)
library(TTR)
library(randomForest)
library(ROCR)
library(party)
library(MASS)
library(caTools)
library(e1071)
library(partykit)
library(gbm)
library(tidyverse)
library(broom)
library(glmnet)
library(mlr)
library(randomForestSRC)
library(kernlab)

################### 0.2 Step: table load ####

setwd('C:\\Users\\Ignacio S-P\\Desktop\\R Data processing\\Houston Texans\\')
Dat1 = read.csv("Soft Tissue Injuries Strains.csv", stringsAsFactors = FALSE)
Dat1 = Dat1[,-5]
Dat2 = read.csv("houstontexans0016 - RPE Catapult 2016-17.csv", stringsAsFactors = FALSE)

################### 1.Step: First changes and filtering ####
## Changing date format
Dat1$Date = as.POSIXct(strptime(Dat1[, "Date"], "%m/%d/%Y"))
Dat2$Date = as.POSIXct(strptime(Dat2[, "Date"], "%m/%d/%Y %I:%M:%S %p"))

## Adding PlayerName variable
Dat2$PlayerName = paste(Dat2$First.Name, Dat2$Last.Name)

## Filtering
DF.Session = Dat2[Dat2$Name == "Session RPE - Practice", ]
DF.Catapult = Dat2[!Dat2$Name == "Session RPE - Practice", ]
DF.Session = DF.Session[!DF.Session$Session.RPE == 0,]
DF.Session = DF.Session[!DF.Session$Strain == -Inf,]
DF.Catapult = DF.Catapult[!DF.Catapult$Total.Distance == 0,]

## Obtaining vectors with unique Playernames or unique colnames
Full.table = c()
Players.vector = unique(c(DF.Session$PlayerName, DF.Catapult$PlayerName))
Variables.vector = unique(c(colnames(DF.Session), colnames(DF.Catapult)))

## Making round date
DF.Session$RoundDate = floor_date(DF.Session$Date, unit = "days")
DF.Catapult$RoundDate = floor_date(DF.Catapult$Date, unit = "days")

## Restore the date format (as date it would be converted to number of seconds in the following operations, so harder to convert after)
DF.Session$Date = format(DF.Session$Date, format = "%m/%d/%Y")
DF.Catapult$Date = format(DF.Catapult$Date, format = "%m/%d/%Y")

################### 2.Step: RPE and GPS data from 1 session is in 2 rows. Correcting this. #####
## Loop that creates a table with unqiue combinations PlayerName-Date.

for (z in Players.vector) {
  # Joins the PlayerName combinations
  temp.ses = DF.Session[DF.Session$PlayerName == z, c("PlayerName", "RoundDate")]
  temp.cat = DF.Catapult[DF.Catapult$PlayerName == z, c("PlayerName", "RoundDate")]
  temp.ses$PlayerDate = paste(temp.ses$PlayerName, temp.ses$RoundDate)
  temp.cat$PlayerDate = paste(temp.cat$PlayerName, temp.cat$RoundDate)
  temp = rbind(temp.ses, temp.cat)
  # Order the rows to check duplicates
  temp = temp[order(temp$PlayerDate),]
  duplicates = c()
  for (z in 1:(nrow(temp)-1)) {
    if ((temp[z, "PlayerName"] == temp[z+1, "PlayerName"]) && (temp[z, "RoundDate"] == temp[z+1, "RoundDate"])) {
      duplicates = rbind(duplicates, z)
    }
  }
  # If there are duplicates, they are dropped
  if(!is.null(duplicates)) {
    temp = temp[-duplicates,]
  }
  # Add the rows to the Full.table
  Full.table = rbind(Full.table, temp)
}

## Creating the columns in a general table (Full.table) and populating them
Variables.vector = Variables.vector[-which(Variables.vector == "PlayerName")]
for (z in Variables.vector) {
  Full.table[,z] = NA
  for (i in 1:nrow(DF.Session)) {
    Full.table[intersect(which(Full.table$RoundDate == DF.Session[i, "RoundDate"]),
                         which(Full.table$PlayerName == DF.Session[i, "PlayerName"])),z] = DF.Session[i,z]
  }
  for (i in 1:nrow(DF.Catapult)) {
    if (!is.na(DF.Catapult[i, z])) {
      Full.table[intersect(which(Full.table$RoundDate == DF.Catapult[i, "RoundDate"]),
                           which(Full.table$PlayerName == DF.Catapult[i, "PlayerName"])),z] = DF.Catapult[i,z]
    }
  }
}

## Removing unnecessary columns

Full.table = Full.table[,-which(colnames(Full.table) %in% c("PlayerDate",
                                                            "Last.Name",
                                                            "First.Name",
                                                            "Name",
                                                            "Hi.Int.Accell.Efforts.Team.Avg",
                                                            "X..of.Hi.Int.Accel.Efforts",
                                                            "AC.Ratio.Flag"))]

################### 3.Step: Merging Full.table to injury table ####
## Removing the injuries previous to the date in which we have GPS data
Dat1.2 = Dat1[Dat1$Date > as.Date("2016-09-11 BST"), ]
Dat1.2 = Dat1.2[Dat1.2$Date < as.Date("2017-01-13 BST"), ]

## Calculating inner join vector of players and the dates in which these players trained and they got injured
Pre.Inj.Players = unique(Dat1.2$Full.Name)
Inj.Players = unique(Full.table$PlayerName)[unique(Full.table$PlayerName) %in% Pre.Inj.Players]

Training.dates = unique(Full.table[which(Full.table$PlayerName %in% Inj.Players), "RoundDate"])
Inj.dates = unique(Dat1.2$Date)[unique(Dat1.2$Date) %in% unique(Full.table[which(Full.table$PlayerName %in% Inj.Players), "RoundDate"])]
# To check list...
# unique(Dat1.2$Date)[order(unique(Dat1.2$Date))]
# Training.dates[order(Training.dates)]
# Inj.dates[order(Inj.dates)]

## Restoring the date format
Full.table$Date = as.POSIXct(strptime(Full.table[, "Date"], "%m/%d/%Y"))

## To make sense out of data in the analysis, the injury is 1 in the row corresponding to the session before the injury date
## We give similar names to the columns in the injuries table
colnames(Dat1.2)[1:2] = c("PlayerName", "NextDate")
Full.table$NextDate = as.Date(NA)

## Creating next date variable. It selects each player rows and every row gets the date value of the next row.
for (z in unique(Full.table$PlayerName)) {
  temp = which(Full.table$PlayerName == z)
  if(length(temp) > 1) {
    Full.table[temp[2:length(temp)-1], "NextDate"] = as.character(Full.table[temp[2:length(temp)], "Date"])
  }
}

## Creating table keys in each table to transfer values to the new injury column in Full.table
Analysis.table = Full.table
Analysis.table$Key = paste(Analysis.table$PlayerName, Analysis.table$NextDate)
Dat1.2$Key = paste(Dat1.2$PlayerName, Dat1.2$NextDate)

## Assigning 0 to non injury rows and 1 to injury rows, those whose key exists in both tables
Analysis.table$Injury = 0
Analysis.table$TypeInjury = "none"
for (z in 1:nrow(Dat1.2)) {
  Analysis.table[Analysis.table$Key == Dat1.2[z, "Key"], "Injury"] = 1
  Analysis.table[Analysis.table$Key == Dat1.2[z, "Key"], "TypeInjury"] = Dat1.2[z, "Injuries.Level.2"]
}

## A table with only injuries
#Analysis.table2 = merge(Analysis.table, Dat1.2, by = "Key")

################### 4.Step: Feature engineering (creating additional variables that models can use) ####
## Creating number of past injuries variable
Analysis.table$Num.Inj = apply(Analysis.table, 1, function (x) {
  return(length(intersect(intersect(which(Analysis.table$PlayerName == x[[1]]), which(Analysis.table$Injury == 1)), which(Analysis.table$Date < x[[3]]))))
})

## Creating days since last injury
Analysis.table$Latest.Inj = apply(Analysis.table, 1, function (x) {
  Temp.Injuries = intersect(intersect(which(Analysis.table$PlayerName == x[[1]]), which(Analysis.table$Injury == 1)), which(Analysis.table$Date < x[[3]]))
  if (length(Temp.Injuries) > 0) {
    Last.Inj = difftime(x[3], max(Analysis.table[Temp.Injuries, 3]), units = "days")
    return(Last.Inj)
  } else {return(365)}
})

## Creating days to next session
Analysis.table$DaystoNext = as.numeric(round(difftime(Analysis.table$NextDate, Analysis.table$RoundDate, units = "days"), digits = 0))

## Creating date as numeric variable
Analysis.table$RoundDate = as.numeric(Analysis.table$RoundDate)

## Temperature variable (obtained on internet)
measures = c(91, 81, 95, 94, 71, 85,150,139,138, 63, 58, 83, 94,117,161, 98, 79, 79,117,135,171, 77, 60, 89,172,149, 96, 87,126,171,199,
             205,203,129, 91, 79, 99, 92,122,108,119,177,187,158,170,196,156,165,153,206,209,205,193,171,116,126,117,120,149,188,
             200,171,208,176,161,158,192,206,207,188,200,193,202,214,222,216,214,214,169,116, 94,137,196,174,132,149,193,167,187,211,239,
             189,140,149,177,192,198,193,206,185,208,224,223,191,199,194,209,218,196,211,211,200,211,202,193,228,245,233,238,254,248,
             247,207,181,198,218,214,217,221,241,256,254,259,254,235,214,203,218,235,200,209,225,238,244,254,271,263,214,229,256,247,254,
             254,221,216,218,227,242,264,264,263,264,276,271,273,297,302,303,300,303,273,284,277,287,287,279,284,287,291,283,272,286,
             292,298,307,309,309,308,308,307,301,303,308,306,312,311,308,304,296,288,283,290,304,308,309,312,304,277,264,278,283,298,297,
             300,306,307,308,306,308,314,312,316,319,321,314,302,259,253,254,256,251,269,277,276,257,278,284,269,262,262,257,252,273,289,
             294,296,281,276,269,281,282,291,275,259,266,273,270,278,283,277,287,291,300,294,279,277,274,268,256,251,246,251,233,209,
             211,233,230,243,267,268,263,252,223,204,226,254,261,248,249,260,266,273,264,248,198,167,181,213,213,223,216,222,227,223,207,
             229,243,246,246,233,211,213,196,206,179,179,193,176,165,210,185,210,225,137,104,137,190,213,156,172,170,153,231,203,188,
             111,131,119,113,109,134,115,108, 58, 73,168,204,202,164,122,121,240, 85, 31, 61,129,195,192,224,226,241,230,233,206,113,144,
             197,204,167,101, 99, 47,-11, 17,102,201,217,227,216,209,194,220,195,173,147,181,197,173,154,174,212,123, 83,109,111,144,173,
             194,204,135,113,197,224,240,236,172,157,237,242,216,179,120,121,140,191,221,192,170,189,196,217,156,146,223,244,
             250,157,141,152,166,216,236,180,207,224,205,141,141,129,137,166,218,208,228,236,230,229,230,230,229,221,247,239,236,200,206,
             211,235,218,235,214,181,189,208,224,236,214,195,223,217,223,241,235,211,228,245,247,236,170,188,229,267,202,262,274,207,
             200,234,258,202,198,208,226,224,231,244,256,254,239,238,241,252,263,278,279,279,263,228,229,216,241,271,294,289,244,254,257)
Temp.Date = seq(1,length(measures), 1)
Temp.Date = as.character(as.Date("2016-01-01 BST") + Temp.Date)
Temperature.TB = cbind(Temp.Date, measures)

## Subtituting the days in which the player played in another city

Temperature.TB[Temp.Date == "2016-09-22", "measures"] = 180
Temperature.TB[Temp.Date == "2016-10-09", "measures"] = 80
Temperature.TB[Temp.Date == "2016-10-24", "measures"] = 160
Temperature.TB[Temp.Date == "2016-11-13", "measures"] = 250
Temperature.TB[Temp.Date == "2016-11-21", "measures"] = 130
Temperature.TB[Temp.Date == "2016-12-04", "measures"] = 10
Temperature.TB[Temp.Date == "2016-12-11", "measures"] = 10
Temperature.TB[Temp.Date == "2017-01-01", "measures"] = 90

Analysis.table$Temperature = NA
for (z in 1:nrow(Analysis.table)) {
  if (!is.na(as.character(Analysis.table[z, "NextDate"]))) {
  print(paste(z, as.character(Analysis.table[z, "NextDate"])))
  Analysis.table[z, "Temperature"] = Temperature.TB[which(Temperature.TB[,"Temp.Date"] == as.character(Analysis.table[z, "NextDate"])), "measures"]
  }
}

## Match/Training session
GameSession = c("2016-09-11","2016-09-18","2016-09-22","2016-10-02","2016-10-16",
                           "2016-10-24","2016-10-30","2016-11-13","2016-11-21","2016-11-27",
                           "2016-12-04","2016-12-11","2016-12-18","2016-12-24","2017-01-01",
                           "2016-01-17")

Analysis.table$Typeofsession = NA
for (z in 1:nrow(Analysis.table)) {
  if (as.character(Analysis.table[z, "NextDate"]) %in% GameSession) {
    Analysis.table[z, "Typeofsession"] = "Match"
  } else {
    Analysis.table[z, "Typeofsession"] = "Training"
  }
}

################### 5.Step: Game sessions NA imputing (adding last value within the last 5 days) ####
Dat = Analysis.table
Dat = Dat[which(is.finite(Dat$Monotony)),]
Dat$Date = as.POSIXct(strptime(Dat[, "Date"], "%Y-%m-%d"))
ImputedDat = Dat[order(Dat$Date),]

## Save which rows are to be imputed (for later inc/exc of the analysis)
Pre.Gamerows = ImputedDat[, 1:44]
Gamerows = ImputedDat[-which(complete.cases(Pre.Gamerows) == TRUE), "Key"]

## Filling NA values of game sessions ("team average" columns)
for (i in nrow(ImputedDat) - (1:nrow(ImputedDat) - 1)) {
  back = 1
  while ((is.na(ImputedDat[i-back, "Total.Distance"])) |
         (ImputedDat[i-back, "Date"] != ImputedDat[i, "Date"])) {
    back = back + 1
    if (difftime(ImputedDat[i, "Date"], ImputedDat[i-back, "Date"], unit = "days") > 5) {
      break
    }
  }
  if (difftime(ImputedDat[i, "Date"], ImputedDat[i-back, "Date"], unit = "days") < 6) {
    for (z in 4:12) {
      if (is.na(ImputedDat[i, z])) {
        ImputedDat[i, z] = ImputedDat[i-back, z] 
      }
    }
  }
}

## Filling NA values of game sessions (rest of the columns)
for (i in nrow(ImputedDat) - (1:nrow(ImputedDat) - 1)) {
  back = 1
  while ((is.na(ImputedDat[i-back, "Total.Distance"])) |
         (ImputedDat[i-back, "PlayerName"] != ImputedDat[i, "PlayerName"])) {
    back = back + 1
    if (difftime(ImputedDat[i, "Date"], ImputedDat[i-back, "Date"], unit = "days") > 5) {
      break
    }
  }
  if (difftime(ImputedDat[i, "Date"], ImputedDat[i-back, "Date"], unit = "days") < 6) {
    for (z in 4:length(ImputedDat)) {
      if (is.na(ImputedDat[i, z])) {
        ImputedDat[i, z] = ImputedDat[i-back, z]
      }
    }
  }
}

## Filtering the columns that still have no values in a key variable of the first set (GPS-RPE): Total.Distance
Datx = ImputedDat
Datx = ImputedDat[which(!is.na(ImputedDat$Total.Distance)),]
# Datx = Datx[which(complete.cases(Datx[, c(1:44)])),]

## Removing the character variables and the ones with most values = 0 that correspond to NA cases
Datx2 = subset(Datx, select = -c(NextDate, TypeInjury))

Datx2$Latest.Position = as.character(Datx2$Latest.Position)
Datx3 = Datx2
################### 6.Step: Adding last features: EWMA and manually adding age ####

## Loop to create EWMA columns
EWMAdecay = seq(0.05, 0.95, 0.1)

## EWMA Load
New.variables <- unlist(lapply(split(Datx3, 1:nrow(Datx3)), function(x) {
  EWMALoad = c()
  pre.temp = Datx3[which(Datx3$PlayerName == x$PlayerName), ]
  temp = pre.temp[which(difftime(x$Date, pre.temp$Date, units = "days") >= 0), ]
  temp$difftime = as.numeric(difftime(x$Date, temp$Date, units = "days")) + x$DaystoNext
  
  for (z in 1:length(EWMAdecay)) {
    temp$weight =  (EWMAdecay[z])^temp$difftime
    EWMALoad[z] = sum(temp$Estimated.Training.Load * temp$weight)
  }
  
  if (is.null(EWMALoad)) {
    return(c(rep(0, length(EWMAdecay))))
  } else {
    return(EWMALoad)
  }
})) %>%
  matrix(length(EWMAdecay), nrow(Datx3)) %>%
  t()

colnames(New.variables) = c(rep("0", length(EWMAdecay)))
for (z in 1:length(EWMAdecay)) {
  colnames(New.variables)[z] = paste("EWMALoad.", EWMAdecay[z], sep = "")
}

## EWMA Load player centered
Datx3$New.Estimated.Training.Load = NA
for (z in 1:nrow(Datx3)) {
  Datx3[z, "New.Estimated.Training.Load"] = Datx3[z, "Estimated.Training.Load"] - mean(Datx3[which(Datx3$PlayerName == Datx3[z, "PlayerName"]), "Estimated.Training.Load"])
}
EWMAdecay = c(0.05, 0.15, 0.95)

New.variables2 <- unlist(lapply(split(Datx3, 1:nrow(Datx3)), function(x) {
  EWMALoad = c()
  pre.temp = Datx3[which(Datx3$PlayerName == x$PlayerName), ]
  temp = pre.temp[which(difftime(x$Date, pre.temp$Date, units = "days") >= 0), ]
  temp$difftime = as.numeric(difftime(x$Date, temp$Date, units = "days")) + x$DaystoNext
  
  for (z in 1:length(EWMAdecay)) {
    temp$weight =  (EWMAdecay[z])^temp$difftime
    EWMALoad[z] = sum(temp$New.Estimated.Training.Load * temp$weight)
  }
  
  if (is.null(EWMALoad)) {
    return(c(rep(0, length(EWMAdecay))))
  } else {
    return(EWMALoad)
  }
})) %>%
  matrix(length(EWMAdecay), nrow(Datx3)) %>%
  t()

colnames(New.variables2) = c(rep("0", length(EWMAdecay)))
for (z in 1:length(EWMAdecay)) {
  colnames(New.variables2)[z] = paste("EWMA.PlayerCentered.Load.", EWMAdecay[z], "B", sep = "")
}

## AGE
Age.TB = data.frame(c(unique(Datx3$PlayerName)), c(27,24,26,31,23,24,24,23,25,23,23,27,25,24,25,24,29,24,33,24,25,22,25,24,27,24,26,24,
                                         29,28,28,24,23,23,27,24,25,23,33,26,28,30,24,24,26,27,26,31,26))
colnames(Age.TB) = c("PlayerName", "Age")
Datx3$Age = NA
for (z in Age.TB$PlayerName) {
  Datx3[Datx3$PlayerName == z, "Age"] = Age.TB[Age.TB$PlayerName == z, 2]
}

## Join to the original table
Datx4 = cbind(Datx3[1:41], Datx3$Age, Datx3$Temperature, Datx3$Typeofsession, New.variables, New.variables2, New.Weight)
colnames(Datx4)[42] = "Age"
colnames(Datx4)[43] = "Temperature"
colnames(Datx4)[44] = "Typeofsession"
Datx4$Temperature = as.numeric(Datx4$Temperature)
Datx4$Injury = as.factor(Datx4$Injury)

################### 7.Step: Config table: table versions for analysis ####
## A data frame with a row with a single combination of selected features is built. This table will guide the table creation

Tableinstructions = data.frame(c(rep("Gamesyes", 18), rep("Gamesno", 18)),
                               c(rep("Sprain", 6), rep("Strain", 6), rep("all", 6)),
                               c(rep("Wellfilt", 2), rep("Wellimp", 2), rep("Wellno", 2)),
                               c(rep("Compyes", 1), rep("Compno", 1)),
                               stringsAsFactors = FALSE)
names(Tableinstructions) = c("Gamessessions", "Injuries", "Wellness", "Components")
Tableinstructions = Tableinstructions[Tableinstructions$Wellness == "Wellno",]
Tableinstructions = Tableinstructions[Tableinstructions$Injuries == "all",]

################### 8.Step: Final table creations and saving ####
Table.list = list()

for (z in 1:nrow(Tableinstructions)) {
  if (Tableinstructions[z, 3] == "Wellimp") {
    Table1 = Datx4
  } else if (Tableinstructions[z, 3] == "Wellfilt") {
    Table1 = Datx4[-which(Datx4$Key %in% Wellimprows), ]
  } else if (Tableinstructions[z, 3] == "Wellno") {
    Table1 = Datx4[, c(1:69)]
  }
  print("Table1")
  print(dim(Table1))

  Table2 = Table1
  print("Table2")
  print(dim(Table2))
  
  if (Tableinstructions[z, 1] == "Gamesyes") {
    Table3 = Table2
  } else if (Tableinstructions[z, 1] == "Gamesno") {
    Table3 = Table2[-which(Table2$Key %in% Gamerows), ]
  }
  print("Table3")
  print(dim(Table3))
  
 if (Tableinstructions[z, 4] == "Compyes") {
   pre.temp = subset(Table3, select = -c(Injury))
   temp <- pre.temp[,sapply(pre.temp, is.numeric)]
   print("temp")
   print(dim(temp))
   Princ = princomp(temp, cor=TRUE)
   Table4 = cbind(Princ$scores, pre.temp[!sapply(pre.temp, is.numeric)], Table3$AC.ratio, Table3$Injury)
   colnames(Table4)[length(Table4)] = "Injury"
   colnames(Table4)[length(Table4)-1] = "AC.ratio"
 } else if (Tableinstructions[z, 4] == "Compno") {
   Table4 = Table3
 }
  print("Table4")
  print(dim(Table4))
  Table.list[[z]] = Table4
}

## Saving the table
setwd('C:\\Users\\Ignacio S-P\\Desktop\\R Data processing\\Houston Texans\\HoustonTables\\')

for (z in 1:length(Table.list)) {
  TableName = paste("Table", Tableinstructions[z,1],
                    Tableinstructions[z,2],
                    Tableinstructions[z,3],
                    Tableinstructions[z,4],
                    Tableinstructions[z,5],
                    ".csv", sep = "")
  output <- file(TableName, "w")
  write.csv(Table.list[z], file = output)
  close(output)
}

## Saving the table route
output <- file("Tableinstructions.csv", "w")
write.csv(Tableinstructions, file = output)
close(output)
