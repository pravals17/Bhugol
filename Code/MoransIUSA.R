rm(list=ls())
library("sf")
library("ggplot2")
library(spatstat)
library(rgdal)
library(plyr)
library(maptools)
library(spdep) #poly2nb

us_boundary <- readOGR('./BhugolCode/Data/ShapeFiles/USA', 'tl_2020_us')
us <- as(us_boundary, "owin")

us_state = readOGR(dsn = './BhugolCode/Data/ShapeFiles/USA', 'tl_2020_us_state')

#read data related to frequent suffixes
freq_suffixes <- read.csv('./BhugolCode/Data/FrequentSuffixes/USA/Freqsuffixpopulatedplace_withLatLong.csv', header = TRUE)

suffixes <- c('ville', 'epark', 'tates', 'town', 'ing', 'hill', 'wood','ights')

frequentsuffixes <- c()
MoranIValue <- c()
ExpectedValue <-c()
Standarddeviation <- c()
p.value <- c()
Alternatehypothesis <- c()
Zscores <- c()

for (suff in suffixes){
  print('...............................................................')
  print(suff)
  print('...............................................................')
  df_subset <- freq_suffixes[which(freq_suffixes$Suffix == suff),]
  Statelist <- unique(df_subset[,3])
  stateName <- c()
  countsuff <- c()
  
  for (state in us_state$STUSPS){
    #count suffixes present in each state of the USA
    if (state %in% Statelist) {
      stateName <- c(stateName, state)
      countsuff <- c(countsuff, table(df_subset$State)[state])
    }
    else{
      stateName <- c(stateName, state)
      countsuff <- c(countsuff, 0)
    }
    
  }
  df_suff <- data.frame(stateName, countsuff)
  dfsuffmerge <- merge(us_state, df_suff, by.x = 'STUSPS', by.y = 'stateName')
  nb = poly2nb(dfsuffmerge, queen = TRUE)
  nbw <-  nb2listw(nb, style='W') #spatial weights based on neighborhood for using in moran function to compute moran's I
  moranI <- moran(df_suff$countsuff, nbw, n=length(nbw$neighbours), S0=Szero(nbw)) #compute moran's I
  moranItest <- moran.test(df_suff$countsuff, nbw, randomisation=FALSE) # randomisation is false, denoting normality
  MC <- moran.mc(df_suff$countsuff, nbw, nsim = 999) #monte carlo simulation
  print(MC)
  
  frequentsuffixes <- c(frequentsuffixes, suff)
  MoranIValue <- c(MoranIValue, MC$statistic)
  ExpectedValue <- c(ExpectedValue, moranItest$estimate)
  Standarddeviation <- c(Standarddeviation, moranItest$statistic)
  p.value <- c(p.value, MC$p.value)
  Alternatehypothesis <- c(Alternatehypothesis, MC$alternative)
  
  z_score <- (MC$statistic - moranItest$estimate['Expectation'])/sqrt(moranItest$estimate['Variance'])
  Zscores <- c(Zscores, (MC$statistic - moranItest$estimate['Expectation'])/sqrt(moranItest$estimate['Variance']) )
  print(z_score)
}

df_moran <- data.frame(frequentsuffixes, MoranIValue, ExpectedValue, Standarddeviation, p.value, Zscores, Alternatehypothesis)