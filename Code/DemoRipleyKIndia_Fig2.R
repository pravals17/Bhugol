rm(list=ls())
library(spatstat)
library(rgdal)
library(maptools)


india_boundary <- readOGR('./BhugolCode/Data/ShapeFiles/India', 'IND_adm0_Mainland') #read the shape file for boundary of india (mainland)
india <- as(india_boundary, "owin")

india_random <- readOGR(dsn = './BhugolCode/Data/ShapeFiles/India', 'demoRipleys')

random_ppp <- as(india_random, 'ppp')
marks(random_ppp) <- NULL
Window(random_ppp) <- india
k <- Kest(random_ppp, correction = 'border')
plot(k) #Ripleys K values for the distribution of place names

plot(random_ppp)#Distribution of place names in India
