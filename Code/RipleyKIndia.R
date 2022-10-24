rm(list=ls())
library(spatstat)
library(rgdal)
library(maptools)


india_boundary <- readOGR('./BhugolCode/Data/ShapeFiles/India', 'IND_adm0_Mainland') #read the shape file for boundary of india (mainland)
india <- as(india_boundary, "owin")


suffixes <- c('pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti','palle')

for (suff in suffixes){
  #Shape files for frequent suffixes and random samples for 'pur' and 'adi' were done using ArcGIS Pro and the gazetteer for India
  if (suff == 'pur'){
    shapefile_suff <- readOGR(dsn = './BhugolCode/Data/ShapeFiles/India', layer = 'purSampled15000') #pur has a lot of place names. So, randomly sampled 15000 place names for computing K
  }
  else if (suff == 'adi'){
    shapefile_suff <- readOGR(dsn = './BhugolCode/Data/ShapeFiles/India', layer = 'adiSampled15000') #adi has a lot of place names. So, randomly sampled 15000 place names for computing K
  } else {
    shapefile_suff <- readOGR(dsn = './BhugolCode/Data/ShapeFiles/India', layer = suff)
  }
  suff_ppp <- as(shapefile_suff, 'ppp')
  Window(suff_ppp) <- india
  k <- Kest(suff_ppp, correction = 'border')
  plot(k, main=suff, xlim=range(0:3), ylim=range(0:100))
}


