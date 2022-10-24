rm(list=ls())
library(spatstat)
library(rgdal)
library(maptools)


us_boundary <- readOGR('./BhugolCode/Data/ShapeFiles/USA', 'tl_2020_us')
us <- as(us_boundary, "owin")

suffixes <- suffixes <- c('ville', 'epark', 'tates', 'town', 'ing', 'hill', 'wood','ights')

for (suff in suffixes){
  #Shape files for frequent suffixes were created using ArcGIS Pro and the gazetteer for the USA
  shapefile_suff <- readOGR(dsn = './BhugolCode/Data/ShapeFiles/USA', layer = suff)
  suff_ppp <- as(shapefile_suff, 'ppp')
  Window(suff_ppp) <- us
  k <- Kest(suff_ppp, correction = 'border')
  plot(k, main=suff,xlim=range(0:6), ylim=range(0:160))
}