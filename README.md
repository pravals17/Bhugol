# Bhugol
A novel data-driven spatially-aware algorithm, Bhugol, that leverages the spatial patterns and the spatial context of place names to automatically geocode the non-gazetteer place names. 


######################### Data #########################################

The following folders contain the datasets used in this study for both the USA and India.

Corpus: Copora of news reports for the USA and India. Note that due to large size, the larger corpus used to compute the co-occurrence statistics in our algorithm is not inlcuded in this upload. However, the folder contains news reports that contain all the NGPAs and NGPs used in the research (The text in the uploads is raw and requires some preprocessing before using them). If needed, the large corpus can be scraped easily from the newspapers using the source URL mentioned in GDELT https://www.gdeltproject.org/ for USA. Please use the news reports from the year 2020 and 2021, respectively for the USA. For India, the large corpus can be created easily using the news reports from the Times of India (timesofindia.indiatimes.com/), The Hindu (https://www.thehindu.com/), The Pioneer (https://www.dailypioneer.com/), Economic Times (https://economictimes.indiatimes.com/), Assam Tribune (https://assamtribune.com/), The Kashmir Oberver (https://kashmirobserver.net/), and Incredible Orissa (https://incredibleorissa.com/) for the year 2015.

Frequent Sufixes: List of place names from the USA and India along with their suffix, Lat, Long. This is used to compute Ripley's K and Morans I. 

GroundTruth: This contains the four ground truth datasets: NGPA-USA, NGP-USA, NGPA-India, and NGP-India.

ShapeFiles: This contains the shape files required for statistical analysis of patterns in place names. (Using the gazetteers, ArcGIS was used to create shape files)

SuffixesHDBSCANCLUSTERS: It includes the clusters and their centroids obtained for frequent suffixes (ArcGIS Pro was used to create clusters of place names with frequent suffixes using HDBSCAN algorithm for both the USA and India).


######################### Named Entity Recognition (NER) System ##############

NER system identifies locations (i.e., place names) mentioned in text and is used in this study to extract cooccurring place names for a given non-gazetteer place name. The following NER systems are used in this study.

Stanford NER: NER used to extract cooccurring place names from text for non-gazetteer places for the USA.
SANS: NER used to extract cooccurring place names from text for non-gazetteer places for India. In order to run the code for NER, please download the gazetteer for India as described in the 'NOTE' section (below) of this readme text.

Both the NER systems are stored in 'NERFiles' folder.


####################### Geocoders used for comparison ##################

The codes and implementation details of the current state-of-the-art geocoders are available from the authors of EUPEG in https://github.com/geoai-lab/EUPEG (Edinburg, TopoCluster, Clavin, and CamCoder) and the code and details for Mordecai geocoder is in https://github.com/openeventdata/mordecai.

###################### Bhugol #########################################

The codes for the statistical analysis of patterns in place names and the different approaches of geocoding in Bhugol are included in the folder 'Code'. 
The code can be run after installing the required libraries and providing the correct path to the datasets. The code is commented for better readability.

###################### NOTE ###############################################

Due to large size the gazetteers for the USA and India used in this research are not included in the upload. However, it is very easy to download the gazetteers from US Geological Survey (https://www.usgs.gov/core-science-systems/ngp/board-on-geographic-names) for the USA and GeoNames (http://www.geonames.org/about.html) for India as described in the paper and use to run the code.
