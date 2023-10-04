# Bhugol
A novel data-driven spatially-aware algorithm, Bhugol, that leverages the spatial patterns and the spatial context of place names to automatically geocode the non-gazetteer place names. 


######################### Data #########################################

The following folders contain the datasets used in this study for both the USA and India.

Corpus: Copora of news reports for the USA and India. Note that due to large size, the larger corpus used to compute the co-occurrence statistic for NGPs in the algorithm is not inlcuded in this repository. However, the folder contains news reports that contain all the NGPAs and NGPs used in the research (The text in the uploads is raw and requires some preprocessing before using them). If needed, the large corpus can be created from the newspapers using the source URL mentioned in GDELT https://www.gdeltproject.org/ for USA. Please use the news reports from the year 2020 and 2021, respectively for the USA. For India, the large corpus can be created easily using the news reports from the Times of India (timesofindia.indiatimes.com/), The Hindu (https://www.thehindu.com/), The Pioneer (https://www.dailypioneer.com/), Economic Times (https://economictimes.indiatimes.com/), Assam Tribune (https://assamtribune.com/), The Kashmir Oberver (https://kashmirobserver.net/), and Incredible Orissa (https://incredibleorissa.com/) for the year 2015.

Frequent Sufixes: List of place names from the USA and India along with their suffix, Lat, Long. This is used to compute Ripley's K and Morans I. 

GroundTruth: This contains the four ground truth datasets: NGPA-USA (csv files for ville, town, hill, wood suffixes), NGP-USA (csv file names NPLatLongUSA), NGPA-India (csv files for pur, adi, gaon, patti, palle), and NGP-India (csv file named NPLatLongIndia).

ShapeFiles: This contains the shape files required for statistical analysis of patterns in place names. The shape files were created using the gazetteers for the respective countries and ArcGIS.

SuffixesHDBSCANCLUSTERS: It includes the clusters and their centroids obtained for frequent suffixes. ArcGIS Pro was used to create clusters of place names with frequent suffixes using HDBSCAN algorithm for both the USA and India.


######################### Named Entity Recognition (NER) System ##############

NER system identifies locations (i.e., place names) mentioned in text and is used in this study to extract cooccurring place names for a given non-gazetteer place name. The following NER systems are used in this study.

Stanford NER: NER used to extract cooccurring place names from text for non-gazetteer places for the USA.
SANS: NER used to extract cooccurring place names from text for non-gazetteer places for India. In order to run the code for NER, please download the gazetteer for India as described in the 'NOTE' section (below) of this readme text.

Both the NER systems are stored in 'NERFiles' folder. Note that SANS was developed in our previous research (Sharma, P., Samal, A., Soh, LK. et al. A spatially-aware algorithm for location extraction from structured documents. Geoinformatica 27, 645–679 (2023). https://doi.org/10.1007/s10707-022-00482-1). Please refer to the paper for details about the NER system.


####################### Geocoders used for comparison ##################

The codes and implementation details of the current state-of-the-art geocoders are available from the authors of EUPEG at https://github.com/geoai-lab/EUPEG (Edinburg, TopoCluster, Clavin, and CamCoder) and the code and details for Mordecai geocoder is availabe at https://github.com/openeventdata/mordecai.

###################### Bhugol #########################################

The codes for the statistical analysis of patterns in place names and the different approaches of geocoding in Bhugol are included in the folder 'Code'. 
The code can be run after installing the required libraries and providing the correct path to the datasets. The code is commented for better readability.

###################### NOTE ###############################################
The gazetteers used in this research for the USA and India are not included in this repository due to their large size for GitHub. However, you can easily download them from the following sources:

For the USA, you can obtain the gazetteer data from the US Geological Survey (https://www.usgs.gov/core-science-systems/ngp/board-on-geographic-names).

For India, you can access gazetteer and census data from GeoNames (http://www.geonames.org/about.html) and the official census website (https://censusindia.gov.in/census.website/), respectively.

Please note that the gazetteers from the US Geological Survey and Geonames are regularly updated and may offer more comprehensive information than what was utilized during the development of this algorithm.
