# -*- coding: utf-8 -*-
"""
Bhugol Spatio-Contextual Approach NGP with Frequent Suffixes
"""

import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger

#All the functions

def haversineDistance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def getCentralFeature(places, lats, longs):
    """
    Calculate the Centroid of a spatial cluster. Spatial clusters have an arbitrary shape. So, centroid in case of spatial clusters
    do not necessarily mean the center. In this case, we use the point within the cluster with the least cumulative distance
    between the point and all other points in the custer.
    """
    centralplace = ''
    centrallat = 0
    centrallong = 0
    min_dist = 100000000000000000
    for i in range(0, len(places)):
        dist = 0
        for j in range(0, len(places)):
            if i != j:
                dist = dist + haversineDistance(longs[i],lats[i], longs[j], lats[j])
        if dist < min_dist:
            min_dist = dist
            centralplace = places[i]
            centrallat = lats[i]
            centrallong = longs[i]
            
    return(centralplace, centrallat, centrallong)

def binary_search(arr, low, high, x): 
    """
    Implementation of a binary search
    """
    # Check base case 
    if high >= low:          
        mid = (high + low) // 2
        if arr[mid] == x: 
            return True 
  
        # If element is smaller than mid, then it can only 
        elif arr[mid] > x: 
            return binary_search(arr, low, mid - 1, x) 
  
        # Else the element can only be present in right subarray 
        else: 
            return binary_search(arr, mid + 1, high, x) 
  
    else: 
        # Element is not present in the array 
        return False

def getGlobalCooccuringplaces(corpus, nongazplc,NER):
    """
    Extract all the place names that cooccur with a given non-gazetteer place name in the corpus.
    """
    nerplc = []
    for report in corpus:
        if nongazplc in report:
            if nongazplc in report:
                nerplc  = nerplc + get_locs_NER(report, NER)
    return(nerplc)

def getCoordinatesofCooccurringPlaces(cooccurringplaces, gaz_placenames, df_gaz):
    """
    Get the geocordinates of all the co-occurring gazetteer place names. For toponyms with same name but different
    geocodes, we use the geocodes of the toponyms that are atleast 161 km apart from each other. Since a toponym has multiple entries in the gazetteer with very very small
    difference in the geocodes.
    """
    cooccuringplaces_lats = []
    cooccuringplaces_longs = []
    cooccuringplaces_latlong = []
    allplaces = []
    #Get the geocoordinates of the cooccuring places
    for place in cooccurringplaces:
        if (place.title() not in gaz_placenames or place == "USA"):
            continue
        index = df_gaz.index[df_gaz['FEATURE_NAME'] == place.title()]
        lats = [df_gaz['PRIM_LAT_DEC'][k] for k in index if df_gaz['PRIM_LAT_DEC'][k] != 0 and df_gaz['PRIM_LONG_DEC'][k] != 0 and df_gaz['STATE_ALPHA'][k] not in ['PR','AK', 'AS', 'GU','HI','CM','TT']] #some entries in gazetteer for the USA have zero as their geocodes. We do not take such place names. Also, our region of interest is the contiguous USA
        longs = [df_gaz['PRIM_LONG_DEC'][k] for k in index if df_gaz['PRIM_LAT_DEC'][k] != 0 and df_gaz['PRIM_LONG_DEC'][k] != 0 and df_gaz['STATE_ALPHA'][k] not in ['PR','AK', 'AS', 'GU','HI','CM','TT']]
        names = [df_gaz['FEATURE_NAME'][k] for k in index if df_gaz['PRIM_LAT_DEC'][k] != 0 and df_gaz['PRIM_LONG_DEC'][k] != 0 and df_gaz['STATE_ALPHA'][k] not in ['PR','AK', 'AS', 'GU','HI','CM','TT']]

        for i in range(0, len(names)):
            lessthan161 = False
            for j in range(0 ,len(allplaces)):
                dist = (haversineDistance(longs[i], lats[i], 
                                             cooccuringplaces_longs[j], cooccuringplaces_lats[j]))
                if (dist < 161 and allplaces[j] == names[i]):
                    lessthan161 = True

            if lessthan161 == False:
                allplaces.append(names[i])
                cooccuringplaces_lats.append(lats[i])
                cooccuringplaces_longs.append(longs[i])
                cooccuringplaces_latlong.append([lats[i],longs[i]])    
    return (allplaces, cooccuringplaces_lats, cooccuringplaces_longs, cooccuringplaces_latlong)


def getNormalizedCooccurrenceScore(corpus,nongazplc,gazplc):
    """
    Compute the normalized co-occurrence score for a given non-gaz place name.
    """
    countnongazplc = 0
    countgazplc = 0
    countgaznongazplc = 0
    for report in corpus:
        if nongazplc in report and gazplc in report:
            countgaznongazplc += 1
            countgazplc += 1
            countnongazplc += 1
        elif nongazplc in report:
            countnongazplc += 1
        elif gazplc in report:
            countgazplc += 1
    return(countgaznongazplc/countgazplc)

# Funciton for Stanford NER    
def get_locs_NER(doc, NER):
    output = get_NER_5WNER(sent_tokenize(doc), NER) #sentence tokenize the document and pass it as parameter to get_NER_5WNER to get the NER tags
    locations = concat_placenames(output)
    
    list_set = set(locations) # get unique place names that co-occur in corpus with the non-gazetteer place name
    
    # convert the set to the list 
    unique_locs = (list(list_set))
    return unique_locs

def concat_placenames(original_tags):
    """
    Combine names of the locations if the locations consists of two words eg. New Delhi
    """
    locations = []
    l = len(original_tags)
    i=0;
    # Iterate over the tagged words.
    while i<l:
        e,t = original_tags[i]
        # If it's a location, then check the next words.
        if t == 'LOCATION':
            j = 1
            s = e
            # Verify the tags for the next words.
            while i+j<len(original_tags):
                # If the next words are also locations, then concatenate them to make a longer string. This is useful for place names with multiple words. e.g., New York
                if original_tags[i+j][1] == 'LOCATION':
                    s = s+" "+original_tags[i+j][0]
                    j+=1
                else:
                    break
            i = i+j
            # Save the locations to a locations list
            locations+=[s]
        else:
            i=i+1

    return locations


def get_NER_5WNER(doc, st):
    """
    Use Stanford NER to identify location names in text.
    """
    ner_tags = []
    tags = []
    for sent in doc:
        tags = st.tag(word_tokenize(sent))
        ner_tags.extend(tags)
    return(ner_tags)

def main():
    # Load Stanford NER
    st = StanfordNERTagger('/BhugolCode/NERFiles/stanford_NER/ner-model-english.ser.gz',
      '/BhugolCode/NERFiles/stanford_NER/stanford-ner.jar',
       encoding='utf-8')
    
    df_gaz = pd.read_csv('/BhugolCode/Data/Gazetteers/USA/GazetteerUSA.csv', encoding="utf8")
    gaz_placenames = df_gaz['FEATURE_NAME'].tolist()
    gaz_placenames = [str(place).strip() for place in gaz_placenames]
    gaz_placenames = [re.sub('\s|\([^()]*\)', ' ', place) for place in gaz_placenames]
    gaz_placenames = [re.sub('[^A-Za-z0-9]+', ' ', place) for place in gaz_placenames]
    
    # Load Corpus for USA
    corpus = []
    for i in range(1,31):
        df = pd.read_csv('/BhugolCode/Data/Corpus/USA/US_2020_articles-' + str(i) + ".csv")
        for j,row in df.iterrows():
            title = str(row['Title'])
            row_content = str(row['Text'])
            row_content = re.sub(r'[.]+(?![0-9])', r'. ', row_content) #re.sub(r"\.(?!\s|$)", ". ", row_content)
            row_content = re.sub('[^A-Za-z0-9.-:,!?\'\'\"\"()%]+', ' ', row_content)
            row_title_content = str(title) + ' ' + row_content
            corpus.append(row_title_content)
            
    for i in range(0,15):
        df = pd.read_csv('/BhugolCode/Data/Corpus/USA/US_2021_articles-' + str(i) + '.csv')
        for j,row in df.iterrows():
            title = str(row['Title'])
            row_content = str(row['Text'])
            row_content = re.sub(r'[.]+(?![0-9])', r'. ', row_content) #re.sub(r"\.(?!\s|$)", ". ", row_content)
            row_content = re.sub('[^A-Za-z0-9.-:,!?\'\'\"\"()%]+', ' ', row_content)
            row_title_content = str(title) + ' ' + row_content
            corpus.append(row_title_content)
    
    MAX_ERROR = 20039 # Maximum possible distance between any two point on earth. Used to compute AUC by using Trapeziodal rule.
    suffixes = ['ville', 'town', 'hill', 'wood']
    for suff in suffixes:
        countNotCoded = 0 #counter to count the number of places not geocoded by Bhugol for each suffix
        groundtruth = pd.read_csv('/BhugolCode/Data/GroundTruth/USA/' + suff + 'latlongUSA.csv')['Placename'].tolist()
        groundtruth_lats = pd.read_csv('/BhugolCode/Data/GroundTruth/USA/' + suff + 'latlongUSA.csv')['Lat'].tolist()
        groundtruth_longs = pd.read_csv('/BhugolCode/Data/GroundTruth/USA/' + suff + 'latlongUSA.csv')['Long'].tolist()
        
        pred_lats = []
        pred_longs = []
    
        clusterer = DBSCAN(eps=50/6371, min_samples=2, algorithm='auto', metric='haversine')
        for place in groundtruth:
            cooccuringplaces = getGlobalCooccuringplaces(corpus,place,st)
            cooccuringplaces = [plc.title() for plc in cooccuringplaces if plc != place]
            cooccuringplaces = list(set(cooccuringplaces))
            
            #Get the geocoordinates of the cooccuring places
            allplaces, cooccuringplaces_lats, cooccuringplaces_longs, cooccuringplaces_latlong = getCoordinatesofCooccurringPlaces(cooccuringplaces,gaz_placenames,df_gaz)
    
            pts=np.radians(cooccuringplaces_latlong)
            if len(pts) == 0: #if no co-occurring place name exist in gazetteer, the algorithm cannot geocode the non-gazetteer place names. So, the geographic centroid of USA is used to geocode.
                pred_lats.append(39.8283)
                pred_longs.append(-97.5795)
                countNotCoded = countNotCoded + 1
                continue
            
            if len(pts) == 1: #if only one co-occurring place name exist in gazetteer, the algorithm uses the place name to geocode.
                pred_lats.append(cooccuringplaces_lats[0])
                pred_longs.append(cooccuringplaces_longs[0])
                continue
    
            if len(pts) != 1: # Store the cluster ID along with the place names and their geocordinates in respective arrays
                clusterer.fit(pts)
                cluster_id = []
                clusteredplaces = []
                clusteredplaces_lats = []
                clusteredplaces_longs = []
                for i in range(0, len(clusterer.labels_)):
                    if clusterer.labels_[i] != -1:
                        cluster_id.append(clusterer.labels_[i])
                        clusteredplaces.append(allplaces[i])
                        clusteredplaces_lats.append(cooccuringplaces_lats[i])
                        clusteredplaces_longs.append(cooccuringplaces_longs[i])
    
            
            if len(cluster_id) == 0 or len(cluster_id) == 1: 
                '''
                If no cluster are obtained using DBSCAN, we find the co-occurring place name with the highest co-occurrence score. 
                NOTE: If clusters are not obtained, DBSCAN assigns the label '-1' to all clusters. In that case the length of cluster_id will be 0.
                ''' 
                cooccuringplaces = [plc.title() for plc in cooccuringplaces if plc.title() in gaz_placenames and plc!='USA' and plc.lower() != place.lower()]
                
                if len(cooccuringplaces) == 0: #if not co-occurring place name is present in gazetteer, use the geographic center to geocode.
                    pred_lats.append(39.8283)
                    pred_longs.append(-97.5795)
                    countNotCoded = countNotCoded + 1
                else:
                    # find the co-occurring place name with the highest co-occurrence score
                    maxncs = 0
                    for coplc in cooccuringplaces:
                        score = getNormalizedCooccurrenceScore(corpus,place,coplc)
                        if score  > maxncs:
                            maxncs = score
                            maxcooccuringplace = coplc
    
                    index = df_gaz.index[df_gaz['FEATURE_NAME'] == maxcooccuringplace.title()][0]
                    pred_lats.append(df_gaz['PRIM_LAT_DEC'][index])
                    pred_longs.append(df_gaz['PRIM_LONG_DEC'][index])
                continue
    
            '''
            If more than one clusters of cooccurring place names exist. 
            Use the cluster with the maximum number of co-occurrigng place names to geocode. 
            '''
            maxvotedcluster = max(list(cluster_id), key=list(cluster_id).count)
            cooccurringcluster_lats = []
            cooccurringcluster_longs = []
            cooccurringcluster_placenames = []
            cooccurringcluster_latlong = []
            for i in range(0, len(cluster_id)):
                if cluster_id[i] == maxvotedcluster:
                    cooccurringcluster_lats.append(clusteredplaces_lats[i])
                    cooccurringcluster_longs.append(clusteredplaces_longs[i])
                    cooccurringcluster_placenames.append(clusteredplaces[i])
                    cooccurringcluster_latlong.append([clusteredplaces_lats[i],clusteredplaces_longs[i]])
            
            #Compute the central feature of the max voted cluster and use it to geocode
            centralplc,centralplc_lat, centralplc_long = getCentralFeature(cooccurringcluster_placenames,
                                                                              cooccurringcluster_lats, 
                                                                               cooccurringcluster_longs)
            pred_lats.append(centralplc_lat)
            pred_longs.append(centralplc_long)
        
        dist_km = []
        for i in range(0, len(groundtruth)):
            dist_km.append(haversineDistance(groundtruth_longs[i], groundtruth_lats[i], pred_longs[i], pred_lats[i]))
        
        mean_dist = sum(dist_km)/len(dist_km)
        A161 = (len([dist for dist in dist_km if dist <= 161])/len(dist_km)) * 100
        AUC =  np.trapz(np.log(dist_km)) / (np.log(MAX_ERROR) * (len(dist_km) - 1)) #Using the Trapeziodal rule
        
        
if __name__ == "__main__":
    main()     