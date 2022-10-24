# -*- coding: utf-8 -*-
"""
Bhugol Integrated Approach NGP with Frequent Suffixes
"""
import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
import joblib
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import operator #for dict.iteritems to obtain the key of the dict with maximum value

#All the supporting functions
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

def getGlobalCooccuringplaces(corpus, nongazplc,NER,placeNames):
    """
    Extract all the gazetteer place names that cooccur with a given non-gazetteer place name in the corpus.
    """
    sansplc = []
    for report in corpus:
        if nongazplc in report:
            if nongazplc in report:
                sansplc  = sansplc + get_locs_NER(report, NER, placeNames)

    return(sansplc)

def getCoordinatesofCooccurringPlaces(cooccurringplaces, gaz_placenames, df_gaz):
    """
    Get the geocordinates of all the co-occurring gazetteer place names. For toponyms with same name but different
    geocodes, we use the geocodes of the toponyms that are atleast 161 km apart from each other. Since we compiled the
    gazetteer using different sources, a toponym has multiple entries in the gazetteer with very very small
    difference in the geocodes. This is just a pre-processing step.
    """
    cooccuringplaces_lats = []
    cooccuringplaces_longs = []
    cooccuringplaces_latlong = []
    allplaces = []
    
    #Get the geocoordinates of the cooccuring places
    for place in cooccurringplaces:
        if (place.title() not in gaz_placenames or place == "India"):
            continue
        index = df_gaz.index[df_gaz['Placename'] == place.title()]
        lats = [df_gaz['Lat'][k] for k in index]
        longs = [df_gaz['Long'][k] for k in index]
        names = [df_gaz['Placename'][k] for k in index]
        
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

def getClusterSimilarityScore(clust,cluster_id,clustered_places,clusteredplaces_lats, clusteredplaces_longs,df_centroid, cooccuringplaces):
    """
    Calculate the cluster similarity between clusters for co-occuring place names and the suffixes.
    """
    lats = [clusteredplaces_lats[i] for i in range(0,len(clusteredplaces_lats)) if cluster_id[i] == clust]
    longs = [clusteredplaces_longs[i] for i in range(0,len(clusteredplaces_longs)) if cluster_id[i] == clust]
    places = [clustered_places[i] for i in range(0,len(clusteredplaces_longs)) if cluster_id[i] == clust]
    # get the centroids of all the cooccurring clusters
    centralplc,centralplc_lat, centralplc_long = getCentralFeature(places, lats, longs)
    
    # distance computation between centroid of the cooccuring place name clusters to  centroid of the suffix clusters
    dist = [haversineDistance(centralplc_long, centralplc_lat, row['Long'], row['Lat']) for i, row in df_centroid.iterrows()]
    normalized_dist = [dist[i]/max(dist) for i in range(0, len(dist))]

    
    normalized_clusteritems = 1 - (len(lats)/len(cooccuringplaces))
    
    overallscore = min(normalized_dist) + (normalized_clusteritems)
    
    return(overallscore)

def getPlaceSimilarityScore(corpus,nongazplc,gazplc, gazplcindex, allplaces,cooccuringplaces_lats,cooccuringplaces_longs, df_centroid, cooccuringplaces):
    """
    Calculate the similarity between co-occuring place names so that we can associate them to the clusters for frequent suffixes.
    """
    #get the Normalized Cooccurrence Score
    ncs = getNormalizedCooccurrenceScore(corpus,nongazplc,gazplc)
    
    #get the lats and longs for all other cooccurring places except the nongazplc to compute the cumulative distance
    lats = [cooccuringplaces_lats[i] for i in range(0, len(allplaces))]
    longs = [cooccuringplaces_longs[i] for i in range(0, len(allplaces))]
    
    #Cumulative Normalized Cooccurring distance computaion
    dist_cooccurring = [haversineDistance(longs[gazplcindex], lats[gazplcindex], longs[i], lats[i]) for i in range(0, len(lats)) if i != gazplcindex]
    normalized_dist_cooccurring = [dist_cooccurring[i]/max(dist_cooccurring) for i in range(0, len(dist_cooccurring))]
    cumulativenormalized_dist_cooccurring = sum(normalized_dist_cooccurring)/len(cooccuringplaces)
    
    # Normalized Distance Computation
    dist_suffix = [haversineDistance(longs[gazplcindex], lats[gazplcindex], row['Long'], row['Lat']) for i, row in df_centroid.iterrows()]
    normalized_dist_suff = [dist_suffix[i]/max(dist_suffix) for i in range(0,len(dist_suffix))]
    minnormalized_dist_suff = min(normalized_dist_suff)
    
    overallscore = (1-ncs) + cumulativenormalized_dist_cooccurring + minnormalized_dist_suff
    
    return(overallscore)

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


#Functions for SANS i.e., NER for India

def has_loc_suffix(word):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    suffix = ['pur', 'adi', 'iya', 'gaon', 'orest', 'ani', 'patti', 'palle', 'khurd', 'purwa', 'dih', 'chak', 'minor', 'garh', 'singh', 'uru', 'palem', 'ain', 'ganj', 'anga', 'and', 'padu', 'uzurg', 'utary', 'pet', 'attu', 'ane', 'angi', 'kh.', 'bk.'] #most common suffixes (top 30) obtained from suffix extractor
    for suf in suffix:
        if word.lower().endswith(suf) and word.lower() != suf:
            return(True)
    return(False)

def has_loc_domainwords(word, postagprev):
    """
    Identify if a given word has a domain word. SANS helper function.
    """
    domainwords = ['nagar','colony','street','road','hill','river','temple','village','sector', 'district', 'taluk', 'town', 'mutt', 'fort', 'masjid', 'church']
    for entry in domainwords:
        if word.lower() == entry:
            if postagprev in ['NNP','NNPS']:
                return(True)
    return(False)

def has_prep(word, postagnext):
    """
    Identify if a given word has a suffix. SANS helper function.
    """
    preps = ['near', 'via', 'in', 'from', 'between', 'at', 'versus', 'like', 'towards', 'of', 'toward', 'across'] # Place name prepositions with location likelihood scores greater than 0.1
    for prep in preps:
        if word.lower() == prep:
            if postagnext in ['NNP','NNPS']:
                return(True)
    return(False)

def get_wordshape(word):
    """
    Identify the shape of a given word. SANS helper function.
    """
    shape1 = re.sub('[A-Z]', 'X',word)
    shape2 = re.sub('[a-z]', 'x', shape1)
    return re.sub('[0-9]', 'd', shape2)

def is_in_gazetteer(word, postag, placeNames):
    """
    Identify if a given word is in the gazetteer. SANS helper function.
    """
    if postag in ['NNP', 'NNPS'] and word in placeNames:
        return True
    return False

def word2features(sent, i, placeNames):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.shape':get_wordshape(word),
        'hassuffix:':has_loc_suffix(word),
        'is_in_gazetteer:':is_in_gazetteer(word, postag, placeNames),
        'wordallcap': len([x for x in word if x.isupper()])==len(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:prep': has_prep(word, postag1),
            '+1:hasdomain':has_loc_domainwords(word1,postag),
            '+1:wordallcap': len([x for x in word1 if x.isupper()])==len(word1),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent, placeNames):
    return [word2features(sent, i, placeNames) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def get_locs_NER(doc, NER, placeNames):
    output = get_NER_5WNER(sent_tokenize(doc), NER, placeNames) #sentence tokenize the document and pass it as parameter to get_NER_5WNER to get the NER tags
    locations = concat_placenames(output)
    
    list_set = set(locations) 
    
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
                # If the next words are also locations, then concatenate them to make a longer string. This is useful for place names with multiple words. e.g., New Delhi
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


def get_NER_5WNER(doc, NER, placeNames):
    """
    Use SANS to identify location names in text.
    """
    input_ner = []
    tags = []
    for sent in doc:
        text_tokens = pos_tag(word_tokenize(sent))
        input_ner.append(sent2features(text_tokens,placeNames)) #create input to the 5WNER tagger from the text file
        tags.extend(word_tokenize(sent))
    output_ner = (NER.predict(input_ner))
    ner_list = [item for sublist in output_ner for item in sublist] #convert the list of list i.e. output_ner which contains all the NER tags for each sentence of the report
    ner_tags = [(w,t) for w,t in zip(tags, ner_list)] #give output same as that of Stanford NER [(word, NER tag)]
    return(ner_tags)

def main():
    # Load SANS
    SANS = joblib.load('/BhugolCode/NERFiles/SANS/crfNER895.pkl')
    
    df_gaz = pd.read_csv('/BhugolCode/Data/Gazetteers/India/GazetteerIndia.csv', encoding="latin-1")
    df_gaz = df_gaz.drop_duplicates()
    gaz_placenames = df_gaz['Placename'].to_list()
    gaz_placenames = [str(place).strip() for place in gaz_placenames]
    gaz_placenames = [re.sub('[^A-Za-z0-9]+', ' ', place) for place in gaz_placenames]
    placeNames = gaz_placenames #for SANS to identify placenames
    placeNames.sort() # Used to create feature related to whether a word is present in the gazetteer the SANS
    
    # Load Corpus for India
    corpus = []
    
    newspapers = ['TimesOfIndia', 'TheHindu', 'ThePioneer', 'EconomicTimes', 'AssamTribune', 'KashmirObserver', 'IncredibleOrissa']
    
    for newspaper in newspapers:
        df_15 = pd.read_csv('/BhugolCode/Data/Corpus/India/' + newspaper +'2015.csv')
        for j,row in df_15.iterrows():
            title = str(row['Title'])
            index = str(title).find('|') #Times of India includes an extra information from Title. So, remove it
            if index == -1:
                    title = title
            else:
                title = title[:index]
            row_content = str(row['Content'])
            row_title_content = str(title) + ' ' + row_content
            row_title_content = re.sub('download the times of india news app for latest city news.', '', row_title_content)
            row_title_content = re.sub('Be Part of Quality Journalism Quality journalism takes a lot of time, money and hard work to produce and despite all the hardships we still do it. Our reporters and editors are working overtime in Kashmir and beyond to cover what you care about, break big stories, and expose injustices that can change lives. Today more people are reading Kashmir Observer than ever, but only a handful are paying while advertising revenues are falling fast. CLICK FOR DETAILS', '', row_title_content)
            corpus.append(row_title_content)
    

    MAX_ERROR = 20039 # Maximum possible distance between any two point on earth. Used to compute AUC by using Trapeziodal rule.
    suffixes = ['pur', 'adi', 'gaon', 'patti', 'palle']
    for suff in suffixes:
        
        countNotCoded = 0 #counter to count the number of places not geocoded by Bhugol for each suffix
        df_centroid = pd.read_csv('/BhugolCode/Data/SuffixesHDBSCANClusters/India/' + suff + 'clustercentroid.csv')
        groundtruth = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Placename'].tolist()
        groundtruth_lats = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Lat'].tolist()
        groundtruth_longs = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Long'].tolist()
       
        pred_lats = []
        pred_longs = []
    
        clusterer = DBSCAN(eps=50/6371, min_samples=2, algorithm='auto', metric='haversine')
        for place in groundtruth:

            cooccuringplaces = getGlobalCooccuringplaces(corpus,place, SANS,placeNames)
            cooccuringplaces = [place.title() for place in cooccuringplaces]
            cooccuringplaces = list(set(cooccuringplaces))
            
            #Get the geocoordinates of the cooccuring places
            allplaces, cooccuringplaces_lats, cooccuringplaces_longs, cooccuringplaces_latlong = getCoordinatesofCooccurringPlaces(cooccuringplaces,gaz_placenames,df_gaz)

            pts=np.radians(cooccuringplaces_latlong)
    
            # When there are no cooccurring places, use the geographic center to geocode.
            if len(pts) == 0:
                pred_lats.append(20.5937)
                pred_longs.append(78.9629)
                countNotCoded = countNotCoded + 1
                continue
    
            # When there are cooccurring places
            clusterer.fit(pts)
            cluster_id = [clusterer.labels_[i] for i in range(0, len(clusterer.labels_)) if clusterer.labels_[i] != -1]
            clustered_places = [allplaces[i] for i in range(0, len(clusterer.labels_)) if clusterer.labels_[i] != -1]
            clusteredplaces_lats = [cooccuringplaces_lats[i] for i in range(0, len(clusterer.labels_)) if clusterer.labels_[i] != -1]
            clusteredplaces_longs = [cooccuringplaces_longs[i] for i in range(0, len(clusterer.labels_)) if clusterer.labels_[i] != -1]

    
            # When no clusters are formed from the cooccurring places, find the best cooccurring place names to geocode 
            if len(cluster_id) == 0:
    
                placesuitabilityscore = {str(cooccuringplaces_lats[i]) + ', ' + str(cooccuringplaces_longs[i]):getPlaceSimilarityScore(corpus, place, 
                                                                allplaces[i], i, allplaces, cooccuringplaces_lats,
                                                                cooccuringplaces_longs, df_centroid, cooccuringplaces)
                                       for i in range(0,len(allplaces))} #since toponyms with same names will result in duplicate keys for the dictionary which gives error, lat and long of each place is used instead
    
                mostsuitableplace = min(placesuitabilityscore.items(), key=operator.itemgetter(1))[0]

                pred_lats.append(float(mostsuitableplace.split(', ')[0]))
                pred_longs.append(float(mostsuitableplace.split(', ')[1]))

                continue
            
            # When clusters are formed from the cooccurring places, find the best cluster to geocode
            clustersuitabilityscore = {clust:getClusterSimilarityScore(clust,cluster_id,clustered_places,clusteredplaces_lats, 
                                                                       clusteredplaces_longs, df_centroid,cooccuringplaces)
                                       for clust in np.unique(cluster_id)}
    
            mostsuitablecluster = min(clustersuitabilityscore.items(), key=operator.itemgetter(1))[0]
    
            lats = [clusteredplaces_lats[i] for i in range(0,len(clusteredplaces_lats)) if cluster_id[i] == mostsuitablecluster]
            longs = [clusteredplaces_longs[i] for i in range(0,len(clusteredplaces_longs)) if cluster_id[i] == mostsuitablecluster]
            places = [clustered_places[i] for i in range(0,len(clusteredplaces_longs)) if cluster_id[i] == mostsuitablecluster]
    
            # get the centroids of all the cooccurring clusters
            centralplc, centralplc_lat, centralplc_long = getCentralFeature(places, lats, longs)
    
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