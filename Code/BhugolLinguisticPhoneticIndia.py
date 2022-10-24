# -*- coding: utf-8 -*-
"""
Linguistic pattern-based approach of geocoding (PHONETIC)
"""
import pandas as pd, numpy as np
from math import radians, cos, sin, asin, sqrt
from fuzzywuzzy import fuzz
import jellyfish

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

def main():
    suffixes = ['pur', 'adi', 'gaon','patti', 'palle']
    MAX_ERROR = 20039 # Maximum possible distance between any two point on earth. Used to compute AUC by using Trapeziodal rule.
    for suff in suffixes:

        df_centroid = pd.read_csv('/BhugolCode/Data/SuffixesHDBSCANClusters/India/' + suff + 'clustercentroid.csv')
        df_cluster = pd.read_csv('/BhugolCode/Data/SuffixesHDBSCANClusters/India/' + suff + 'cluster.csv')
        groundtruth = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Placename'].tolist()
        groundtruth_lats = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Lat'].tolist()
        groundtruth_longs = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Long'].tolist()
        
        # Find the clusters
        clusters = df_cluster.cluster.unique()
        df_result = pd.DataFrame(columns=['Placename', 'Actual Lat', 'Actual Long','Lat', 'Long', 'Error'])
        parentcluster = []
        max_similarity = []

        for i in range(0, len(groundtruth)):
            max_meansim = 0
            cluster_id = 0
            for ids in clusters:
                similarity = []
                places = df_cluster[df_cluster['cluster'] == ids] ['Placename'].tolist()
                    
                for place in places:
                    if place != groundtruth[i]:
                        similarity. append(fuzz.ratio(jellyfish.nysiis(groundtruth[i]), jellyfish.nysiis(place)))
                if np.mean(similarity) > max_meansim:
                    max_meansim = np.mean(similarity)
                    cluster_id = ids

            parentcluster.append(cluster_id)
            max_similarity.append(max_meansim)
        
        predicted_lats = []
        predicted_longs = []
        
        for i in range(0, len(groundtruth)):
            predicted_lats.append(float(df_centroid[df_centroid['Cluster'] == parentcluster[i]]['Lat']))
            predicted_longs.append(float(df_centroid[df_centroid['Cluster'] == parentcluster[i]]['Long']))
        
        for i in range(0, len(groundtruth)):
            df = pd.DataFrame([[groundtruth[i], groundtruth_lats[i], groundtruth_longs[i], predicted_lats[i], predicted_longs[i], haversineDistance(predicted_longs[i], predicted_lats[i], groundtruth_longs[i],  groundtruth_lats[i])]], columns = df_result.columns)
            df_result = df_result.append(df)
            
        mean = np.mean(df_result['Error'].tolist())
        A161 = (len([dist for dist in df_result['Error'].tolist() if dist <= 161])/len(df_result['Error'].tolist())) * 100
        AUC =  np.trapz(np.log(df_result['Error'].tolist())) / (np.log(MAX_ERROR) * (len(df_result['Error'].tolist()) - 1)) #Using the Trapeziodal rule


if __name__ == "__main__":
    main()