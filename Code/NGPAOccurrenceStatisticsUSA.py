# -*- coding: utf-8 -*-
"""
Non-gazetteer place name with frequent suffixes occurrence statistics for the USA
"""

import pandas as pd
import re

# corpus for the USA
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
    df = pd.read_csv("/BhugolCode/Data/Corpus/USA/US_2021_articles-" + str(i) + ".csv")
    for j,row in df.iterrows():
        # Preprocessing the text
        title = str(row['Title'])
        row_content = str(row['Text'])
        row_content = re.sub(r'[.]+(?![0-9])', r'. ', row_content)
        row_content = re.sub('[^A-Za-z0-9.-:,!?\'\'\"\"()%]+', ' ', row_content)
        row_title_content = str(title) + ' ' + row_content
        corpus.append(row_title_content)
        

suffixes = ['ville', 'town', 'hill', 'wood']
for suff in suffixes:
    print('.................................................')
    print(suff)
    groundtruth = pd.read_csv('/BhugolCode/Data/GroundTruth/USA/' + suff + 'latlongUSA.csv')['Placename'].tolist()
    
    count = 0
    for plc in groundtruth:
        for news in corpus:
            if plc in news:
                count = count + 1
    print("Frequeny: {}".format(count))
    print("Total Place: {}".format(len(groundtruth)))
    print("Average: {}".format(count/len(groundtruth)))