# -*- coding: utf-8 -*-
"""
Non-gazetteer place name with frequent suffixes occurrence statistics for India
"""

import pandas as pd
import re

corpus = []

newspapers = ['TimesOfIndia', 'TheHindu', 'ThePioneer', 'EconomicTimes', 'AssamTribune', 'KashmirObserver', 'IncredibleOrissa']

for newspaper in newspapers:
    df_15 = pd.read_csv('/BhugolCode/Data/Corpus/India/' + newspaper +'2015.csv')
    for j,row in df_15.iterrows():
        #Preprocessing the text.
        title = str(row['Title'])
        index = str(title).find('|') #Title in Times of India newspaper include an extra information from Title. So, remove it
        if index == -1:
                title = title
        else:
            title = title[:index]
        row_content = str(row['Content'])
        row_title_content = str(title) + '. ' + row_content
        row_title_content = re.sub('download the times of india news app for latest city news.', '', row_title_content)
        row_title_content = re.sub('Be Part of Quality Journalism Quality journalism takes a lot of time, money and hard work to produce and despite all the hardships we still do it. Our reporters and editors are working overtime in Kashmir and beyond to cover what you care about, break big stories, and expose injustices that can change lives. Today more people are reading Kashmir Observer than ever, but only a handful are paying while advertising revenues are falling fast. CLICK FOR DETAILS', '', row_title_content)
        corpus.append(row_title_content)
        
suffixes = ['pur', 'adi', 'gaon', 'patti', 'palle']
for suff in suffixes:
    print('.................................................')
    print(suff)
    groundtruth = pd.read_csv('/BhugolCode/Data/GroundTruth/India/' + suff + 'latlongIndia.csv')['Placename'].tolist()
    
    count = 0
    for plc in groundtruth:
        for news in corpus:
            if plc in news:
                count = count + 1
    print("Frequeny: {}".format(count))
    print("Total Place: {}".format(len(groundtruth)))
    print("Average: {}".format(count/len(groundtruth)))