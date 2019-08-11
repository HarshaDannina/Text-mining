# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:23:29 2019

@author: HARSHA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:14:30 2019

@author: HARSHA
"""

import pandas as pd
from pattern.en import sentiment
#import HTMLParser
import re
from collections import Counter
from nltk.corpus import stopwords
import string
from collections import OrderedDict
from nltk import bigrams
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('Nashville.csv')
#print(data.count())
data = data[data.columns[0:6]]

list_data=list(data['Findings'])

from collections import Counter
Counter(list_data)

Cleansed_data=[]
for j in list_data:
    Special_chars = re.sub(r'[\-\!\@\$\%\^\&\*\(\)\_\+\[\]\;\'\#\,\/\{\}\"\<\>\?\|]','',j)
    lower = Special_chars.lower()
    Widespace = lower.strip()
    Cleansed_data.append(Special_chars)
#print(Cleansed_data[0:5])
#print(len(Cleansed_data))


def GetListOfSubstrings(stringSubject,string1,string2):
    MyList = []
    intstart=0
    strlength=len(stringSubject)
    continueloop = 1

    while(intstart < strlength and continueloop == 1):
        intindex1=stringSubject.find(string1,intstart)
        if(intindex1 != -1): #The substring was found, lets proceed
            intindex1 = intindex1+len(string1)
            intindex2 = stringSubject.find(string2,intindex1)
            if(intindex2 != -1):
                subsequence=stringSubject[intindex1:intindex2]
                MyList.append(subsequence)
                intstart=intindex2+len(string2)
            else:
                continueloop=0
        else:
            continueloop=0
    return MyList

feature_names = ['number of sessile','size in number(cm)','size in number(mm)','small','diminutive','semi','medium','large','sessile','pedunculated','flat','mass','serrated','smooth','cecum','ascending','ileum','hepatic','transverse','splenic','sigmoid','rectosigmoid','descending','rectum','appendix','esophagus','left','right', 'cold snare','hot snare','snare cautery','electrocautery snare','snare','biopsy','excisional biopsy','biopsy forcep','cold biopsy','resection','removed','retrieved']
from numpy import nan as Nan
list_data = [[Nan for j in range(len(feature_names))] for i in range(len(Cleansed_data))]
list_mat = {}
data_frames_list = [0,1,2,3,4,5,6,7,8,9]
List = []
s = ','
for k in range(len(data_frames_list)):
    for i in range(len(Cleansed_data)):
        List = GetListOfSubstrings(Cleansed_data[i],"POLYP","Path")
        countvec_subset = CountVectorizer(vocabulary= ['dummy','dummi','dumme','small','diminutive','semi','medium','large','sessile','pedunculated','flat','mass','serrated','smooth','cecum','ascending','ileum','hepatic','transverse','splenic','sigmoid','rectosigmoid','descending','rectum','appendix','esophagus','left','right', 'cold snare','hot snare','snare cautery','electrocautery snare','snare','biopsy','excisional biopsy','biopsy forcep','cold biopsy','resection','removed','retrieved'], lowercase = True , decode_error = 'ignore', binary = True, ngram_range = (1,10))
        
        if  len(List)>0:
            try:
                x = [List[k]]
                token0 = re.findall('one|two|three|four|five|multiple|few|six|seven|eight|nine|single',x[0])
                token1 = re.findall('(-?\d+\.?\d*) cm',x[0])
                token2 = re.findall('(-?\d+\.?\d*) mm',x[0])
                #if len(token0)>0:
                list_data[i][0] = s.join(token0)
                #if len(token1)>0:
                list_data[i][1] = s.join(token1)+str('cm') if s.join(token1) is not '' else Nan
                #if len(token2)>0:
                list_data[i][2] = s.join(token2)+str('mm') if s.join(token2) is not '' else Nan
                counts = countvec_subset.fit_transform(x)
                #print counts
                dense = counts.todense()
                dense.shape
                dense
                data_mat=pd.DataFrame(dense)
                list_values = data_mat.values.tolist()
                for j in range(3,len(list_values[0])):
                    if list_values[0][j] == 1:
                        list_data[i][j] = list_values[0][j]
                    else:
                        list_data[i][j] = Nan
            except IndexError:
                for j in range(len(feature_names)):
                    list_data[i][j] = Nan
        else:
                #data_mat = pd.Series([Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan], index = feature_names)
                #list_mat.append(list(pd.Series([np.nan])), ignore_index = True)
                #list_data[i].append(list(data_mat.values))
                for j in range(len(feature_names)):
                    list_data[i][j] = Nan
    
    
    list_mat[k] = pd.DataFrame(list_data, columns = feature_names)
#print(list_data[9])



final_mat = pd.concat([data['Random_ID'],data['Findings'],list_mat[0],list_mat[1],list_mat[2],list_mat[3],list_mat[4],list_mat[5],list_mat[6],list_mat[7],list_mat[8],list_mat[9]],axis=1)
final_mat.to_csv('test2.csv')
