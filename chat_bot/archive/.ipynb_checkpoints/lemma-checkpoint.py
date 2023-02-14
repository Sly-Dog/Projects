### standard ###
import itertools
import json 
import re
import numpy as np
from joblib import Parallel, delayed


#### site-packages ###
import pandas as pd
from pymystem3 import Mystem
from nltk.corpus import stopwords
import gensim
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, preprocessing
from sklearn import linear_model, metrics
from sklearn.neighbors import KNeighborsClassifier

'''Class for properties column preprocessing'''
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
class properties_prerpoc():
    def __init__(self, ser):
        self.ser = ser

    def show_ser(self):
        return self.ser
        
    '''Get dictionaries from suitable string  '''
    def get_dict(string):
        try: result = eval(string) # .values()
        except: result = None # urls also will be here
        
        return result
    
    '''Get series from prepared Series with dictionaries''' ###############################################DFDSFSDFSDFSFSDFSDFSDFSDFSDFSD
    def get_series_of_dicts(self):
        for i in self.ser.index:
            self.ser.at[i] = get_dict(self.ser[i])
    
    '''Get all keys and values from prepared Series with dictionaries'''
    def get_all_keys_and_values(self):
        clean_properties = self.ser.dropna()  #

        clean_properties_keys = []
        clean_properties_values = []

        for dct in clean_properties.values:
            clean_properties_keys += dct.keys()
            clean_properties_values += dct.values()

        clean_properties_keys = pd.Series(clean_properties_keys)
        clean_properties_values = pd.Series(clean_properties_values)
        
        return clean_properties_keys,clean_properties_values
    
    ''' Make Series with lists of keys and values 
        from prepared Series with dictionaries    '''
    def compress_expand_lists_to_ser(self):
        properties_key_lists = self.ser.copy()
        properties_value_lists = self.ser.copy()

        for i in self.ser.index:
            try:
                properties_key_lists[i] = list(properties_key_lists[i].keys())
                properties_value_lists[i] = list(properties_value_lists[i].values())
            except:
                properties_key_lists[i] = None
                properties_value_lists[i] = None
            
        return properties_key_lists, properties_value_lists
    
    '''Replace string "duno" with ["duno"]''' ##################################### Не робит, ибо не dunno, а ненаю
    def replace_dunno_with_list(self, column):
        for i in column.index:
            name = column.name
            if df[name][i] == "dunno":
                df.loc[:, name][i] = ["dunno"]



#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
'''Get dictionaries from suitable string  '''
def get_dict(string):
    try: result = eval(string) # .values()
    except: result = None # urls also will be here
    return result
#___________________________________________________#



"""Class for preprocessing especially for model"""
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
class model_prerpoc():
    def __init__(self, ser):
        self.ser = ser
    
    '''Transforms series to one string without stop-words and garbage symbols'''
    def prepearing_for_lemmatization(self):
        series = self.ser
        
        if series.dtype == object:
            
            ## making one big string
            series = series.apply(lambda x: str(x)[1:-1])
            union_string = series.str.cat(sep="`<>`")
            
            ##replace some garbage symbols
            union_string = union_string.replace("\n", " ")
            union_string = union_string.replace("?", " ")
            union_string = union_string.replace(")", " ")
            union_string = union_string.replace("(", " ")
            union_string = union_string.replace("!", " ")
            union_string = union_string.replace(".", " ")
            union_string = union_string.replace(",", " ")
            union_string = union_string.replace("\'", " ")
            union_string = union_string.replace("'", " ")
            union_string = union_string.replace("   ", " ")
            union_string = union_string.replace("  ", " ")
            
            return union_string
        
        # union descrirption into one string
        
        union_string = series.str.cat(sep="`<>`")

        '''replace stopwords except "не"'''
        #‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
        stop = stopwords.words("russian")
        stop.remove("не")

        union_string = re.sub(
            "\\s" + " ".join(stop).replace(' ','\s|\s'),
            " ", union_string)
        
        union_string = re.sub("не\s", "не", union_string)     # не !!!
        #______________________________________________________#

        '''replace some garbage symbols'''
        #‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾# 
        union_string = union_string.replace("\n", " ")
        union_string = union_string.replace("?", " ")
        union_string = union_string.replace(")", " ")
        union_string = union_string.replace("(", " ")
        union_string = union_string.replace("!", " ")
        union_string = union_string.replace(".", " ")
        union_string = union_string.replace(",", " ")
        union_string = union_string.replace("\'", " ")
        union_string = union_string.replace("'", " ")
        union_string = union_string.replace("   ", " ")
        union_string = union_string.replace("  ", " ")
        
        return union_string

    '''lemmatization with pymystem3 and parallelizm'''
    def lemmatization(self, text_for_mystem):
        # split large string to 6 lists with smaller strings for parallel lemmatization
        arrays = pd.Series(np.array_split(text_for_mystem.split('`<>`'), 6))

        ## concatenate strings in arrays with separator for future split
        union_string_6 = []
        for array in arrays:
            union_string_6.append(pd.Series(array).str.cat(sep="`<>`"))


        ## here is parallel lemmatization. 40 seconds duration
        m = Mystem().lemmatize
        results = Parallel(n_jobs=-1)(delayed(m)(x) for x in union_string_6)

        ## replace pointless tabulation
        strings = []
        for array in results:
            text = " ".join(array).replace("   ", " ").replace("  ", " ")
            text = re.sub("не\s", "не", text) ### !  join "не"
            strings.append(text)

        ## split strings to arrays
        lemma_array = []
        for string in strings:
            lemma_array.append(string.split("`<>`"))

        output = []
        for lst in lemma_array:
            for string in lst:
                output.append(string)

        lemma_series = pd.Series(output)
        
        return lemma_series
    
##################### MODEL PREPROCESSING #####################################
###############################################################################

topics = []

with open("Otvety.txt", "r", encoding="utf-8") as f:
    next(f)
    co = -1
    for i, row in enumerate(f):
        if row == "---\n": 
            topics.append([])
            co += 1
        else:
            topics[co].append(row)

questions = []
answers   = []

for lst in topics:
    questions.append(lst[0])
    answers.append(lst[1])
    
del topics

###############################################################################
###############################################################################

qa = pd.Series(questions, answers)

qa_preproc = model_prerpoc(qa)
qa_text = qa_preproc.prepearing_for_lemmatization()


###############################################################################

arrays = pd.Series(np.array_split(qa_text.split('`<>`'), len(qa) // 20000))   ## Параллелизовать эту суку!!!


# %%time

lemma_array = []


for arr in arrays[0][:1]:
    step = 6


    lst6 = [[], [], [], [], [], []]

    r = step
    l = 0
    for _ in range (len(arr) // step):
        for i, item in enumerate(arr[l:r]):
            if   i%6 == 0: lst6[5].append(item)
            elif i%5 == 0: lst6[4].append(item)
            elif i%4 == 0: lst6[3].append(item)
            elif i%3 == 0: lst6[2].append(item)
            elif i%2 == 0: lst6[1].append(item)
            elif i%1 == 0: lst6[0].append(item)

        r += step
        l += step

    if len(arr) % 6 != 0: lst6[0] += list(arr[l:])


    union_string_6 = []
    for lst in lst6:
        union_string_6.append(pd.Series(lst).str.cat(sep="`<>`")) ### Возможно замедляет 
    
    from pymystem3 import Mystem
    m = Mystem().lemmatize
    results = Parallel(n_jobs=-1)(delayed(m)(x) for x in union_string_6)

    for result in results:
        ser = pd.Series(result)
        ser.to_csv('questions.csv', index=False, header=False, mode="a")
    
# Ну ещё можно чтобы было 46 ячеек подряд.... не 