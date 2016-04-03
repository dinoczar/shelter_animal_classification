# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:51:49 2016

@author: Chris
"""

##############################################################################
##############################################################################
#code for querying animals that have a name and have age less than 500 days
#df[df['Has_Name'] == True][df[df['Has_Name'] == True]['AgeinDays'] <500]

#Useful Functions

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def tokenize(text):
    import re
    REGEX = re.compile(r"/\s*")
    return [tok.strip().lower() for tok in REGEX.split(text)]


def has_name(name):
    """"Checks each entry to see if animal has a name or not."""
    if type(name) == str:
        return True
    else:
        return False


def convert_age(s):
    '''Converts age from a string format (e.g. 1 year) to age 
    in number of days. Returns number as int.'''
    age = 0    
    try:
        data = s.split()
        if data[1] == 'day' or data[1] == 'days':
            age += int(data[0])
        elif data[1] == 'week' or data[1] == 'weeks':
            age += int(data[0])*7
        elif data[1] == 'month' or data[1] == 'months':
            age+= int(int(data[0]) *30.5)
        elif data[1] == 'year' or data[1] == 'years':
            age+= int(data[0])*365
    except AttributeError:
        age = float('nan')
    return age


def day(date_format):
    """Takes input in the form yyyy-mm-dd and returns the day of the week"""    
    import datetime
    d = [int(i) for i in date_format.split('-')]
    d = datetime.date(d[0], d[1], d[2])
    return d.strftime('%a') 

def month(date_format):
    '''Takes string in form yyyy-mm-dd and returns the month as a string'''
    import datetime
    d = [int(i) for i in date_format.split('-')]
    d = datetime.date(d[0], d[1], d[2])
    return d.strftime('%b')

def year(date_format):
    return int(date_format.split('-')[0])

def get_neutered(gender):
    '''Takes argument in the form "gender neutered/spayed" and returns
    whether the animal was neutered or intact'''
    gender = str(gender)
    if gender.find('Neutered') >= 0: return 'neuter'
    if gender.find('Spayed') >= 0: return 'neuter'
    if gender.find('Intact') >= 0: return 'intact'
    return 'unknown'


def get_gender(gender):
    gender = str(gender)
    if gender.find('Male') >= 0: return 'male'
    if gender.find('Female') >= 0: return 'female'
    return 'unknown'


def clean(df, avg_age = None):
    df['Has_Name'] = [has_name(name) for name in df.Name]
    df['AgeinDays'] = [convert_age(age) for age in df.AgeuponOutcome]
    if avg_age == None:
        int(df.AgeinDays.mean())
        df['AgeinDays'] = df.AgeinDays.fillna(int(df.AgeinDays.mean()))    
    else:
        df['AgeinDays'] = df.AgeinDays.fillna(avg_age)

    df['Hour'] = [(d.split()[1]).split(':')[0] for d in df.DateTime]
    df['Year'] = [year(d.split()[0]) for d in df.DateTime]
    df['Month'] = [month(d.split()[0]) for d in df.DateTime]
    df['Day'] = [day(d.split()[0]) for d in df.DateTime]
    df['Gender'] = df.SexuponOutcome.apply(get_gender)
    df['Neutered'] = df.SexuponOutcome.apply(get_neutered)
    df['Color'] = [', '.join(c.split('/')) for c in df.Color]
    return df

def preprocess(df, breed_vec, color_vec, feature_vec):
    breed_mat = breed_vec.transform(df.pop('Breed'))
    color_mat = color_vec.transform(df.pop('Color'))
    X = feature_vec.transform(df.T.to_dict().values())
    from scipy.sparse import hstack
    X = hstack((X, color_mat, breed_mat))
    return X.toarray()


#############################################################################



#############################################################################
def main():    
    #LOAD DATA AND PROCESS DATA
    import timeit
    start = timeit.default_timer()    
    
    df = pd.read_csv('train.csv')
    df = clean(df)
    mean_age = int(df.AgeinDays.mean())
    
    test_df = pd.read_csv('test.csv')
    test_df = clean(test_df, avg_age = mean_age)
    
    
    #drop all unnecessary columns
    df = df.drop(['AnimalID','Name','OutcomeSubtype', 'DateTime', 
                  'AgeuponOutcome', 'SexuponOutcome'], axis = 1)
    ID = test_df.pop('ID')
    test_df = test_df.drop(['Name','AgeuponOutcome', 
                            'SexuponOutcome'], axis = 1)
    
    #set target as y
    y = df.pop('OutcomeType')
    y_vec = CountVectorizer()
    
    
    
    #throw breed and color to countvectorizers
    breed_vec = CountVectorizer(tokenizer = tokenize)
    breed_vec.fit(df.Breed)
    color_vec = CountVectorizer(tokenizer = tokenize)
    color_vec.fit(df.Color)
    
    #throw everything else to dictvectorizer
    feature_vec = DictVectorizer()
    feature_vec.fit(df.T.to_dict().values())
    
    
    X = preprocess(df, breed_vec, color_vec, feature_vec)
    X_test = preprocess(test_df, breed_vec, color_vec, feature_vec)
    
    #from sklearn.ensemble import RandomForestClassifier
    #clf = RandomForestClassifier(n_estimators = 500)
    
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(learning_rate = 0.05,
                                     n_estimators = 750,
                                     max_depth = 6)    
    
    '''
    #CODE FOR ESTIMATING THE LOG LOSS OF TRAINING DATA
    cut = int(len(y)*0.8)
    X_train = X[:cut]
    X_val = X[cut:]
    y_train = y[:cut]
    y_mat = y_vec.fit_transform(y).toarray()
    
    
    clf.fit(X_train, y_train)
    results = clf.predict_proba(X_val)
    
    
    from sklearn.metrics import log_loss
    
    print([log_loss(y_mat[cut:], results)])
    '''
    
    clf.fit(X,y)
    results = clf.predict_proba(X_test)
    result_df = pd.DataFrame(data = results, columns = clf.classes_)
    result_df.insert(0, 'ID', ID)
    result_df.to_csv('submission.csv', index = False)
    
    stop = timeit.default_timer()
    print(stop - start)