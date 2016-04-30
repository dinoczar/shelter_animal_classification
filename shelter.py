# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:51:49 2016

@author: Chris
"""

##############################################################################
##############################################################################

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

def is_mix(breed):
    if breed.find('Mix') >= 0 or len(breed.split('/')) > 1:
        return True
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

def get_color(color):
    color = str(color)
    if color.find('White')>=0: return 'white'
    if color.find('Black')>=0: return 'black'
    if color.find('Brown')>=0: return 'brown'
    if color.find('Tabby')>=0: return 'tabby'
    if color.find('Tan')>=0: return 'tan'
    if color.find('Blue')>=0: return 'blue'
    if color.find('Orange')>=0: return 'orange'
    if color.find('Brindle')>=0: return 'brindle'
    if color.find('Red')>=0: return 'red'
    if color.find('Tricolor')>=0: return 'tricolor'
    if color.find('Tortie')>=0: return 'tortie'
    if color.find('Cream')>=0: return 'cream'
    if color.find('Point') >= 0: return 'point'
    if color.find('Calico')>=0: return 'calico'
    return 'n/a'

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
    df['Mix'] = df.Breed.apply(is_mix)    
    return df

def preprocess(df, color_vec, feature_vec, fit = False):
    if fit == True:
        #breed_mat = breed_vec.fit_transform(df.pop('Breed'))
        color_mat = color_vec.fit_transform(df.pop('Color'))
        X = feature_vec.fit_transform(df.T.to_dict().values())
    else:
        #breed_mat = breed_vec.transform(df.pop('Breed'))
        color_mat = color_vec.transform(df.pop('Color'))
        X = feature_vec.transform(df.T.to_dict().values())
    from scipy.sparse import hstack
    X = hstack((X, color_mat))
    return X.toarray()

def load_data():
    df = pd.read_csv('train.csv')
    
    
    df = clean(df)
    mean_age = int(df.AgeinDays.mean())
    
    test_df = pd.read_csv('test.csv')
    test_df = clean(test_df, avg_age = mean_age)
    
    
    #drop all unnecessary columns
    df = df.drop(['AnimalID','Name','OutcomeSubtype', 'DateTime', 
                  'AgeuponOutcome', 'SexuponOutcome'], axis = 1)
    
    
    test_df = test_df.drop(['Name','AgeuponOutcome', 
                            'SexuponOutcome'], axis = 1)
                            
                            
    df = df.drop('Breed', axis=1)    
    test_df = test_df.drop('Breed', axis = 1)
    return df, test_df

def estimate_loss(X, y, X_test, clf):
    #CODE FOR ESTIMATING THE LOG LOSS OF TRAINING DATA        
    cut = int(len(y)*0.8)
    y_vec = CountVectorizer()
    X_train = X[:cut]
    X_val = X[cut:]
    y_train = y[:cut]
    y_mat = y_vec.fit_transform(y).toarray()
    
    
    clf.fit(X_train, y_train)
    results = clf.predict_proba(X_val)
    
    
    from sklearn.metrics import log_loss
    
    print([log_loss(y_mat[cut:], results)])


#############################################################################



#############################################################################
def main():
    #LOAD DATA AND PROCESS DATA
    import timeit
    start = timeit.default_timer()    
    
    print('Loading data...')
    df, test_df = load_data()
    ID = test_df.pop('ID')
    
    
    print('Preprocessing data...')    
    #set target as y
    y = df.pop('OutcomeType')
    
    #throw breed and color to countvectorizers
    #breed_vec = CountVectorizer(tokenizer = tokenize)
    color_vec = CountVectorizer(tokenizer = tokenize)
    
    #throw everything else to dictvectorizer
    feature_vec = DictVectorizer()
    
    
    X = preprocess(df, color_vec, feature_vec, fit = True)
    X_test = preprocess(test_df, color_vec, feature_vec)
    
    del df
    del test_df
    
    print('Training classifier...')
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators = 1250)    
    
    
    #Use gradient boosting classifier as base
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(learning_rate = 0.05,
                                     n_estimators = 1000,
                                     max_depth = 6)    
    
    
    gbc.fit(X,y)
    print('Making predictions...')
    results = gbc.predict_proba(X_test)
    
    #Create an ensemble using a number of random forest classifiers
    N = 25
    for i in range(N):
        print(str(i)+' Training classifier...')
        rfc.fit(X,y)
        results+= rfc.predict_proba(X_test)        
    
    #Give the  gradient boosting classifier more votes in the ensemble 
    results = results/(N+5)
    
    print('Writing results to file...')
    result_df = pd.DataFrame(data = results, columns = gbc.classes_)
    result_df.insert(0, 'ID', ID)
    result_df.to_csv('gbc_rf_201600413.csv', index = False)
    
    
    
    
    
    stop = timeit.default_timer()
    print('Execution time: %.2f minutes' %((stop - start)/60))

if __name__ == '__main__':
    main()