# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:17:18 2016

@author: Chris
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def has_name(name):
    """"Checks each entry to see if animal has a name or not."""
    if type(name) == str:
        return True
    else:
        return False


def convert_age(s):
    '''Converts age from a string format (e.g. 1 year) to age 
    in number of days. Returns type integers.'''
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

def show_table(dictionary):
    '''Prints the key-value pairs of a dictionary'''
    for key in dictionary:
        print(str(key) + ": " + str(dictionary[key]))

def wait():
    '''Pauses program. Asks user to hit enter to continue program.'''
    wait = raw_input('\n\nPress enter to continue...\n')



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


#LOAD DATA AND PROCESS DATA
df = pd.read_csv('train.csv')
df['Has_Name'] = [has_name(name) for name in df.Name]


df['AgeinDays'] = [convert_age(age) for age in df.AgeuponOutcome]
avg_age = int(df.AgeinDays.mean())
for i in range(len(df.AgeinDays)):
    #Replace nan ages with the average age
    if df.AgeinDays[i] == float('nan'):
        df.AgeinDays[i] = avg_age
    #Replace nan sex with 'Unknown'
    if df.SexuponOutcome[i] == float('nan'):
        df.SexuponOutcome[i] = 'Unknown'

#df['Hour'] = []
df['Month'] = [month(d.split()[0]) for d in df.DateTime]
df['Day'] = [day(d.split()[0]) for d in df.DateTime]

df['Gender'] = df.SexuponOutcome.apply(get_gender)
df['Neutered'] = df.SexuponOutcome.apply(get_neutered)

df['Breed'] = [b.split('/') for b in df.Breed]
df['Color'] = [c.split('/') for c in df.Color]


#drop all unnecessary columns
df = df.drop(['AnimalID','Name','OutcomeSubtype', 'DateTime', 'AgeuponOutcome',
              'SexuponOutcome'], axis = 1)


#USE OneHotEncoder for SexuponOutcome, AnimalType, Day, Month

#Use CountVectorizer for Breed and Color


#COUNT THE NUMBER OF EACH OUTCOME FOR ANIMALS WITH NAMES
named = df[df['Has_Name']==True]
named_outcomes = Counter(named.OutcomeType)

    
    
#PLOT DATA FOR ANIMALS WITH NAMES
plt.figure(1, figsize = (6, 1.615*6))
plt.pie(x = list(named_outcomes.viewvalues()),
        labels = list(named_outcomes.viewkeys()),
        autopct='%1.1f%%')

plt.axis('equal')
plt.title('Outcomes for Animals with Names')
plt.show()

print('Outcomes for Animals with Names:')
show_table(named_outcomes)
print('Total: '+ str(len(named)))

wait()

#COUNT NUMBER OF EACH OUTCOME FOR ANIMALS WITHOUT NAMES
unnamed = df[df['Has_Name'] == False]
unnamed_outcomes = Counter(unnamed.OutcomeType)

#PLOT DATA
plt.figure(2, figsize = (6, 1.615*6))
plt.pie(x = list(unnamed_outcomes.viewvalues()),
        labels = list(unnamed_outcomes.viewkeys()),
        autopct = '%1.1f%%')
    
plt.axis('equal')
plt.title('Outcomes for Animals without Names')
plt.show()

print('\n\nOutcomes for Animals without Names:')
show_table(unnamed_outcomes)
print('Total: '+ str(len(unnamed)))

wait()

#LOOK AT OUTCOMES FOR EACH ANIMAL TYPE (CAT/DOG)
dog_outcomes = Counter(df[df['AnimalType']=='Dog']['OutcomeType'])
cat_outcomes = Counter(df[df['AnimalType']=='Cat']['OutcomeType'])

plt.figure(3, figsize = (6,1.615*6))
plt.pie(x = list(dog_outcomes.viewvalues()),
        labels = list(dog_outcomes.viewkeys()),
        autopct = "%1.1f%%")
plt.axis('equal')
plt.title('Outcomes for Dogs')
plt.show()

print('Outcomes for Dogs')
show_table(dog_outcomes)

wait()


plt.figure(4, figsize = (6,1.615*6))
plt.pie(x = list(cat_outcomes.viewvalues()),
        labels = list(cat_outcomes.viewkeys()),
        autopct = "%1.1f%%")
plt.axis('equal')
plt.title('Outcomes for Cats')
plt.show()

print("Outcome for Cats")
show_table(cat_outcomes)

wait()
    