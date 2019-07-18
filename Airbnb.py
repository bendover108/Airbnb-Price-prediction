#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[176]:


os.chdir('C:\\Users\\sprad\\OneDrive\\Desktop\\Eynos')
df = pd.read_csv('listings_summary.csv')


# In[3]:


def missing_values_table(df):
    total_missing = df.isnull().sum().sort_values(ascending=False)
    percentage_missing = (100*df.isnull().sum()/len(df)).sort_values(ascending=False)
    missing_table = pd.DataFrame({'missing values':total_missing,'% missing':percentage_missing})
    return missing_table
missing_values = missing_values_table(df)
missing_values


# In[4]:


drop_columns = ['xl_picture_url','jurisdiction_names','thumbnail_url','medium_url',
                 'host_acceptance_rate','square_feet','license','monthly_price','weekly_price','notes']


# In[5]:


df = df.drop(labels=drop_columns, axis=1)


# In[6]:


df


# In[7]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)


# In[8]:


print("The dataset has {} rows and {} columns.".format(train.shape[0],train.shape[1]))


# In[9]:


print("The dataset has {} rows and {} columns.".format(test.shape[0],train.shape[1]))


# In[10]:


#print('It contains {} duplicates'.format(train.duplicated().sum()))


# In[11]:


#print('It contains {} duplicates'.format(train.duplicated().sum()))


# In[12]:


train.head()


# In[13]:


test.head()


# In[14]:


#train.shape


# In[15]:


#test.shape


# In[16]:


train['host_response_rate']=train['host_response_rate'].astype(str)


# In[17]:


#train['host_response_rate']=train['host_response_rate'].fillna(0, inplace=True)


# In[18]:


train['host_response_rate']


# In[19]:


train['host_response_rate']= train['host_response_rate'].map(lambda x: x.replace('%',''))


# In[20]:


train['host_response_rate']=train['host_response_rate'].astype(float)
train['host_response_rate'].mean()


# In[21]:


train['host_response_rate']=train['host_response_rate'].astype(str)


# In[22]:


train['host_response_rate']=train['host_response_rate'].map(lambda x: x.replace('nan','91.80'))


# In[23]:


#to find the mean of train['host_response_rate'] 
#train['host_response_rate'].astype(float).mean()


# In[24]:


train[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(3)


# In[25]:


train['price']=train['price'].astype(str)
train['cleaning_fee']=train['cleaning_fee'].astype(str)
train['extra_people']=train['extra_people'].astype(str)
train['security_deposit']=train['security_deposit'].astype(str)


# In[26]:


train['price']=train['price'].map(lambda x : x.replace('$',''))
train['price']=train['price'].map(lambda x : x.replace(',',''))


# In[27]:


train['cleaning_fee']=train['cleaning_fee'].map(lambda x : x.replace('$',''))
train['cleaning_fee']=train['cleaning_fee'].map(lambda x : x.replace(',',''))


# In[28]:


train['extra_people']=train['extra_people'].map(lambda x : x.replace('$',''))
train['extra_people']=train['extra_people'].map(lambda x : x.replace(',',''))


# In[29]:


train['security_deposit']=train['security_deposit'].map(lambda x : x.replace('$',''))
train['security_deposit']=train['security_deposit'].map(lambda x : x.replace(',',''))


# In[30]:


train['price']=train['price'].astype(float)
train['price'].mean()


# In[31]:


train['cleaning_fee']=train['cleaning_fee'].astype(float)
train['cleaning_fee'].mean()


# In[32]:


train['extra_people']=train['extra_people'].astype(float)
train['extra_people'].mean()


# In[33]:


train['security_deposit']=train['security_deposit'].astype(float)
train['security_deposit'].mean()


# In[34]:


train['cleaning_fee']=train['cleaning_fee'].astype(str)
train['cleaning_fee']=train['cleaning_fee'].map(lambda x : x.replace('nan','26.66'))


# In[35]:


train['price']=train['price'].astype(str)
train['price']=train['price'].map(lambda x : x.replace('nan','67.82'))


# In[36]:


train['extra_people']=train['extra_people'].astype(str)
train['extra_people']=train['extra_people'].map(lambda x : x.replace('nan','8.25'))


# In[37]:


train['security_deposit']=train['security_deposit'].astype(str)
train['security_deposit']=train['security_deposit'].map(lambda x : x.replace('nan','203.11'))


# In[38]:


train[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(10)


# In[39]:


def missing_values_table(train):
    total_missing = train.isnull().sum().sort_values(ascending=False)
    percentage_missing = (100*train.isnull().sum()/len(train)).sort_values(ascending=False)
    missing_table = pd.DataFrame({'missing values':total_missing,'% missing':percentage_missing})
    return missing_table
missing_values = missing_values_table(train)
missing_values


# In[40]:


train['host_response_time']


# In[41]:


train['host_response_time'].mode()


# In[42]:


train['host_response_time']=train['host_response_time'].astype(str)
train['host_response_time']=train['host_response_time'].map(lambda x : x.replace('NaN','within an hour'))
train['host_response_time']=train['host_response_time'].map(lambda x : x.replace('nan','within an hour'))


# In[43]:


train['interaction'].isnull().sum()


# In[44]:


len(train['interaction'].unique())


# In[45]:


train['interaction']


# In[46]:


train['interaction'] = train['interaction'].fillna(str(train['interaction'].mode()))


# In[47]:


train['interaction']=train['interaction'].fillna(str(train['interaction'].mode()))


# In[48]:


train['interaction']


# In[49]:


train['access']=train['access'].fillna(str(train['access'].mode()))
train['access']


# In[50]:


train['host_about']


# In[51]:


train['host_about']=train['host_about'].fillna(str(train['host_about'].mode()))
train['host_about']


# In[52]:


train['review_scores_value']=train['review_scores_value'].fillna(train['review_scores_value'].mean())


# In[53]:


train['review_scores_checkin']=train['review_scores_checkin'].fillna(train['review_scores_checkin'].mean())


# In[54]:


train['review_scores_location']=train['review_scores_location'].fillna(train['review_scores_location'].mean())


# In[55]:


train['review_scores_communication']=train['review_scores_communication'].fillna(train['review_scores_communication'].mean())


# In[56]:


train['review_scores_accuracy']=train['review_scores_accuracy'].fillna(train['review_scores_accuracy'].mean())


# In[57]:


train['review_scores_cleanliness']=train['review_scores_cleanliness'].fillna(train['review_scores_cleanliness'].mean())


# In[58]:


train['review_scores_rating']=train['review_scores_rating'].fillna(train['review_scores_rating'].mean())


# In[59]:


train['reviews_per_month']


# In[60]:


train['reviews_per_month']=train['reviews_per_month'].fillna(train['reviews_per_month'].mean())


# In[61]:


train['neighbourhood']=train['neighbourhood'].fillna(train['neighbourhood'].mode())


# In[62]:


train['neighbourhood'].mode()


# In[63]:


train['neighbourhood']=train['neighbourhood'].astype(str)


# In[64]:


train['neighbourhood']=train['neighbourhood'].map(lambda x : x.replace('NaN','Neukölln'))
train['neighbourhood']=train['neighbourhood'].map(lambda x : x.replace('nan','Neukölln'))


# In[65]:


train['neighbourhood']


# In[66]:


train['text']=train['space']+' '+train['description']+' '+train['neighborhood_overview']+' '+train['summary']


# In[67]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    #sentence=sentence.replace('{html}',"") 
    #cleanr = re.compile('<.*?>')
    #cleantext = re.sub(cleanr, '', sentence)
    #rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


# In[68]:


columns_to_keep = ['last_scraped','experiences_offered','neighborhood_overview','transit','access','interaction',
                  'house_rules','host_name','host_since','host_location','host_about',
                  'host_response_rate','host_is_superhost','host_neighbourhood','host_listings_count',
                  'host_total_listings_count','host_identity_verified','street','neighbourhood', 'neighbourhood_cleansed',
                  'neighbourhood_group_cleansed', 'city', 'state','market','smart_location','property_type', 'room_type', 'accommodates',
                  'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price',
                  'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people',
                  'minimum_nights', 'maximum_nights', 'calendar_updated',
                  'has_availability', 'availability_30', 'availability_60',
                  'availability_90', 'availability_365', 'calendar_last_scraped',
                  'number_of_reviews','review_scores_rating', 'review_scores_accuracy',
                  'review_scores_cleanliness', 'review_scores_checkin',
                  'review_scores_communication','requires_license', 'instant_bookable',
                  'is_business_travel_ready', 'cancellation_policy',
                  'require_guest_profile_picture', 'require_guest_phone_verification',
                  'calculated_host_listings_count', 'reviews_per_month',]


# In[69]:


train['neighborhood_overview']=train['neighborhood_overview'].fillna(train['neighborhood_overview'].mode())


# In[70]:


train['neighborhood_overview']=train['neighborhood_overview'].astype(str)


# In[71]:


train['neighborhood_overview']=train['neighborhood_overview'].map(lambda x : x.replace('NaN','Welcome Traveler. The Singer Hostel and  Apartment is located right in the center of Berlin. All major attractions are easily accessible and there are plenty of opportunities to have fun in the most trendy suburbs of Berlin - Friedrichshain, Kreuzberg (East Side Gallery, Bars, Clubs, Mercedes Benz Arena, subculture), Prenzlauer Berg (coffee shops, restaurants, shopping streets) and Mitte (tourist attractions).'))


# In[72]:


train['neighborhood_overview']=train['neighborhood_overview'].map(lambda x : x.replace('nan','Welcome Traveler. The Singer Hostel and  Apartment is located right in the center of Berlin. All major attractions are easily accessible and there are plenty of opportunities to have fun in the most trendy suburbs of Berlin - Friedrichshain, Kreuzberg (East Side Gallery, Bars, Clubs, Mercedes Benz Arena, subculture), Prenzlauer Berg (coffee shops, restaurants, shopping streets) and Mitte (tourist attractions).'))


# In[73]:


train['neighborhood_overview']


# In[74]:


train['transit'].mode()


# In[76]:


train['transit']=train['transit'].astype(str)


# In[77]:


train['transit']


# In[78]:


train['transit']=train['transit'].map(lambda x : x.replace('nan','U-Bahn / Underground /Subway / Metro: U5 (Strausberger Platz); S-Bahn / city train: S5; S7; S75 (JannowitzbrÃ¼cke) (Alexanderplatz) BUS 142 Bike rental at the TV tower House for car parking at Ostbahnhof: 4,00 EUR. / day House for car parking  at ALEXA: 10,00 EUR. / day'))


# In[79]:


train['house_rules']


# In[80]:


train['house_rules'] = train['house_rules'].astype(str)


# In[81]:


train['house_rules']=train['house_rules'].map(lambda x : x.replace('nan','No shoes in the house'))


# In[82]:


train['host_name'].isnull().sum()


# In[83]:


train['host_name']


# In[84]:


train['host_name'].mode()


# In[85]:


train['host_name'] = train['host_name'].astype(str)


# In[86]:


train['host_name']=train['host_name'].map(lambda x : x.replace('nan','Anna'))


# In[87]:


train['host_since']


# In[88]:


train['host_since'].isnull().sum()


# In[89]:


train['host_since'].mode()


# In[90]:


train['host_since']=train['host_since'].astype(str)


# In[91]:


train['host_since']=train['host_since'].map(lambda x : x.replace('nan','2017-02-20'))


# In[92]:


train['host_location'].isna().sum()


# In[93]:


train['host_location'].mode()


# In[94]:


train['host_location']=train['host_location'].astype(str)


# In[95]:


train['host_location']=train['host_location'].map(lambda x : x.replace('nan','Berlin, Berlin, Germany'))


# In[96]:


train['host_is_superhost'].isnull().sum()


# In[97]:


train['host_is_superhost'].mode()


# In[98]:


train['host_is_superhost']=train['host_is_superhost'].astype(str)


# In[99]:


train['host_is_superhost']=train['host_is_superhost'].map(lambda x : x.replace('nan','f'))


# In[100]:


train['host_neighbourhood'].isnull().sum()


# In[101]:


train['host_neighbourhood'].mode()


# In[102]:


train['host_neighbourhood']=train['host_neighbourhood'].astype(str)


# In[103]:


train['host_neighbourhood']=train['host_neighbourhood'].map(lambda x : x.replace('nan','Neukölln'))


# In[104]:


train['host_listings_count'].isnull().sum()


# In[105]:


train['host_listings_count']=train['host_listings_count'].fillna(train['host_listings_count'].mean())


# In[106]:


train['host_total_listings_count'].isnull().sum()


# In[107]:


train['host_total_listings_count']=train['host_total_listings_count'].fillna(train['host_total_listings_count'].mean())


# In[108]:


train['host_identity_verified'].isnull().sum()


# In[109]:


train['host_identity_verified'].mode()


# In[110]:


train['host_identity_verified']=train['host_identity_verified'].astype(str)


# In[111]:


train['host_identity_verified']=train['host_identity_verified'].map(lambda x : x.replace('nan','f'))


# In[112]:


train['street'].isnull().sum()


# In[113]:


train['neighbourhood'].isnull().sum()


# In[114]:


train['neighbourhood_cleansed'].isnull().sum()


# In[115]:


train['neighbourhood_group_cleansed'].isnull().sum()


# In[116]:


train['city'].isnull().sum()


# In[117]:


train['city'].mode()


# In[118]:


train['city']=train['city'].astype(str)


# In[119]:


train['city']=train['city'].map(lambda x : x.replace('nan','Berlin'))


# In[120]:


columns_to_keep = ['last_scraped','text','experiences_offered','neighborhood_overview','transit','access','interaction',
                  'house_rules','host_name','host_since','host_location','host_about',
                  'host_response_rate','host_is_superhost','host_neighbourhood','host_listings_count',
                  'host_total_listings_count','host_identity_verified','street','neighbourhood', 'neighbourhood_cleansed',
                  'neighbourhood_group_cleansed', 'city', 'state','market','smart_location','property_type', 'room_type', 'accommodates',
                  'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price',
                  'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people',
                  'minimum_nights', 'maximum_nights', 'calendar_updated',
                  'has_availability', 'availability_30', 'availability_60',
                  'availability_90', 'availability_365', 'calendar_last_scraped',
                  'number_of_reviews','review_scores_rating', 'review_scores_accuracy',
                  'review_scores_cleanliness', 'review_scores_checkin',
                  'review_scores_communication','requires_license', 'instant_bookable',
                  'is_business_travel_ready', 'cancellation_policy',
                  'require_guest_profile_picture', 'require_guest_phone_verification',
                  'calculated_host_listings_count', 'reviews_per_month',]


# In[121]:


train['state'].isnull().sum()


# In[122]:


train['state'].mode()


# In[123]:


train['state']=train['state'].astype(str)


# In[124]:


train['state']=train['state'].map(lambda x:x.replace('nan','Berlin'))


# In[125]:


train['market'].isnull().sum()


# In[126]:


train['market'].mode()


# In[127]:


train['market']=train['market'].astype(str)


# In[128]:


train['market']=train['market'].map(lambda x:x.replace('nan','Berlin'))


# In[129]:


train['bathrooms'].isnull().sum()


# In[130]:


train['bathrooms']=train['bathrooms'].fillna(train['bathrooms'].mean())


# In[131]:


train['bedrooms'].isnull().sum()


# In[132]:


train['bedrooms']=train['bedrooms'].fillna(train['bedrooms'].mean())


# In[133]:


train['beds'].isnull().sum()


# In[134]:


train['beds']=train['beds'].fillna(train['beds'].mean())


# In[135]:


train['reviews_per_month'].isnull().sum()


# In[136]:


train[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(3)


# In[137]:


train1=train[columns_to_keep]


# In[138]:


train1


# In[139]:


get_ipython().run_cell_magic('time', '', "train1['clean_text']=train1['text'].map(lambda s:preprocess(s)) ")


# In[140]:


train1.columns


# In[141]:


x1 = train1.duplicated


# In[142]:


train1


# In[151]:


train1['text'].isnull().sum()


# In[144]:


train1['text'].mode()


# In[145]:


train1['text']=train1['text'].astype(str)


# In[ ]:


train1['text']=train1['text'].map(lambda x : x.replace('nan','The Singer 109 Hostel is located in the heart of Berlin, in a former factory building. We offer double rooms and 3-8 bed rooms. The team speaks German and English. (24h Reception and Bar/No Curfew/No Lock Out) Breakfast buffet is served from 07:00 a.m.  - 11:00 a.m.  In the evening, guests can get to know one another over a beer and a chat, during the Happy Hour times. In this time, you can make new friends and plans! with en-suite bathroom, TV, WIFI, bed linen, and storage Breakfast buffet is available for â‚¬ 5.00 including juices, coffee and tea. The reception is open 24 hours. "Late Check Out" is possible after prior agreement! Padlock: 5,00 â‚¬ deposit Keycard: 2,00 â‚¬ deposit The Singer 109 Hostel is located in the heart of Berlin, in a former factory building. We offer double rooms and 3-8 bed rooms. The team speaks German and English. (24h Reception and Bar/No Curfew/No Lock Out) Breakfast buffet is served from 07:00 a.m.  - 11:00 a.m.  In the evening, guests can get to know one another over a beer and a chat, during the Happy Hour times. In this time, you can make new friends and plans! We can offer our guests - Pool table/Snooker - Football table - Internet, WIFI free - Printer (0.50 â‚¬ per page) - Laundry service 5,00 EUR. - Towel: 2,00 EUR. - Padlock: 5,00 EUR. / deposit - Key Card: 2,00 EUR. / deposit - Designated Smoking Area - 24h Bar - Vending Machine - Free Taxi -Call - Free City Maps - M Welcome Traveler. The Singer Hostel and  Apartment is located right in the center of Berlin. All major attractions are easily accessible and there are plenty of opportunities to have fun in the most trendy suburbs of Berlin - Friedrichshain, Kreuzberg (East Side Gallery, Bars, Clubs, Mercedes Benz Arena, subculture), Prenzlauer Berg (coffee shops, restaurants, shopping streets) and Mitte (tourist attractions). with en-suite bathroom, TV, WIFI, bed linen, and storage Breakfast buffet is available for â‚¬ 5.00 including juices, coffee and tea. The reception is open 24 hours. "Late Check Out" is possible after prior agreement! Padlock: 5,00 â‚¬ deposit Keycard: 2,00 â‚¬ deposit'))


# In[146]:


train1.columns


# In[147]:


def column_index(df, query_cols):
    cols = train1.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


# In[148]:


column_index(train1, ['last_scraped', 'text', 'experiences_offered', 'neighborhood_overview',
       'transit', 'access', 'interaction', 'house_rules', 'host_name',
       'host_since', 'host_location', 'host_about', 'host_response_rate',
       'host_is_superhost', 'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_identity_verified', 'street',
       'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'market',
       'smart_location', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price',
       'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people',
       'minimum_nights', 'maximum_nights', 'calendar_updated',
       'has_availability', 'availability_30', 'availability_60',
       'availability_90', 'availability_365', 'calendar_last_scraped',
       'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'requires_license', 'instant_bookable',
       'is_business_travel_ready', 'cancellation_policy',
       'require_guest_profile_picture', 'require_guest_phone_verification',
       'calculated_host_listings_count', 'reviews_per_month', 'clean_text'])


# In[149]:


get_ipython().run_cell_magic('time', '', "train1['clean_text']=train1['text'].map(lambda s:preprocess(s))")


# In[152]:


train1.to_csv('clean.csv',index=False)


# In[153]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
response = vectorizer.fit_transform(train1['clean_text'])


# In[154]:


response


# In[234]:


cols=vectorizer.get_feature_names()
x=response.todense()
x=pd.DataFrame(x)


# In[235]:


x.info()


# In[236]:


x


# In[158]:


train2=train1


# In[160]:


train2=pd.concat([train2,x], axis=1)


# In[164]:


train2


# In[163]:


train2=train2.drop(['text','neighborhood_overview','transit','house_rules','host_name','host_since','last_scraped'],axis=1)


# In[162]:


train2.to_csv('final clean.csv', index=False)


# In[178]:


df['price']=df['price'].map(lambda x: x.replace('$',''))


# In[181]:


df['price']=df['price'].map(lambda x: x.replace(',',''))


# In[182]:


df['price']=df['price'].astype(float)


# In[184]:


train2['price']=df['price']


# In[187]:


train2['price']=train2.drop(['price'],axis=1)


# In[206]:


train2=train2.drop(['experiences_offered', 'access', 'interaction', 'host_location', 'host_about', 'host_response_rate', 'host_is_superhost', 'host_neighbourhood', 'host_identity_verified', 
                    'street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'market', 
                    'smart_location', 'property_type', 'room_type', 'bed_type', 'amenities', 'price', 'security_deposit', 
                    'cleaning_fee', 'extra_people', 'calendar_updated', 'has_availability', 'calendar_last_scraped', 'requires_license',
                    'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture', 
                    'require_guest_phone_verification', 'clean_text'],axis=1)


# In[222]:


train2['target']


# In[237]:


from sklearn.model_selection import train_test_split
X = x
#X=word_vec
y = train1['price'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33,random_state=1317)


# In[238]:


import lightgbm as lgb
# Wrap our training and validation sets in LightGBM Datasets.
d_train = lgb.Dataset(X_train, label=y_train,free_raw_data=False)
d_valid = lgb.Dataset(X_val, label=y_val,free_raw_data=False)
#d_train = lgb.Dataset(X_train, label=y_train)
#d_valid = lgb.Dataset(X_val, label=y_val)


# In[239]:


params={ 
        'objective': 'regression'
        
       }


# In[240]:


gbm = lgb.train(params=params, # parameter dict to use
                    train_set=d_train,
                    init_model=None, # initial model to use, for continuous training.
                    num_boost_round=100, # the boosting rounds or number of iterations.
                    early_stopping_rounds=10, # early stopping iterations.
                    # stop training if *no* metric improves on *any* validation data.
                    valid_sets=d_valid,
                     # dict to store evaluation results in.
                   verbose_eval=True
               )


# In[241]:


y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)


# In[243]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_val, y_pred))
mae=mean_absolute_error(y_val, y_pred)
print("Root mean squared error: ",rms)
print("Mean absolute error: ",mae)


# In[250]:


train.columns


# In[ ]:




