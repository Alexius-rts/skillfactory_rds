#!/usr/bin/env python
# coding: utf-8

# ### Итоговое задание Беломойкина Алексея по Проекту 3. О вкусной и здоровой пище (SF-DST-18)

# ![](https://www.pata.org/wp-content/uploads/2014/09/TripAdvisor_Logo-300x119.png)
# # Predict TripAdvisor Rating
# ## В этом соревновании нам предстоит предсказать рейтинг ресторана в TripAdvisor
# **По ходу задачи:**
# * Прокачаем работу с pandas
# * Научимся работать с Kaggle Notebooks
# * Поймем как делать предобработку различных данных
# * Научимся работать с пропущенными данными (Nan)
# * Познакомимся с различными видами кодирования признаков
# * Немного попробуем [Feature Engineering](https://ru.wikipedia.org/wiki/Конструирование_признаков) (генерировать новые признаки)
# * И совсем немного затронем ML
# * И многое другое...   

# # import

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

import operator
from collections import Counter
import datetime as DT

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42


# In[3]:


# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# # DATA

# In[4]:


DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')


# In[5]:


df_train.info()


# In[6]:


df_train.head(5)


# In[7]:


df_test.info()


# In[8]:


df_test.head(5)


# In[9]:


sample_submission.head(5)


# In[10]:


sample_submission.info()


# In[11]:


# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем


# In[12]:


df.info()


# Подробнее по признакам:
# * City: Город 
# * Cuisine Style: Кухня
# * Ranking: Ранг ресторана относительно других ресторанов в этом городе
# * Price Range: Цены в ресторане в 3 категориях
# * Number of Reviews: Количество отзывов
# * Reviews: 2 последних отзыва и даты этих отзывов
# * URL_TA: страница ресторана на 'www.tripadvisor.com' 
# * ID_TA: ID ресторана в TripAdvisor
# * Rating: Рейтинг ресторана

# In[13]:


df.sample(5)


# In[14]:


df.Reviews[1]


# Как видим, большинство признаков у нас требует очистки и предварительной обработки.

# # Cleaning and Prepping Data
# Обычно данные содержат в себе кучу мусора, который необходимо почистить, для того чтобы привести их в приемлемый формат. Чистка данных — это необходимый этап решения почти любой реальной задачи.   
# ![](https://analyticsindiamag.com/wp-content/uploads/2018/01/data-cleaning.png)

# ## Предобработка данных и определение базовых функций.

# In[15]:



#Приведем имена столбцов к нижнему регистру, и переименуем некоторые из них

df.columns = map(str.lower, df.columns)
df.rename(columns={'cuisine style': 'cuisine_style', 'price range': 'price_range', 'number of reviews':'number_reviews'}, inplace=True)


# Функция по сбору информации по каждому столбцу

def info_column(column):
    print('Число уникальных значений:', df[column].nunique(), '\n')
    print('Число упоминания каждого значения: \n',
          df[column].value_counts(), '\n')
    print('Число пустых значений в столбце:',
          (df[column].isnull()).sum(), '\n')
    return  # df.loc[:, [column]].info()

# прорисовка гистограммы исходных данных
def hist_source_data(column):
    df[column].hist(alpha=0.4, bins=100, range=(
        start_point, end_point), label='Исходные данные {}'. format(column))
    plt.legend()
    return


# ## Первичный отсмотр данных

# In[16]:


df.info()


# In[17]:


# построим карту пропусков данных
sns.heatmap(df.isnull(), cbar=True)


# In[18]:


# описываем функцию, заменяющую в строковых столбцах  пробел на None

def clear_column(column):
    return df[column].astype(str).apply(lambda x: None if x.strip() == '' else x)

for col in df.columns:
    clear_column(col)


# ## 1. Обработка NAN 
# У наличия пропусков могут быть разные причины, но пропуски нужно либо заполнить, либо исключить из набора полностью. Но с пропусками нужно быть внимательным, **даже отсутствие информации может быть важным признаком!**   
# По этому перед обработкой NAN лучше вынести информацию о наличии пропуска как отдельный признак 

# In[19]:


# возьмем столбец Number of Reviews
df['number_reviews_isNAN'] = pd.isna(df['number_reviews']).astype('uint8')


# In[20]:


# процент пропусков в столбце: 23.2%
df['cuisine_style'].isnull().value_counts(normalize=True) * 100


# In[21]:


# Пустые значения в 'cuisine_style' заменяем на ['Unknown']
df['cuisine_style'] = df['cuisine_style'].fillna("['Unknown']")


# In[22]:


# процент пропусков в столбце: 34.715
df['price_range'].isnull().value_counts(normalize=True) * 100


# In[23]:


df['price_range_isNAN'] = pd.isna(df['price_range']).astype('uint8')


# In[24]:


df['reviews'] = df['reviews'].fillna('[[], []]') 
df['reviews_isNAN'] =df['reviews'].apply(lambda x: 1 if x =='[[], []]' else 0).astype('uint8')


# ### 2. Обработка признаков
# Для начала посмотрим какие признаки у нас могут быть категориальными.

# In[25]:


df.nunique(dropna=False)


# In[26]:


df.sample(5)


# ### 2.0. restaurant_id

# In[27]:


# создадим столбец с перекодированым 'restaurant_id'
df['code_restaurant_id'] = df['restaurant_id'].apply(lambda x: int(x[3:]))


# ### 2.1. city

# In[28]:


info_column('city')


# In[29]:


df.city.unique()


# In[30]:


# Создаем эталонную таблицу с эталонными  значениями некоторых столбцов для каждого 'city'
reference_table = pd.DataFrame(df.city.unique(), columns = ['city'])


# ### 2.2. cuisine_style

# In[31]:


info_column('cuisine_style')


# In[32]:


df['cuisine_style'].value_counts()[:10]


# In[33]:


# Создадим столбец со списками кухонь
df['cuisines'] = df['cuisine_style'].str.findall(r"'(\b.*?\b)'") 


# In[34]:


# Создадим список списков всех кухонь и подсчитываем количество упоминаний каждой кухни
cuis_list = df['cuisines'].tolist()


def find_list_cuisines(list_of_lists):
    result = []
    for lst in list_of_lists:
        result.extend(lst)
    return result


cuisines_counter = Counter(find_list_cuisines(cuis_list))
cuisines_counter.most_common()


# ### 2.3. ranking

# In[35]:


info_column('ranking')


# ### 2.4. rating

# In[36]:


start_point = df['ranking'].min()
end_point = df['ranking'].max()
hist_source_data('ranking')
#Распределение нормальное. Выбросов и пропусков нет 


# ### 2.4 . rating

# In[37]:


info_column('rating')


# In[38]:


start_point = df['ranking'].min()
end_point = df['ranking'].max()
hist_source_data('ranking')
#Распределение нормальное. Выбросов и пропусков нет 


# ### 2.5. price_range

# In[39]:


info_column('price_range')


# In[40]:


# процент пропусков в столбце: 34.722
df['price_range'].isnull().value_counts(normalize=True) * 100


# In[41]:


#  Заполним пропуски пробелами
df['price_range'].fillna('',inplace = True)


# In[42]:


# Создаем в эталонной таблице столбец 'range_of_price' с модой диапазона  непустых цен для каждого 'city'

def find_price_mode(row):
    return df[(df['city'] == row['city']) & (df['price_range']!='')]['price_range'].mode()[0]


reference_table['range_of_price'] = reference_table.apply(find_price_mode, axis=1)


# In[43]:


reference_table['range_of_price'].value_counts()


# In[44]:


# Пустым значениям в df['price_range'] присваиваем значение из эталонной таблицы в соответствии с 'city' 
def insert_range_of_price(row):
    if row['price_range'] == '': 
        return str(reference_table[reference_table['city'] == row['city']]['range_of_price'].values)[2:-2]
    return row['price_range']


df['price_range'] = df.apply(insert_range_of_price, axis=1)


# In[45]:


df['price_range'].value_counts()


# In[46]:


#Перекодируем  'price_range' в цифры и создадим новый столбец 'сode_price_range':
def code_price_range (row):
    if row['price_range'] == '$':
        return 1
    if row['price_range'] == '$$ - $$$':
        return 2
    return 3
df['code_price_range'] = df.apply(code_price_range, axis=1)


# ### 2.6. number_reviews

# In[47]:


info_column('number_reviews')


# In[48]:


# для начала заполним пропуски в 'number_reviews' 0, чтобы не вылетали последующие функции
df['number_reviews'].fillna(0,inplace = True)


# In[49]:


# Создаем в эталонной таблице столбец 'number_reviews' с модой количества отзывов для каждого 'city'

def find_number_reviews_mode(row):
    return df[df['city'] == row['city']]['number_reviews'].mode()[0]


reference_table['number_reviews'] = reference_table.apply(find_number_reviews_mode, axis=1)


# In[50]:



# нулевым значениям в df[number_reviews'] присваиваем значение из эталонной таблицы в соответствии с 'city' 
def insert_number_reviews(row):
    if row['number_reviews'] == 0: 
        return str(reference_table[reference_table['city'] == row['city']]['number_reviews'].values)[2:-2]
    return row['number_reviews']


df['last_review_date'] = df.apply(insert_number_reviews, axis=1)


# ### 2.7. reviews

# In[51]:


info_column('reviews')


# In[52]:


#Создаем столбец 'last_review_date' с датами отзывов
df['last_review_date'] = df['reviews'].str.findall('\d+/\d+/\d\d\d\d')


# In[53]:


# оставляем в 'last_review_date' дату последнего отзыва
# Если ее нет ставим None  
def date_last_review (row):
    if row['last_review_date'] == []:
        return None
    return max(row['last_review_date'])
df['last_review_date'] = df.apply(date_last_review, axis=1)


# In[54]:


# Переводим столбец df['last_review_date'] в формат DateTime
df['last_review_date'] = pd.to_datetime(df['last_review_date'])
# находим самую раннюю дату
oldest_date =df[df['last_review_date']!=None]['last_review_date'].min()
print(oldest_date)
# заполняем ею пропуски
df['last_review_date'].fillna(oldest_date,inplace=True)


# In[55]:


df['last_review_date'].unique()


# ### 2.8. URL_TA

# In[56]:


info_column('url_ta')


# In[57]:


df['code_url_ta']= df['url_ta'].str.split('-').apply(lambda x: x[1][1:]).astype('int64')


# ### 2.9. ID_TA

# In[58]:


info_column('id_ta')


# In[59]:


df['code_id_ta']= df['id_ta'].apply(lambda x: x[1:]).astype('int64')


# ## Создание дополнительных признаков

# In[60]:


# Создаем признак является ли ресторан сетевым, т.е. встречается ли его id более одного раза
a = df.id_ta.value_counts()
df['chain_restaurant'] = df.id_ta.apply(
    lambda x: 1 if a[x] > 1 else 0).astype('uint8')
df['chain_restaurant'].value_counts()


# In[61]:


# Создаем список столиц
capital_list = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna',
                'Rome', 'Madrid', 'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam',
                'Lisbon', 'Prague', 'Oslo', 'Helsinki', 'Edinburgh', 'Ljubljana', 'Athens',  'Luxembourg']
cities_nord_europe = ['Stockholm', 'London', 'Dublin', 'Copenhagen', 'Amsterdam', 'Oslo',
                      'Helsinki', 'Edinburgh']
cities_central_europe = ['Paris', 'Berlin', 'Munich', 'Bratislava', 'Vienna',  'Brussels',
                         'Zurich', 'Warsaw', 'Budapest', 'Hamburg', 'Prague', 'Geneva',
                         'Ljubljana', 'Luxembourg', 'Krakow', 'Ljubljana']
cities_south_europe = ['Oporto',  'Milan', 'Rome',
                       'Barcelona', 'Madrid', 'Lyon', 'Lisbon', 'Geneva']


# In[62]:


# создаем признак нахождения ресторана в столице
def find_capital_city (row):
    if row['city'] in capital_list:
        return 1
    return 0
df['capital_city'] = df.apply(find_capital_city, axis=1).astype('uint8')


# создаем признак нахождения ресторана в северной Европе
def find_nord_europe (row):
    if row['city'] in cities_nord_europe:
        return 1
    return 0
df['nord_europe'] = df.apply(find_nord_europe , axis=1).astype('uint8')

# создаем признак нахождения ресторана в центральной Европе
def find_central_europe (row):
    if row['city'] in cities_central_europe:
        return 1
    return 0
df['central_europe'] = df.apply(find_central_europe , axis=1).astype('uint8')

# создаем признак нахождения ресторана в южной Европе
def find_south_europe (row):
    if row['city'] in cities_south_europe:
        return 1
    return 0
df['south_europe'] = df.apply(find_south_europe , axis=1).astype('uint8')

# Создадим признак, явдяется ли город морским портом
seaport_list = ['Stockholm', 'London', 'Barcelona', 'Dublin', 'Copenhagen', 'Amsterdam', 
'Hamburg', 'Lisbon', 'Oslo',  'Helsinki', 'Edinburgh', 'Athens']

df['seaport'] = df['city'].apply(lambda x: 1 if x in seaport_list else 0).astype('uint8')

#Создание столбцов с дополнительными признаками городов
for item in list(df.city.unique()):
    df[item] = df.city.apply(lambda x: 1 if x==item else 0)
#df = pd.get_dummies(df, columns=['city'], dummy_na=False)


# In[63]:


# Создаем признак количества кухонь в ресторане
df['count_cuisines'] = df['cuisines'].apply(lambda x: len(x))


# In[64]:


# Cоздадим столбец присутствия редких кухонь (менее 20 на 40000 ресторанов)
rare_cuisine_list = [x[0] for x in cuisines_counter.most_common()[-27:]]

for cuisine in rare_cuisine_list:
    df['rare_cuisine'] = df['cuisines'].apply(lambda x: 1 if cuisine in x else 0).astype('uint8')


# In[65]:


# Cоздадим столбец присутствия 100 популярных кухонь
popular_cuisine_list = [x[0] for x in cuisines_counter.most_common()[:50]]

for cuisine in popular_cuisine_list:
    df['popular_cuisine'] = df['cuisines'].apply(lambda x: 1 if cuisine in x else 0).astype('uint8')


# In[66]:


# Cоздадим столбец нечасто встречающися кухонь
infrequent_cuisine_list = [x[0] for x in cuisines_counter.most_common()[50:-26]]

for cuisine in infrequent_cuisine_list:
    df['infrequent_cuisine'] = df['cuisines'].apply(lambda x: 1 if cuisine in x else 0).astype('uint8')


# In[67]:


# Создадим списки с положительными и негативными словами из отзывов:

good_words_list =['Good', ' good', ' excellent', 'fantast', 'Fantast', 'Excellent', 'Fine', ' fine', 
                  'Better', ' better', 'Delicious', ' delicious' 'Nice', 'nice', ' tasty', 'Tasty', 
                  'Worthy', 'worthy', ' friendly', 'Friendly' ' best', 'Best', ' cozy', 'Cozy', 
                  'Magnifi', ' magnifi', 'Elegant', ' elegant']

bad_words_list =['Bad', 'bad', ' ugly', 'Ugly', ' slow', 'Slow', 'Nightmare', ' nightmare', ' lazy', 'Lazy', 
           ' expensive', 'Expensive', ' worst', 'Worst']

# Создадим признаки наличия хороших и плохих отзывов
df['good_reviews'] = df['reviews'].apply(lambda x: 1 if any(word in x for word in good_words_list) else 0).astype('uint8')


df['bad_reviews'] = df['reviews'].apply(lambda x: 1 if any(word in x for word in bad_words_list) else 0).astype('uint8')


# In[68]:


# Создаем столбец с Timedelta с момента последнего отзыва
df['days_from_review'] = DT.datetime.today() - df['last_review_date']
# и преобразуем его в количество дней
df['days_from_review'] = df['days_from_review'].map(lambda x: x.days)


# In[69]:


# Создадим признак с величиной населения городов
population_dict = {'Paris': 2140526, 'Stockholm': 961609, 'London': 8787892, 'Berlin': 3601131, 'Munich': 1456039, 'Oporto': 221800,
       'Milan': 1366180, 'Bratislava': 437725, 'Vienna': 1840573, 'Rome':2872800, 'Barcelona':  1620343, 'Madrid': 3223334,
       'Dublin' : 553165, 'Brussels' : 1198726, 'Zurich' : 434008, 'Warsaw' : 1702139, 'Budapest' : 1752286, 'Copenhagen' : 615993,
       'Amsterdam' : 859732, 'Lyon' : 515695, 'Hamburg' : 1830584, 'Lisbon': 553000, 'Prague' : 1280508, 'Oslo' : 673469,
       'Helsinki' : 643272, 'Edinburgh': 513210 , 'Geneva': 201818, 'Ljubljana' : 284355, 'Athens' : 655780,
       'Luxembourg' : 122273, 'Krakow' : 779115}

df['population'] = df['city'].apply(lambda x: population_dict[x])


# In[70]:


df.info(verbose = True)


# # EDA 
# [Exploratory Data Analysis](https://ru.wikipedia.org/wiki/Разведочный_анализ_данных) - Анализ данных
# На этом этапе мы строим графики, ищем закономерности, аномалии, выбросы или связи между признаками.
# В общем цель этого этапа понять, что эти данные могут нам дать и как признаки могут быть взаимосвязаны между собой.
# Понимание изначальных признаков позволит сгенерировать новые, более сильные и, тем самым, сделать нашу модель лучше.
# ![](https://miro.medium.com/max/2598/1*RXdMb7Uk6mGqWqPguHULaQ.png)

# ### Посмотрим распределение признака

# In[71]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4));
df_train['Ranking'].hist(bins=100, ax=axes[0])
df_train.boxplot(column='Ranking', ax=axes[1])


# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

# In[72]:


df_train['City'].value_counts(ascending=True).plot(kind='barh')


# А кто-то говорил, что французы любят поесть=) Посмотрим, как изменится распределение в большом городе:

# In[73]:


df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)


# In[74]:


df.groupby(['city'])['count_cuisines'].mean().plot(kind='bar', ylim=(1.5, 4), grid=True, title='Среднее количество кухонь в ресторанах города')


# In[75]:


# Сформируем новый признак - среднее число кухонь для города (<2, 2<=x<3,>=3):
mean_cuisines_list =dict(round(df.groupby(['city'])['count_cuisines'].mean(),2))

df['mean_cuisines<2'] = df.city.apply(lambda x: 1 if mean_cuisines_list[x]<2 else 0)
df['2<=mean_cuisines<3'] = df.city.apply(lambda x: 1 if 2<=mean_cuisines_list[x]<3 else 0)
df['mean_cuisines>=3'] = df.city.apply(lambda x: 1 if mean_cuisines_list[x]>=3 else 0)


# In[76]:


# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()


# Получается, что Ranking имеет нормальное распределение, просто в больших городах больше ресторанов, из-за мы этого имеем смещение.

# ### Посмотрим распределение целевой переменной

# In[77]:


df_train['Rating'].value_counts(ascending=True).plot(kind='barh')


# ### Посмотрим распределение целевой переменной относительно признака

# In[78]:


df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)


# In[79]:


df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)


# ### [Корреляция признаков](https://ru.wikipedia.org/wiki/Корреляция)
# На этом графике можно видеть, как признаки связаны между собой и с целевой переменной.

# In[80]:


df.columns


# In[81]:


# Создаем датафрейм с основными числовыми признаками и смотрим их корреляцию
df_base = df[['rating','ranking', 'number_reviews', 'code_restaurant_id', 'code_url_ta', 'code_price_range', 'price_range_isNAN', 'reviews_isNAN', 'days_from_review', 
              'code_id_ta', 'capital_city', 'count_cuisines', 'rare_cuisine','infrequent_cuisine', 'popular_cuisine', 'chain_restaurant', 'seaport', 
              'good_reviews', 'bad_reviews', 'mean_cuisines<2','2<=mean_cuisines<3','mean_cuisines>=3', 'population' ]] 

correlation = df_base.corr()
plt.figure(figsize=(21, 14))
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# # Data Preprocessing

# In[82]:


#Неактивированная функция
#Теперь, для удобства и воспроизводимости кода, завернем всю обработку в одну большую функцию.


# на всякий случай, заново подгружаем данные
#df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
#df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
#df_train['sample'] = 1 # помечаем где у нас трейн
#df_test['sample'] = 0 # помечаем где у нас тест
#df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

#data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
#data.info()

def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### 1. Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)
    
    
    # ################### 2. NAN ############################################################## 
    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...
    df_output['Number of Reviews'].fillna(0, inplace=True)
    # тут ваш код по обработке NAN
    # ....
    
    
    # ################### 3. Encoding ############################################################## 
    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)
    # тут ваш код не Encoding фитчей
    # ....
    
    
    # ################### 4. Feature Engineering ####################################################
    # тут ваш код не генерацию новых фитчей
    # ....
    
    
    # ################### 5. Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим
    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']
    df_output.drop(object_columns, axis = 1, inplace=True)
    
    return df_output

#df_preproc = preproc_data(data)


# In[83]:


# Создаем датафрейм с числовыми столбцами исходного датафрейма
df_preproc = df.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8'])


# In[84]:


# У 'ranking' и 'code_restaurant_id' одинаковая корреляция со всеми столбцами, поэтому удалим 'code_restaurant_id'
df_preproc.drop(['code_restaurant_id'], axis = 1,inplace=True, errors='ignore')
# Из-за низкой кореляции с целевой переменной удаляем признаки морского порта и характеристики отзывов:
df_preproc.drop([ 'seaport', 'good_reviews', 'bad_reviews', 'code_id_ta'], axis = 1,inplace=True, errors='ignore')
df_preproc.sample(10)


# #### Запускаем и проверяем что получилось

# In[85]:


df_preproc.info()


# In[86]:


# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.rating.values            # наш таргет
X = train_data.drop(['rating'], axis=1)


# **Перед тем как отправлять наши данные на обучение, разделим данные на еще один тест и трейн, для валидации. 
# Это поможет нам проверить, как хорошо наша модель работает, до отправки submissiona на kaggle.**

# In[87]:


# Воспользуемся специальной функцией train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[88]:


# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape


# # Model 
# Сам ML

# In[89]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[90]:


# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)


# In[91]:


# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)


# In[92]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# In[93]:


# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')


# # Submission
# Если все устраивает - готовим Submission на кагл

# In[94]:


test_data.sample(10)


# In[95]:


test_data = test_data.drop(['rating'], axis=1)


# In[96]:


sample_submission


# In[97]:


predict_submission = model.predict(test_data)


# In[98]:


predict_submission


# In[99]:


sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)

