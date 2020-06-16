#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import operator


# In[2]:


data = pd.read_csv('movie_bd_v5.csv')
data.head(10)


# In[3]:


data.info()


# # Предобработка

# In[4]:


answers = {} # создадим словарь для ответов

data['profit'] = data['revenue'] - data['budget'] # create columns 'profit'

# тут другие ваши предобработки колонок например:

#the time given in the dataset is in string format.
#So we need to change this in datetime format
data['date_of_release'] = pd.to_datetime(data.release_date)

# create column 'len_title' containing lenght of title films
data['len_title'] = data['original_title'].str.len()

# create column 'len_overview' containing quantity words in overview of films
def len_overview(row):
    count = 1
    for letter in row['overview']:
        if letter == ' ':
            count+=1
    return count
data['len_overview'] = data.apply(len_overview, axis=1)



# Задание 11-12, 17. Описываем функцию, которая выбирает значения из столбца 'genres' и помещает их в список genre_list
def search_genres_v1(row):
    temp_list = row['genres'].split('|')
    for genre in temp_list:
        genre_list.append(genre)
    return row['genres']


# Задание 11-12, 17. Описываем функцию, которая выбирает значения из столбца 'genres' и помещает их в словарь genres_list,
# подсчитывая их количество
def search_genres_v2(row):
    temp_list1 = row['genres'].split('|')
    for genre in temp_list1:
        if genre in genres_list.keys():
            genres_list[genre]+= 1
        else:
            genres_list[genre] = 1  
    return row['genres']


# Задание 13,23. Описываем функцию, которая выбирает значения из столбца 'director' и помещает их в словарь director_list
def counting_director(row):
    temp_list = row['director'].split('|')
    for director in temp_list:
        if director in director_list.keys():
            director_list[director]+= 1
        else:
            director_list[director] = 1
    return row['director']

# Задание 15-16. Описываем функцию, которая выбирает значения из столбца 'cast' и помещает их в словарь actor_list
def search_actor(row):
    temp_list = row['cast'].split('|')
    for actor in temp_list:
        if actor in actor_list.keys():
            actor_list[actor]+= 1
        else:
            actor_list[actor] = 1
    return row['cast']


# Задание 21. Описываем функцию, которая перебирает  строковые значения месяцев из столбца 'release_date' 
# и помещает их количество в словарь month_list
# start_month,end_month - числовые значения месяцев, в диапазоне которых нужно вести поиск 
# (end_month - порядковое значение месяца+1)

def search_month_v1(row):
    string = row['release_date']             
    for i in range(start_month,end_month):
        if string.startswith('{}/'.format(i)):
            if i in month_list.keys():
                month_list[i] += 1
            else:
                month_list[i] = 1
    return (row['release_date'])


# Задание 21,22. Описываем функцию, которая перебирает значения месяцев из столбца 'date_of_release' 
# и помещает их  количество в словарь month_list
# start_month,end_month - числовые значения месяцев, в диапазоне которых нужно вести поиск 
# (end_month - порядковое значение месяца+1)

def search_month_v2(row):
    for i in range(start_month,end_month):
        if row['date_of_release'].month == i:
            if i in month_list.keys():
                month_list[i] += 1
            else:
                month_list[i] = 1
    return (row['release_date']) 

# Задание 24, 25. Описываем функцию, которая выбирает значения из столбца 'production_companies' и помещает их в словарь company_list    
def search_company(row):
    temp_list = row['production_companies'].split('|')
    for company in temp_list:
        if company in company_list.keys():
            company_list[company]+= 1
        else:
            company_list[company] = 1
    return row['production_companies']


# Задание 27. Описываем функцию, которая выбирает значения из столбца 'cast' и парами помещает их в словарь pair_actors_list    
def search_pair_actors(row):
    temp_list = row['cast'].split('|')
    for i in range(len(temp_list)-1):
        for j in range(i+1, len(temp_list)):
            string = temp_list[i] + ' & ' + temp_list[j] 
            if  string not in pair_actors_list.keys():
                pair_actors_list[string]= 1
            else:
                pair_actors_list[string] += 1
    return row['cast']


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# ВАРИАНТ 1

# In[5]:


data[data['budget'] == data['budget'].max()]


# ВАРИАНТ 2

# In[6]:


data.groupby(['budget'])[['imdb_id', 'original_title']].max()


# In[7]:


answers['1'] = 'Pirates of the Caribbean: On Stranger Tides (tt1298650)'
# если ответили верно, можете добавить комментарий со значком "+"


# # 2. Какой из фильмов самый длительный (в минутах)?

# ВАРИАНТ 1

# In[8]:


data[data['runtime'] == data['runtime'].max()]


# ВАРИАНТ 2

# In[9]:


data.groupby(['runtime'])[['imdb_id', 'original_title']].max()


# In[10]:


answers['2'] = 'Gods and Generals (tt0279111)'


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# ВАРИАНТ 1

# In[11]:


data[data['runtime'] == data['runtime'].min()]


# ВАРИАНТ 2

# In[12]:


data.groupby(['runtime'])[['imdb_id', 'original_title']].min()
answers['3'] = 'Winnie the Pooh (tt1449283)'


# # 4. Какова средняя длительность фильмов?
# 

# In[13]:


round(data['runtime'].mean())


# In[14]:


answers['4'] = '110'


# # 5. Каково медианное значение длительности фильмов? 

# In[15]:


round(data['runtime'].median())


# In[16]:


answers['5'] = '107'


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# ВАРИАНТ 1

# In[17]:


# лучше код получения столбца profit вынести в Предобработку что в начале
data[data['profit']>0].groupby(['profit'])[['imdb_id', 'original_title']].max()


# ВАРИАНТ 2

# In[18]:


data[data['profit'] == data['profit'].max()]


# In[19]:


answers['6'] = 'Avatar (tt0499549)'


# # 7. Какой фильм самый убыточный? 

# ВАРИАНТ 1

# In[20]:


data[data['profit'] == data['profit'].min()]


# ВАРИАНТ 2

# In[21]:


data[data['profit']<0].groupby(['profit'])[['imdb_id', 'original_title']].min()


# In[22]:


answers['7'] = 'The Lone Ranger (tt1210819)'


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[23]:


len(data[data['profit'] > 0])


# In[24]:


answers['8'] = '1478'


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# ВАРИАНТ 1

# In[25]:



data2 = data[data['release_year'] == 2008]
data2[(data2['revenue'] == data2['revenue'].max())]


# ВАРИАНТ 2

# In[26]:


data[data['release_year'] == 2008].groupby(['revenue'])[['imdb_id', 'original_title']].max()


# In[27]:


answers['9'] = 'The Dark Knight (tt0468569)'


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# ВАРИАНТ 1

# In[28]:


data2 = data[(2012 <= data['release_year']) & (data['release_year'] <= 2014)]
data2[(data2['profit'] == data2['profit'].min())]


# ВАРИАНТ 2

# In[29]:


data[(2012 <= data['release_year']) & (data['release_year'] <= 2014)].groupby(['profit'])[['imdb_id', 'original_title']].min()


# In[30]:


answers['10'] = 'The Lone Ranger (tt1210819)'


# # 11. Какого жанра фильмов больше всего?

# In[31]:


# эту задачу тоже можно решать разными подходами, попробуй реализовать разные варианты
# если будешь добавлять функцию - выноси ее в предобработку что в начале


# ВАРИАНТ 1

# In[32]:


genre_list = [] # Создаем пустой список для жанров
data.apply(search_genres_v1, axis=1) # помещаем в данный список все значения жанров из столбца 'genres'
# плюс этого метода - можно вывести отсортировать значения счетчика

c = Counter()
for item in genre_list:
    c[item]+=1

display(c.most_common())


# ВАРИАНТ 2

# In[33]:


genres_list={} # создаем пустой словарь
data.apply(search_genres_v2, axis=1) # помещаем в данный словарь значения жанров, подсчитываю количество их упоминаний

genres_list = sorted(genres_list.items(), key=operator.itemgetter(1), reverse = True)
genres_list


# In[34]:


answers['11'] = 'Drama'
# проверка
len(data[data.genres.str.contains('Drama')])


# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[35]:


genres_list={}
# делаем выборку фильмов чья прибыль больше 0, и затем проверяем какого жанра они чаще всего 
sample = data[data['profit'] > 0].groupby(['profit'])[['imdb_id', 'original_title', 'profit', 'genres']].max()
sample.apply(search_genres_v2, axis=1)

genres_list = sorted(genres_list.items(), key=operator.itemgetter(1), reverse = True)
genres_list


# In[36]:


answers['12'] = 'Drama'


# # 13. У какого режиссера самые большие суммарные кассовые сбооры?

# In[37]:


data.groupby(['director'])['revenue'].sum().sort_values(ascending=False)


# In[38]:


answers['13'] = 'Peter Jackson'


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[39]:


director_list = {}

# делаем выборку фильмов в стиле Action
sample = data[data.genres.str.contains('Action', na=False)]
# проверяем, какой режиссер чаще других снимал такие фильмы
sample.apply(counting_director, axis=1)



director_list = sorted(director_list.items(), key=operator.itemgetter(1), reverse = True)
director_list[0:9]


# In[40]:


answers['14'] = 'Robert Rodriguez'


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# ВАРИАНТ 1

# In[41]:


# Делаем выборку фильмов,вышедших в 2012 году и не принесших убытка: 
list_films = data[(data['release_year'] == 2012) & (data['profit'] > 0)]

# Делаем из этих фильмов выборку фильмов, чьи кассовые сборы были выше среднего
sample = list_films.loc[list_films['revenue'] > list_films['revenue'].mean()]

actor_list = {}
# проверяем, какие актеры, и как часто снимались в этих фильмах
sample.apply(search_actor, axis=1)
actor_list = sorted(actor_list.items(), key=operator.itemgetter(1), reverse = True)
actor_list[0:13]


# ВАРИАНТ 2

# In[42]:


# Делаем выборку фильмов,вышедших в 2012 году: 
list_films = data[data['release_year'] == 2012]

actor_list = {}
# проверяем, какие актеры, и как часто снимались в этих фильмах
list_films.apply(search_actor, axis=1)
actor_list = pd.Series(actor_list)

# подсчитываем сумму сборов для каждого из актеров, по фильмам в которых он снимался
for actor in actor_list.index:
    actor_list[actor] = list_films['revenue'][list_films['cast'].map(lambda x: True if actor in x else False)].sum()

#actor_list = sorted(actor_list.items(), key=operator.itemgetter(1), reverse = True)
actor_list = pd.DataFrame(actor_list)
actor_list.sort_values(0, ascending = False)


# In[43]:


answers['15'] = 'Chris Hemsworth'


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[44]:


# Делаем выборку фильмов, чей бюджет был выше среднего
sample = data.loc[data['budget'] > data['budget'].mean()]

actor_list = {}
# проверяем, какие актеры, и как часто снимались в этих фильмах
sample.apply(search_actor, axis=1)
actor_list = sorted(actor_list.items(), key=operator.itemgetter(1), reverse = True)
actor_list[:9]


# In[45]:


answers['16'] = 'Matt Damon'


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[46]:


# делаем выборку фильмов c Николасом Кейджем
sample = data[data.cast.str.contains('Nicolas Cage', na=False)]

genres_list = {}
sample.apply(search_genres_v2, axis=1)

genres_list = sorted(genres_list.items(), key=operator.itemgetter(1), reverse = True)
genres_list


# In[47]:


answers['17'] = 'Action'


# # 18. Самый убыточный фильм от Paramount Pictures

# In[48]:


sample = data[data.production_companies.str.contains('Paramount Pictures', na=False)]
sample[sample['profit'] == sample['profit'].min()]


# In[49]:


answers['18'] = 'K-19: The Widowmake (tt0267626)'


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[50]:


data.groupby(['release_year'])[['revenue']].sum().sort_values(by='revenue', ascending=False)


# In[51]:


answers['19'] = '2015'


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[52]:


sample = data[data.production_companies.str.contains('Warner Bros', na=False)]
sample.groupby(['release_year'])[['profit']].sum().sort_values(by='profit', ascending=False)


# In[53]:


answers['20'] = '2014'


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# ВАРИАНТ 1

# In[54]:


month_list ={} # создаем пустой словарь для счетчиков месяцев
# указываем в каком диапазоне месяцев используем функцию перебора месяцев
start_month = 1
end_month = 13    

data.apply(search_month_v1, axis=1)

month_list = sorted(month_list.items(), key=operator.itemgetter(1), reverse = True)
month_list


# ВАРИАНТ 2

# In[55]:


month_list ={} # создаем пустой словарь для счетчиков месяцев
# указываем в каком диапазоне месяцев используем функцию перебора месяцев
start_month = 1
end_month = 13    

data.apply(search_month_v2, axis=1)

month_list = sorted(month_list.items(), key=operator.itemgetter(1), reverse = True)
month_list


# In[56]:


answers['21'] = 'сентябрь'


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# ВАРИАНТ 1

# In[57]:


month_list ={} # создаем пустой словарь для счетчиков месяцев
# указываем в каком диапазоне месяцев используем функцию перебора месяцев
start_month = 1
end_month = 13    

count_films = 0
data.apply(search_month_v2, axis=1)

for i in range (6,9):
    count_films += month_list.get(i)
    
count_films    


# ВАРИАНТ 2

# In[58]:


sample = data[(data['date_of_release'].dt.month == 6) | (data['date_of_release'].dt.month == 7) | (data['date_of_release'].dt.month == 8)]
len(sample)


# ВАРИАНТ 3

# In[59]:


sample = data[(data['release_date'].str.match('6/', na=False)) | (data['release_date'].str.match('7/', na=False)) | (data['release_date'].str.match('8/', na=False))]
len(sample)


# In[60]:


answers['22'] = '450'


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[61]:


director_list = {}
sample = data[(data['date_of_release'].dt.month == 12) | (data['date_of_release'].dt.month == 1) | (data['date_of_release'].dt.month == 2)]
sample.apply(counting_director, axis=1)

director_list = sorted(director_list.items(), key=operator.itemgetter(1), reverse = True)
director_list[0:9]


# In[62]:


answers['23'] = 'Peter Jackson'


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[63]:


#решение подсмотрено в slack - составит общий список студий, и посчитать сколько среднее значение длины названий фильмов у студии
company_list={} # создаем пустой словарь
data.apply(search_company, axis=1) # помещаем в данный словарь названия студий, подсчитывая количество их упоминаний

# переводим словарь в Series
company_list = pd.Series(company_list)

# В цикле перебираем название студий и считаем mean длины названий фильмов, если данная студия их снимала 
for company in company_list.index:
    company_list[company] = data['len_title'][data['production_companies'].map(lambda x: True if company in x else False)].mean()

company_list = sorted(company_list.items(), key=operator.itemgetter(1), reverse = True)
company_list[0:9]


# In[64]:


answers['24'] = 'Four By Two Productions'


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[65]:


#решение подсмотрено в slack - составит общий список студий, и посчитать сколько среднее значение длины названий фильмов у студии
company_list={} # создаем пустой словарь
data.apply(search_company, axis=1) # помещаем в данный словарь названия студий, подсчитывая количество их упоминаний

# переводим словарь в Series
company_list = pd.Series(company_list)

# В цикле перебираем название студий и считаем mean описания фильмов, если данная студия их снимала 
for company in company_list.index:
    company_list[company] = data['len_overview'][data['production_companies'].map(lambda x: True if company in x else False)].mean()

company_list = sorted(company_list.items(), key=operator.itemgetter(1), reverse = True)
company_list[0:9]


# In[66]:


answers['25'] = 'Midnight Picture Show'


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[67]:



list_vote = []
def vote(row):
    list_vote.append(data['vote_average'])
    return row['vote_average']

data.apply(vote, axis=1)
range_vote = np.percentile(list_vote,99) # gives the 99th percentile

sample = data.loc[data['vote_average'] > range_vote].reset_index(drop=True)
sample


# In[68]:


answers['26'] = 'Inside Out, The Dark Knight, 12 Years a Slave'


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[69]:


pair_actors_list = {}
data.apply(search_pair_actors, axis=1)
pair_actors_list = sorted(pair_actors_list.items(), key=operator.itemgetter(1), reverse = True)
pair_actors_list[0:9]


# In[70]:


answers['27'] = 'Daniel Radcliffe & Rupert Grint'


# # Submission

# In[71]:


# в конце можно посмотреть свои ответы к каждому вопросу
answers


# In[72]:


# и убедиться что ни чего не пропустил)
len(answers)


# In[ ]:




