#!/usr/bin/env python
# coding: utf-8

# Вашей задачей будет построить скоринг модель для вторичных клиентов банка, которая бы предсказывала вероятность дефолта клиента. Для этого нужно будет определить значимые параметры заемщика.
# 
# 
# данные:
# 
# Описания полей
# - client_id - идентификатор клиента
# - education - уровень образования
# - sex - пол заемщика
# - age - возраст заемщика
# - car - флаг наличия автомобиля
# - car_type - флаг автомобиля иномарки
# - decline_app_cnt - количество отказанных прошлых заявок
# - good_work - флаг наличия “хорошей” работы
# - bki_request_cnt - количество запросов в БКИ
# - home_address - категоризатор домашнего адреса
# - work_address - категоризатор рабочего адреса
# - income - доход заемщика
# - foreign_passport - наличие загранпаспорта
# - sna - связь заемщика с клиентами банка
# - first_time - давность наличия информации о заемщике
# - score_bki - скоринговый балл по данным из БКИ
# - region_rating - рейтинг региона
# - app_date - дата подачи заявки
# - default - флаг дефолта по кредиту

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
PATH_to_file = '/kaggle/input/sf-dst-scoring/'


# In[ ]:


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42


# In[ ]:


# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# ### Импорт библиотек

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings("ignore")

import datetime as DT


# ### Определение базовых функций

# In[ ]:


# Функция по сбору информации по каждому столбцу


def my_describe(df):
    """Отображение описательных статистик датафрейма в удобной форме"""
    temp = {}
    temp['Имя признака'] = list(df.columns)
    temp['Тип'] = df.dtypes
    temp['Всего значений'] = df.describe(include='all').loc['count']
    temp['Число пропусков'] = df.isnull().sum().values 
    temp['Кол-во уникальных'] = df.nunique().values
    temp['Минимум'] = df.describe(include='all').loc['min']
    temp['Максимум'] = df.describe(include='all').loc['max']
    temp['Среднее'] = df.describe(include='all').loc['mean']
    temp['Медиана'] = df.describe(include='all').loc['50%']
    temp = pd.DataFrame.from_dict(temp, orient='index')
    display(temp.T)
    return

# прорисовка гистограммы исходных данных
def graph_source_data(column):
    start_point = df[column].min()
    end_point = df[column].max()
    fig, ax = plt.subplots(1,2)
    ax[0].hist(df[column], alpha=0.4, bins=40, range=(
        start_point, end_point), label='Исходные данные {}'. format(column))
    ax[0].set_title(column)
    ax[1].boxplot(df[column])
    plt.legend()
    return


# Пропишем функцию расчета IQR и квартилей. И прорисовку  их графика

def IQR_perc(df):
    temp = {}
    temp['Имя признака'] = num_cols
    start_point = df[num_cols].min()
    end_point = df[num_cols].max()
    #IQR = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    #perc25 = df[num_cols].quantile(0.25)
    #perc75 = df[num_cols].quantile(0.75)
    temp['IQR'] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    temp['perc25'] = df[num_cols].quantile(0.25)
    temp['perc75'] = df[num_cols].quantile(0.75)
    temp['Л. граница выбросов'] = df[num_cols].quantile(0.25) - 1.5*(df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25))
    temp['П. граница выбросов'] =df[num_cols].quantile(0.75) + 1.5*(df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25))
    temp = pd.DataFrame.from_dict(temp, orient='index')
    display(temp.T)
    return



def show_confusion_matrix(y_true, y_pred):
    """Функция отображает confusion-матрицу"""
    color_text = plt.get_cmap('GnBu')(1.0)
    class_names = ['Дефолт', 'НЕ дефолт']
    cm = confusion_matrix(y_true, y_pred)
    cm[0,0], cm[1,1] = cm[1,1], cm[0,0]
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), title="Матрица ошибок")
    ax.title.set_fontsize(15)
    sns.heatmap(df, square=True, annot=True, fmt="d", linewidths=1, cmap="GnBu")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=12)
    ax.set_ylabel('Предсказанные значения', fontsize=14, color = color_text)
    ax.set_xlabel('Реальные значения', fontsize=14, color = color_text)
    b, t = plt.ylim()
    plt.ylim(b+0.5, t-0.5)
    fig.tight_layout()
    plt.show()
    
    
def all_metrics(y_true, y_pred, y_pred_prob):
    """Функция выводит в виде датафрейма значения основных метрик классификации"""
    dict_metric = {}
    P = np.sum(y_true==1)
    N = np.sum(y_true==0)
    TP = np.sum((y_true==1)&(y_pred==1))
    TN = np.sum((y_true==0)&(y_pred==0))
    FP = np.sum((y_true==0)&(y_pred==1))
    FN = np.sum((y_true==1)&(y_pred==0))
    
    dict_metric['P'] = [P,'Дефолт']
    dict_metric['N'] = [N,'БЕЗ дефолта']
    dict_metric['TP'] = [TP,'Истинно дефолтные']
    dict_metric['TN'] = [TN,'Истинно НЕ дефолтные']
    dict_metric['FP'] = [FP,'Ложно дефолтные']
    dict_metric['FN'] = [FN,'Ложно НЕ дефолтные']
    dict_metric['Accuracy'] = [accuracy_score(y_true, y_pred),'Accuracy=(TP+TN)/(P+N)']
    dict_metric['Precision'] = [precision_score(y_true, y_pred),'Точность = TP/(TP+FP)'] 
    dict_metric['Recall'] = [recall_score(y_true, y_pred),'Полнота = TP/P']
    dict_metric['F1-score'] = [f1_score(y_true, y_pred),'Среднее гармоническое Precision и Recall']
    dict_metric['ROC_AUC'] = [roc_auc_score(y_true, y_pred_prob),'ROC-AUC']    

    temp_df = pd.DataFrame.from_dict(dict_metric, orient='index', columns=['Значение', 'Описание метрики'])
    display(temp_df) 

def show_roc_curve(y_true, y_pred_prob):
    """Функция отображает ROC-кривую"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure()
    plt.plot([0, 1], label='Случайный классификатор', linestyle='--')
    plt.plot(fpr, tpr, label = 'Логистическая регрессия')
    plt.title('Логистическая регрессия ROC AUC = %0.3f' % roc_auc_score(y_true, y_pred_prob))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()


# ### DATA

# In[ ]:


DATA_DIR = '/kaggle/input/sf-dst-scoring/'
df_train = pd.read_csv(DATA_DIR+'/train.csv')
df_test = pd.read_csv(DATA_DIR+'test.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')


# In[ ]:


df_train.info()


# In[ ]:


df_train.head(5)


# In[ ]:


# Построит диаграмму для переменной  'default'  
ax = sns.countplot(x="default", data=df_train)


# In[ ]:


df_test.info()


# In[ ]:


df_test.head(5)


# In[ ]:


sample_submission.head(5)


# In[ ]:


a = sample_submission.reset_index()
a


# In[ ]:


sample_submission.info()


# In[ ]:


# ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['default'] = 0 # в тесте у нас нет значения default, мы его должны предсказать, по этому пока просто заполняем нулями

df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем


# 
# # **Очистка и подготовка данных**

# ## Предобработка данных

# In[ ]:


# описываем функцию, заменяющую в строковых столбцах  пробел на None

def clear_column(column):
    return df[column].astype(str).apply(lambda x: None if x.strip() == '' else x)

for col in df.columns:
    clear_column(col)


# ## 1. Первичный отсмотр данных

# In[ ]:


df.info()


# ### Работа с пропусками

# In[ ]:


# построим карту пропусков данных
f, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=True)
df.isnull().sum()


# У наличия пропусков могут быть разные причины, но пропуски нужно либо заполнить, либо исключить из набора полностью. Но с пропусками нужно быть внимательным, даже отсутствие информации может быть важным признаком!
# По этому перед обработкой NAN лучше вынести информацию о наличии пропуска как отдельный признак

# In[ ]:



# Подсчитаем общее число строк с пропусками
df.shape[0] - df.dropna().shape[0]


# In[ ]:


column = 'education'
print(f'Общая доля пропусков в столбце {column}:', round((df[column].isnull().value_counts(normalize=True) * 100),2)[1], '%')


# In[ ]:


# посмотрим значения 'education'
df.education.value_counts().plot.barh()
plt.show()


# In[ ]:


#пропуски наблюдаем только в столбц "education". Создадим новый столбец с признаком пропуска данных в education
df['education_isNAN'] = pd.isna(df['education']).astype('uint8')


# Заполним пропуски, исходя из того, что уровень образования должен коррелироваться с уровнем дохода:

# In[ ]:


# посмотрим каков для каждого типа образования средний доход
edu_income_mean = round(df.groupby(['education']).income.mean())
# отсортируем получившийся датафрейм
edu_income_mean.sort_values(inplace=True)


# In[ ]:


edu_income_mean
# SCH - school, ...., ACD - academic#
plt.figure(figsize=(6, 6))
plt.ylim(top=200_000)
ax = sns.boxplot(x="education", y="income", data=df)


# In[ ]:


# заполним пропуски в 'education' словом 'Unknown'
df['education'] = df['education'].fillna('Unknown')
# нулевым значениям в data['education'] присваиваем значение исходя из значения дохода в сравнении с вилкой среднего дохода
# для каждого типа образования c учетом датафрейма edu_income_mean


def insert_education(row):
    if row['education'] == 'Unknown':
        if row['income'] >= (edu_income_mean.iloc[4]+edu_income_mean.iloc[3])/2:
            return edu_income_mean.index[4]
        if (edu_income_mean.iloc[3]+edu_income_mean.iloc[2])/2 <= row['income'] < (edu_income_mean.iloc[4]+edu_income_mean.iloc[3])/2:
            return edu_income_mean.index[3]
        if (edu_income_mean.iloc[2]+edu_income_mean.iloc[1])/2 <= row['income'] < (edu_income_mean.iloc[3]+edu_income_mean.iloc[2])/2:
            return edu_income_mean.index[2]
        if (edu_income_mean.iloc[1]+edu_income_mean.iloc[0])/2 <= row['income'] < (edu_income_mean.iloc[2]+edu_income_mean.iloc[1])/2:
            return edu_income_mean.index[1]
        if row['income'] < (edu_income_mean.iloc[0]+edu_income_mean.iloc[1])/2:
            return edu_income_mean.index[0]
    return row['education']


df['education'] = df.apply(insert_education, axis=1)


# In[ ]:


df.education.value_counts()


# # 2. Обработка признаков

# Для начала посмотрим какие признаки у нас могут быть категориальныe, ,бинарные и числовые:

# In[ ]:


df.nunique(dropna=False)


# In[ ]:


pd.set_option('display.max_columns', None)
df.head(2)


# In[ ]:


# сформируем списки столбцов по группам исходя из типов признаков
data_cols = ['app_date']
num_cols = ['age', 'bki_request_cnt', 'decline_app_cnt', 'income', 'score_bki']
bin_cols =['sex', 'car', 'car_type', 'foreign_passport', 'good_work', 'education_isNAN'] 
cat_cols =['education', 'region_rating', 'home_address','work_address', 'sna', 'first_time']


# ### 2.1. Временные ряды

# In[ ]:


column = 'app_date'
print(df[column].min())
print (df[column].max())


# In[ ]:


# Преобразуем строковый признак 'app_date' в дату
df[column] = df[column].apply(lambda x: DT.datetime.strptime(x, '%d%b%Y').date())


# In[ ]:


# определим диапазон дат и вместо них, назначим признаком число дней от начальной (минимальной)даты.
print(df[column].min())
print(df[column].max())
start_date = df[column].min()
df[column] = df[column].apply(lambda x: (x - start_date).days)


# In[ ]:


graph_source_data(column)
num_cols.append(column)


# ### 2.2. Числовые признаки

# In[ ]:


# посмотрим параметры IQR, гистограммы и боксплоты числовых признаков
IQR_perc(df)
for column in num_cols:
    graph_source_data(column)


# ### 2.2.1 Удаление выбросов

# In[ ]:


# После удаления выбросов модель ухудшилась, плюс submission не прошел проверку по количеству строк
"""# подсчитаем количество значений, лежащих за пределами границ выбросов:

print('Количество значений за пределами границ выбросов в "bki_request_cnt":',len(df.loc[df['bki_request_cnt'] > 7.5]))
print((len(df.loc[df['bki_request_cnt'] > 7.5]) /df.shape[0])*100, '% общего датасета')

print('Количество значений за пределами границ выбросов в "decline_app_cnt":',len(df.loc[df['decline_app_cnt'] > 5]))
print((len(df.loc[df['decline_app_cnt'] > 5]) /df.shape[0])*100, '% общего датасета')

# Увеличим правую границу выбросов в 'income'в полтора раза 
print('Количество значений за пределами границ выбросов в "income":',len(df.loc[df['income'] > 135_000]))
print((len(df.loc[df['income'] > 135_000]) /df.shape[0])*100, '% общего датасета')

print('Количество значений за пределами границ выбросов в "score_bki":',len((df.loc[df['score_bki'] > -0.52907]) + (df.loc[-3.29925 > df['score_bki']])))
a =len((df.loc[df['score_bki'] > -0.52907]) + (df.loc[-3.29925 > df['score_bki']]))  
print( (len((df.loc[df['score_bki'] > -0.52907]) + (df.loc[-3.29925 > df['score_bki']]))/df.shape[0])*100, '% общего датасета')


# Удалим данные выбросы
df = df.drop(df[df['bki_request_cnt'] > 7.5].index)
df = df.drop(df[df['decline_app_cnt'] > 5].index)
df = df.drop(df[df['income'] > 135_000].index)
df = df.drop(df[df['score_bki'] > -0.52907].index)
df = df.drop(df[df['score_bki'] < -3.29925].index)     

# посмотрим параметры IQR, гистограммы и боксплоты числовых признаков
IQR_perc(df)
for column in num_cols:
    graph_source_data(column)"""


# После построения гистограмм стало очевидно, что распределения числовых переменных 'age', 'decline_app_cnt', 'bki_request_cnt', 'income' имеют тяжёлый правый хвост (у 'app_date' - левый), кроме того, почти все числовые признаки (кроме 'age') содержат выбросы, к которым чувствительна LogisticRegression.
# 
# Для того чтобы сделать распределение данных переменных более нормальным, можно работать с логарифмированными величинами этих переменных. Тогда можно избежать чувствительности к сильным отклонениям в суммах у линейных моделей.
# 
# Анализ распределений и boxplot-ов показывает, что признак 'score_bki' может быть как положительным, так и отрицательным, его логарифмировать напрямую нельзя, но его распределение и так похоже на нормальное. Возьмем логарифм от числовых признаков за исключением  'score_bki' и построим графики распределения логарифмированных переменных, потом оценим выбросы.

# In[ ]:


## только распределение 'score_bki' выглядит нормально
# прологарифмируем остальные столбцы, чтобы сделать распределение данных переменных более нормальным
num_cols.remove('score_bki')
df[num_cols] = df[num_cols].apply(lambda y: np.log(y+1))
num_cols.append('score_bki')


# In[ ]:


for column in num_cols:
    graph_source_data(column)


# После логарифмирования некоторые переменные стали менее смещёнными.

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(22, 12))
plt.subplots_adjust(wspace = 0.2)
axes = axes.flatten()
for i in range(len(num_cols)):
    sns.boxplot(x="default", y=num_cols[i], data=df, orient = 'v', ax=axes[i],  showmeans = True)


# #### Выводы:
# - age: Дефолтные клиенты в среднем: младше,
# - 'bki_request_cnt': Дефолтные клиенты в среднем имеют больше запросов в БКИ
# - 'decline_app_cnt' Дефолтные клиенты в среднем имеют большее количество отмененных заявок
# - 'income': Дефолтные клиенты в среднем имеют более низкий доход-
# - 'app_date': Дефолтные клиенты в среднем имеют более раннюю дату подачи заявки
# - 'score_bki': Дефолтные клиенты в среднем имеют более высокий скорринговый балл. Возможно это связанно с тем, 
#  что до этого более часто брали кредиты, и как следствие улучшали свою кредитную историю 

# ### 2.2.2 Корреляция

# In[ ]:


correlation = df[num_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# Признаки между собой слабо скоррелированы, а значит мы оставляем их все. Никакой признак не удаляем

# ### 2.2.3 Значимость непрерывных переменных

# В качестве меры значимости мы будем использовать значение f-статистики. Чем значение статистики выше, тем меньше вероятность того, что средние значения не отличаются, и тем важнее данный признак для нашей линейной модели.

# In[ ]:


# Проанализируем наши числовые признаки
imp_num = pd.Series(f_classif(df[num_cols], df['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')


# Видим, что наиболее важным признаком явл. score_bki - скоринговый балл по данным из БКИ, а наименее важным - возраст (age)

# ## 2.3. Бинарные признаки

# In[ ]:


# Для перевода бинарных признаков в числа мы будем использовать LabelEncoder
# В алфавитном порядке: No=0, Yes=1

label_encoder = LabelEncoder()

for column in bin_cols:
    df[column] = label_encoder.fit_transform(df[column])
    
# убедимся в преобразовании    
df.head()


# ## 2.4. Категориальные признаки

# In[ ]:


# Переведем категориальные признаки в числа
# в алфавитном порядке => education{0: 'ACD', 1: 'GRD', 2: 'PGR', 3: 'SCH', 4: 'UGR'}
label_encoder = LabelEncoder()

for column in cat_cols:
    df[column] = label_encoder.fit_transform(df[column])
df.head()


# In[ ]:


"""Для оценки значимости категориальных и бинарных переменных будем использовать функцию mutual_info_classif 
из библиотеки sklearn. Данная функция опирается на непараметрические методы, основанные на оценке энтропии 
в группах категориальных переменных."""

imp_cat = pd.Series(mutual_info_classif(df[bin_cols + cat_cols], df['default'],
                                     discrete_features =True), index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')


# Видим, что наиболее важным признаком явл. sna - связь заемщика с клиентами банка, а наименее важным -пол (sex), и вновь добавленый признак отсутствия образования(у менее чем полупроцента людей из выборки)

# ## 3. Подготовка данных к машинному обучению

# Разбиваем датасет на тренировочный и тестовый, удалив лишние столбцы

# In[ ]:


train_data = df.query('sample == 1').drop(['sample', 'client_id'], axis=1)

test_data = df.query('sample == 0').drop(['sample','default'], axis=1)
# Сохраним ID клиентов из тестового набора для  формирования Submission
id_test = test_data.client_id
# Удалим ID клиентов из тестового набора для последующего формирования признакового проостранства
test_data = test_data.drop(['client_id'], axis=1)


# переведем категориальные данные в фиктивные переменные с удалением исходных столбцов из датафрейма
train = pd.get_dummies(train_data, columns=cat_cols, dummy_na=False, dtype='uint8')


# In[ ]:


train.head(2)


# In[ ]:


y = train.default.values            # наша целевая переменная
X = train.drop(['default'], axis=1)


# In[ ]:


# Разделим данные для обучения следующим образом:
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.20, random_state=RANDOM_SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## 4. Оценка качества модели

# ### 4.1 Модель без гиперпараметров

# In[ ]:


#model = LogisticRegression()
model_one = LogisticRegression(random_state=RANDOM_SEED)
model_one.fit(X_train, y_train)

# Предсказываем вероятность и значения целевой переменной
y_pred_prob = model_one.predict_proba(X_test)[:,1]
y_pred = model_one.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)


# ##### ВЫВОДЫ: 
# Не угаданы дефолтные клиенты. Несмотря на то, что ROC-AUC высокий (эта кривая плохо оценивает эффективность алгоритма на несбалансированных данных). Значение точности, полноты и f1 сигнализирует о том что что-то не в порядке. Построенная модель очень плохая: из матрицы ошибок видно, что мы почти не угадываем дефолтных клиентов (37 из 1827). Это показывает и метрика recall = 0.020252. Таким образом, на основе выводов модели деньги будут выданы людям, которые их не смогут вернуть

# ### 4.2. Модель без гиперпараметров, но с нормализацией числовых признаков (RobastScaler)

# Попробуем выполнить нормировку с помощью RobastScaler, которая при нормализации использует медианы и   квантили, поэтому не чувствительна к выбросам и может приводить к лучшим результатам.

# In[ ]:


# При помощи RobastScaler нормируем числовые данные из тренировочного датасета сразу после разделения 
# и приводим к виду 2-мерного массива
X_num = RobustScaler().fit_transform(train_data[num_cols].values)
X_num_test = RobustScaler().fit_transform(test_data[num_cols].values)
# преобразуем категориальные данные в 2-х мерный массив, наподобие get_dummies
"""По умолчанию OneHotEncoder преобразует данные в разреженную матрицу(sparse = True), чтобы не расходовать память на хранение многочисленных 
нулей. При (sparse = False) вернет массив"""
X_cat = OneHotEncoder(sparse = False).fit_transform(train_data[cat_cols].values)
X_cat_test = OneHotEncoder(sparse = False).fit_transform(test_data[cat_cols].values)
# Объединяем (числовые, двоичные, и категориальные)
X = np.hstack([X_num, train_data[bin_cols].values, X_cat])
Test = np.hstack([X_num_test, test_data[bin_cols].values, X_cat_test])
y = train_data['default'].values

# Разбиваем датасет на тренировочный и тестовый, выделив 20% данных на валидацию
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Обучаем модель
model_two = LogisticRegression(random_state=RANDOM_SEED)

model_two.fit(X_train, y_train)

# Предсказываем
y_pred_prob = model_two.predict_proba(X_test)[:,1]
y_pred = model_two.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)


# ВЫВОД: RobustScaler не улучшил качество модели

# ### 4.3. Модель с гиперпараметрами

# In[ ]:


get_ipython().run_cell_magic('timeit', '-n1 -r1', "from sklearn.model_selection import GridSearchCV\n\nC = np.logspace(0, 4, 10)\niter_ = 50\nepsilon_stop = 1e-3\n \nhyperparameters = [\n    {'penalty': ['l1'], \n     'C': C,  \n     'max_iter':[iter_],\n     'tol':[epsilon_stop]},\n    {'penalty': ['l2'], \n     'C': C, \n     'max_iter':[iter_],\n     'tol':[epsilon_stop]},\n    {'penalty': ['none'], \n     'C': C, \n     'max_iter':[iter_],\n     'tol':[epsilon_stop]},\n]\n\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\n\n# Создаем сетку поиска с использованием 5-кратной перекрестной проверки\nclf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)\n\nbest_model = clf.fit(X_train, y_train)\n\n# View best hyperparameters\nprint('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])\nprint('Лучшее C:', best_model.best_estimator_.get_params()['C'])\nprint('Лучшее max_iter:', best_model.best_estimator_.get_params()['max_iter'])\nprint('Лучшее tol:', best_model.best_estimator_.get_params()['tol'])")


# In[ ]:


y = train.default.values            # наша целевая переменная
X = train.drop(['default'], axis=1)

# Разделим данные для обучения следующим образом:
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.20, random_state=RANDOM_SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# C=166.81
model_3 = LogisticRegression(penalty='l2', C=166.81005372000593, max_iter=50, class_weight ='balanced', tol= 0.001)
model_3.fit(X_train, y_train)

# Предсказываем вероятность и значения целевой переменной
y_pred_prob = model_3.predict_proba(X_test)[:,1]
y_pred = model_3.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)


# Recall	0.687466

# ВЫВОДЫ: 
#  - Построенная модель по сравнению с первой уже лучше определяет дефолтных клиентов, но в ней увеличилась ошибка распознавания не дефолтных клиентов. В такой ситуации банк рискует недополучить прибыль. Удалось увеличить показания Recall и Precision.
# - Удаление малозначительных признаков 'education_isNAN', 'age' и 'sex' метрики модели не  улучшило, а ухудшило, в следствие чего признаки пришлось оставить

# ### Модель 4. Полиномиальные признаки, RobastScaler и гиперпараметры

# In[ ]:


# Преобразуем числовые признаки в полиномиальные
poly = PolynomialFeatures(3)
X_num = poly.fit_transform(train_data[num_cols].values)
X_num_test =poly.fit_transform(test_data[num_cols].values)
# При помощи RobastScaler нормируем числовые данные из тренировочного датасета сразу после разделения 
# и приводим к виду 2-мерного массива

X_num = RobustScaler().fit_transform(X_num)
X_num_test = RobustScaler().fit_transform(X_num_test)
# преобразуем категориальные данные в 2-х мерный массив, наподобие get_dummies
"""По умолчанию OneHotEncoder преобразует данные в разреженную матрицу(sparse = True), чтобы не расходовать память на хранение многочисленных 
нулей. При (sparse = False) вернет массив"""
X_cat = OneHotEncoder(sparse = False).fit_transform(train_data[cat_cols].values)
X_cat_test = OneHotEncoder(sparse = False).fit_transform(test_data[cat_cols].values)
# Объединяем (числовые, двоичные, и категориальные)
X = np.hstack([X_num, train_data[bin_cols].values, X_cat])
Test = np.hstack([X_num_test, test_data[bin_cols].values, X_cat_test])
y = train_data['default'].values

# Разбиваем датасет на тренировочный и тестовый, выделив 20% данных на валидацию
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Находим лучшие гиперпараметры
from sklearn.model_selection import GridSearchCV

C = np.logspace(0, 4, 10)
iter_ = 50
epsilon_stop = 1e-3
 
hyperparameters = [
    {'penalty': ['l1'], 
     'C': C,  
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['l2'], 
     'C': C, 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['none'], 
     'C': C, 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
]

model = LogisticRegression()
model.fit(X_train, y_train)

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])
print('Лучшее max_iter:', best_model.best_estimator_.get_params()['max_iter'])
print('Лучшее tol:', best_model.best_estimator_.get_params()['tol'])


# In[ ]:


# Передача лучших гиперпараметров непосредственно в модель, что позволяет пропустить пункт обучения модели
#model_4 = best_model.best_estimator_


# Обучаем модель
model_4 = LogisticRegression(penalty='l2', C=166.81005372000593, max_iter=50, class_weight ='balanced', tol= 0.001, random_state=RANDOM_SEED)

model_4.fit(X_train, y_train)

# Предсказываем
y_pred_prob = model_4.predict_proba(X_test)[:,1]
y_pred = model_4.predict(X_test)

# Оценка качества модели
all_metrics(y_test, y_pred, y_pred_prob)
show_roc_curve(y_test, y_pred_prob)
show_confusion_matrix(y_test, y_pred)


# 

# ВЫВОДЫ: 
#  - Построенная модель при добавлении полиномиальных признаков по сравнению с третьей  дала некоторое улучшение - recall улучшился на 22 тысячных

# ### 5. Submission

# По ходу анализа набора данных  train соответствующие действия производились с набором test:
# 
# - заполнили пропуски в test.education исходя из значения дохода;
# - преобразовали признак app_date (дата подачи заявки) в разницу между датой, указанной в столбце app_date и датой подачи первой заявки (01.01.2014) в днях;
# - взяли логарифм от числовых признаков age, decline_app_cnt, bki_request_cnt, incom, app_date
# - преобразовали бинарные переменные при помощи класса LabelEncoder;
# - преобразовали признак education в численный формат;
# - стандартизировали числовые признаки(RobustScaler), а также воспользоваться dummy-кодированием для категориальных переменных

# готовим Submission на кагл

# In[ ]:


test_data.sample(5)


# In[ ]:


sample_submission


# In[ ]:


pred_prob_submission = model_4.predict_proba(Test)[:,1]

submission = pd.DataFrame({'client_id': id_test, 
                            'default': pred_prob_submission})
submission.to_csv('submission.csv', index=False)
submission


# #### В рамках имеющегося времени удалось:
# - Построить 4 модели
# - От дисбаланса классов  в выборке избавились посредство гиперпараметра class_weight ='balanced'
# - Избавиться от выбросов в числовых переменных, но это несколько ухудшило модель(~ на 2 сотые) 
