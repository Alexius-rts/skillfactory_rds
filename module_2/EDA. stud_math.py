#!/usr/bin/env python
# coding: utf-8

# ### Вводная

#     Вас пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН,
# чья миссия состоит в повышении уровня благополучия детей по всему миру.
# 
#     Cуть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике,
# чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
# 
#     И сделать это можно с помощью модели, которая предсказывала бы результаты госэкзамена по математике для каждого ученика
# школы (вот она, сила ML!). Чтобы определиться с параметрами будущей модели, проведите разведывательный анализ данных
# и составьте отчёт по его результатам. 
# 
# 
#     Описание датасета
#     Посмотрим на переменные, которые содержит датасет:
# 
# 
# 
# 1 school — аббревиатура школы, в которой учится ученик
# 
# 2 sex — пол ученика ('F' - женский, 'M' - мужской)
# 
# 3 age — возраст ученика (от 15 до 22)
# 
# 4 address — тип адреса ученика ('U' - городской, 'R' - за городом)
# 
# 5 famsize — размер семьи('LE3' <= 3, 'GT3' >3)
# 
# 6 Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 
# 7 Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# 8 Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# 9 Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# 10 Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# 11 reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 
# 12 guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 
# 13 traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 
# 14 studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 
# 15 failures — количество внеучебных неудач (n, если 1<=n<=3, иначе 0)
# 
# 16 schoolsup — дополнительная образовательная поддержка (yes или no)
# 
# 17 famsup — семейная образовательная поддержка (yes или no)
# 
# 18 paid — дополнительные платные занятия по математике (yes или no)
# 
# 19 activities — дополнительные внеучебные занятия (yes или no)
# 
# 20 nursery — посещал детский сад (yes или no)
# 
# 21 higher — хочет получить высшее образование (yes или no)
# 
# 22 internet — наличие интернета дома (yes или no)
# 
# 23 romantic — в романтических отношениях (yes или no)
# 
# 24 famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 
# 25 freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 
# 26 goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 
# 27 health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 
# 28 absences — количество пропущенных занятий
# 
# 29 score — баллы по госэкзамену по математике

# In[122]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from itertools import combinations
from scipy.stats import ttest_ind


pd.set_option('display.max_rows', 30)  # показывать больше строк
pd.set_option('display.max_columns', 30)  # показывать больше колонок

df = pd.read_csv('stud_math.csv')


# In[123]:


# Предобработка.


# Функция по сбору информации по каждому столбцу

def info_column(column):
    print('Число уникальных значений:', df[column].nunique(), '\n')
    print('Число упоминания каждого значения: \n',
          df[column].value_counts(), '\n')
    print('Число пустых значений в столбце:',
          (df[column].isnull()).sum(), '\n')
    return  # df.loc[:, [column]].info()


# Пропишем функцию расчета IQR и квартилей. И прорисовку  их графика

def IQR_perc(column):
    IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
    perc25 = df[column].quantile(0.25)
    perc75 = df[column].quantile(0.75)

    print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75), "IQR: {}, ".format(IQR),
          "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))

    df[column].loc[df[column].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(
        bins=20, range=(start_point, end_point), label='IQR')
    return plt.legend()


# Простановка в столбец вместо пропусков наиболее часто встречаемое значение - mode

def insert_mode(column):
    df[column].fillna(df[column].mode()[0], inplace=True)
    return


# Простановка вместо пропусков округленное среднее значение - mean (!В результате округления оно в принципе == mode!)

def insert_mean(column):
    mean_value = round(df[column].mean())
    df[column].fillna(float(mean_value), inplace=True)
    return


# прорисовка гистограммы исходных данных
def hist_source_data(column):
    df[column].hist(alpha=0.4, bins=20, range=(
        start_point, end_point), label='Исходные данные {}'. format(column))
    plt.legend()
    return


# In[124]:


len(df.columns)


# In[125]:


df.head(20)


# ### Первичный отсмотр данных

# In[126]:


df.info()


# In[127]:


df.columns = map(str.lower, df.columns)


# ###  Первичный анализ данных в столбцах.

# In[128]:


# В виду того, что значения столбца "studytime,granular" равны значениям столбца "study",
# умноженные на "-3", удаляем его из датафрейма
df = df.drop(['studytime, granular'], axis=1)


# In[129]:


# описываем функцию, заменяющую в строковых столбцах  пробел на None

def clear_column(column):
    return df[column].astype(str).apply(lambda x: None if x.strip() == '' else x)


for col in list_obj:
    clear_column(col)


# ## 1. School

# In[130]:


column = 'school'
info_column(column)
# School имеет всего 2 значения: GP и MS. GP в 7.5 раз  больше, чем MS. Пустых значений нет


# In[131]:


# Задаем границы графика распределения
start_point = 0
end_point = 5
hist_source_data(column)


# ## 2. Sex

# In[132]:


column = 'sex'
info_column(column)
# Пустых значений нет. Девочек чуть больше, чем мальчиков


# In[133]:


hist_source_data(column)


# ## 3. Age

# In[134]:


column = 'age'
info_column(column)
# Возраст от 16 до 22.Пустых значений нет


# In[135]:


# Задаем границы графика распределения
start_point = 14
end_point = 23

IQR_perc(column)  # вызываем функцию расчета IQR и квартилей

# прорисовывыаем поверх распределение исходных данных
hist_source_data(column)

# Т.к. в описании указано, что возраст учащихся до 22 лет включительно, не будем удалять 22 года как выброс


# ## 4. Address

# In[136]:


column = 'address'
info_column(column)


# In[137]:


df[df.address.isnull()]


# In[138]:


# Находим среднее  timetravel для каждого типа адреса
mean_time_r = df.groupby('address')['traveltime'].mean().loc['R']
mean_time_u = df.groupby('address')['traveltime'].mean().loc['U']


# In[139]:


# Заполняем address следующим образом: сравниваем абсолютные значения разностей timetravel
# и его средних для разных типов address
for i in df[df.address.isnull()].index:
    if not np.isnan(df.traveltime.loc[i]):
        if abs(df.traveltime.loc[i]-mean_time_r) < abs(df.traveltime.loc[i]-mean_time_u):
            df.address.loc[i] = 'R'
        else:
            df.address.loc[i] = 'U'


# In[140]:


# Удаляем оставшиеся 3 строки с пустым address и traveltime - данные столбцы напрямую взаимосвязаны, и мы не сможем определить какие значения присвоить пропущенным значениям
df.dropna(subset=['address'], inplace=True)


# In[141]:


# Задаем границы графика распределения
start_point = 0
end_point = 5
hist_source_data(column)


# ## 5. Famsize

# In[142]:


column = 'famsize'
info_column(column)


# In[143]:


# присвоим пропущеным значениям наиболее часто встречающееся значение
insert_mode(column)
hist_source_data(column)


# ## 6.  Pstatus

# In[144]:


column = 'pstatus'
info_column(column)


# In[145]:


# присвоим пропущеным значениям наиболее часто встречающееся значение
insert_mode(column)
hist_source_data(column)


# ## 7. Medu

# In[146]:


column = 'medu'
info_column(column)


# In[147]:


# присвоим пропускам округленное среднее значение (зачастую оно же mode) medu
insert_mean(column)

start_point = 0
end_point = 5

IQR_perc(column)
hist_source_data(column)

# Выбросов нет


# ## 8. Fedu

# In[148]:


column = 'fedu'
info_column(column)


# In[149]:


# Удалим строку с явно неправильным значением
df = df.drop(np.where(df.fedu == 40.0)[0])


# In[150]:


insert_mean(column)

start_point = 0
end_point = 5

IQR_perc(column)
hist_source_data(column)

# 0  - в пределах нормы: не удаляем


# ## 9. Mjob

# In[151]:


column = 'mjob'
info_column(column)


# In[152]:


# присвоим пропущеным значениям работы отцов наиболее часто встречающееся значение - other
insert_mode(column)
hist_source_data(column)


# ## 10. Fjob

# In[153]:


column = 'fjob'
info_column(column)


# In[154]:


# присвоим пропущеным значениям работы матерей наиболее часто встречающееся значение - other
insert_mode(column)
hist_source_data(column)


# ## 11. Reason

# In[155]:


column = 'reason'
info_column(column)


# In[156]:


# присвоим пропущенным значениям причины выбора школы наиболее часто встречающееся значение - course
insert_mode(column)
hist_source_data(column)


# ## 12. Guardian

# In[157]:


column = 'guardian'
info_column(column)


# In[158]:


# присвоим пропущенным значениям наиболее часто встречающееся значение - mother
insert_mode(column)
hist_source_data(column)


# ## 13. traveltime

# In[159]:


column = 'traveltime'
info_column(column)


# In[160]:


# присвоим пропущенным значениям округленное среднее значение
insert_mean(column)

IQR_perc(column)
hist_source_data(column)
# 4 - в пределах нормы. Не удаляем


# ## 14. Studytime

# In[161]:


column = 'studytime'
info_column(column)


# In[162]:


# присвоим пропущенным значениям округленное среднее значение
insert_mean(column)

IQR_perc(column)
hist_source_data(column)
# 4 - в пределах нормы. Не удаляем


# ## 15. Failures

# In[163]:


column = 'failures'
info_column(column)


# In[164]:


# присвоим пропущенным значениям округленное среднее значение
insert_mean(column)

IQR_perc(column)
hist_source_data(column)
# 1-3  в пределах нормы. Не удаляем


# ## 16. schoolsup

# In[165]:


column = 'schoolsup'
info_column(column)


# In[166]:


insert_mode(column)
hist_source_data(column)


# ## 17. Famsup

# In[167]:


column = 'famsup'
info_column(column)


# In[168]:


insert_mode(column)
hist_source_data(column)


# ## 18. Paid

# In[169]:


column = 'paid'
info_column(column)


# In[170]:


insert_mode(column)
hist_source_data(column)


# ## 19. Activities

# In[171]:


column = 'activities'
info_column(column)


# In[172]:


insert_mode(column)
hist_source_data(column)


# ## 20. Nursery

# In[173]:


column = 'nursery'
info_column(column)


# In[174]:


insert_mode(column)
hist_source_data(column)


# ## 21. Higher

# In[175]:


column = 'higher'
info_column(column)


# In[176]:


insert_mode(column)
hist_source_data(column)


# ## 22. Internet

# In[177]:


column = 'internet'
info_column(column)


# In[178]:


insert_mode(column)
hist_source_data(column)


# ## 23. Romantic

# In[179]:


column = 'romantic'
info_column(column)


# In[180]:


insert_mode(column)
hist_source_data(column)


# ## 24. Famrel

# In[181]:


column = 'famrel'
info_column(column)


# In[182]:


# удаляем строку с неправильным значением -1
df = df.drop(np.where(df.famrel == -1.0)[0])


# In[183]:


insert_mean(column)
IQR_perc(column)
hist_source_data(column)
# оставляем все данные в  допустимом интервале (0.0 - 5.0)


# ## 25. Freetime

# In[184]:


column = 'freetime'
info_column(column)


# In[185]:


insert_mean(column)
IQR_perc(column)
hist_source_data(column)
# оставляем все данные в  допустимом интервале (0.0 - 5.0)


# ## 26. Goout

# In[186]:


column = 'goout'
info_column(column)


# In[187]:


insert_mean(column)
IQR_perc(column)
hist_source_data(column)


# ## 27. Health

# In[188]:


column = 'health'
info_column(column)


# In[189]:


insert_mean(column)
IQR_perc(column)
hist_source_data(column)


# ## 28. Absences

# In[190]:


column = 'absences'
info_column(column)


# In[191]:


# зададим пределы графика гистограммы до явного выброса: 40
star_point = 0
end_point = 40

insert_mean(column)
IQR_perc(column)
hist_source_data(column)


# In[192]:


# Т.к. за пределами стандартных границ выбросов имеется достаточно данных, вручную увеличим размах усов выбросов:
IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
perc25 = df[column].quantile(0.25)
perc75 = df[column].quantile(0.75)

print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75), "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 2.5*IQR, l=perc75 + 2.5*IQR))

# подсчитаем количество значений, лежащих за пределами удлиненных границ выбросов:
print('Количество значений за пределами удлиненных границ выбросов:',
      len(df.loc[df['absences'] >= 28]))


# In[193]:


# Удалим данные выбросы
df = df.drop(df[df['absences'] >= 28].index)


# In[194]:


IQR_perc(column)
hist_source_data(column)


# ## 29. Score

# In[195]:


column = 'score'
info_column(column)


# In[196]:


insert_mode(column)


# In[197]:


IQR_perc(column)
hist_source_data(column)


# In[198]:


# 0 удаляем, т.к. это видимо те кто не явился, или не сдавал работу на проверку.

df = df.drop(df[df['score'] == 0].index)


# In[199]:


df


# ## Корреляционный анализ

# In[201]:


# создаем список столбцов с числовыми значениями
list_num = ['age', 'absences','score']
# создаем список столбцов со категориальными значениями
list_obj = list(set(df.columns).difference(list_num))


# In[202]:


# датафейм из  столбцов с числовыми значениями
df_num = df.loc[:, df.columns.isin(list_num)]


# In[203]:


sns.pairplot(df_num, kind='reg')


# In[204]:


correlation = df_num.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
# Скорррелированных столбцов нет, все столбцы оставляем


# ## Анализ номинативных переменных

# In[205]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=df.loc[df.loc[:, column].isin(
                    df.loc[:, column].value_counts().index[:10])],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[206]:


for col in list_obj:
    get_boxplot(col)


# In[207]:


# По графикам похоже, что параметры reason, school, paid,guardian не влияют на результаты экзамена по математике


# In[212]:



def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'],
                     df.loc[df.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[213]:


for col in list_obj:
    get_stat_dif(col)


# Как мы видим, в категориальных данных серьёзно отличаются 8 параметров: 'sex', 'address', 'medu', 'fedu',  'schoolsup',  'failures', 'studytime', 'goout'
# 
# Оставим эти переменные в датасете для дальнейшего построения модели. 
# 
# Итак, в нашем случае важные переменные, которые, возможно, оказывают влияние на оценку экзамена по математике, это: 
# 'age', 'absences', 'sex', 'failures', 'address', 'medu', 'fedu',  'schoolsup', 'studytime', 'goout'.
# И столбец целевой переменной - 'score'
# 

# In[214]:


column_for_model = df.loc[:, ['age', 'absences', 'sex', 'failures', 'address', 'medu', 'fedu',  'schoolsup', 'studytime', 'goout', 'score']]
column_for_model.head()


# ## Выводы.

# - В данных мало пустых значений: максимальные пропуски были в столбцах:  pstatus - 12%, famsup - 10% и paid -11%
# - выброcы вне пределов нормальных данных были только в одной строке в famrel (-1.0), что говорит о том, что данные достаточно чистые 
# - Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это 'age', 'absences', 'sex', 'failures', 'address', 'medu', 'fedu',  'schoolsup', 'studytime', 'goout', 'score'

# In[ ]:




