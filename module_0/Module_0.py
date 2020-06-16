#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

def score_game(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1,101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print("Ваш алгоритм угадывает число в среднем за {} попыток".format(score))
    return score
 
    
def game_core_v3(number):
    '''Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того,
       больше оно или меньше нужного. Функция принимает загаданное число и возвращает число попыток'''
    count = 1
    predict = np.random.randint(1,101)
    
    # Вводим флаги: more=0 обозначает, что predict изначально не был больше загаданного числа number
    # less=0 обозначает, что predict изначально не был меньше загаданного числа number 
    more=0
    less=0
    
    while True:
        if number == predict:
            break
        
        if number > predict:
            # если number изначально был меньше predict
            if less == 1:
                predict += 1
                count += 1
            # если number ранее не был меньше predict   
            if less == 0:
                predict = int(predict + 10) # увеличиваем изначально загаданное число на 10
                count += 1
                # указываем, что number изначально больше predict
                more = 1
        
        if number < predict: 
            # если number изначально был больше predict
            if more == 1:
                predict -= 1
                count += 1
            # если number ранее не был больше predict
            if more == 0:
                predict = int(predict - 10) # уменьшаем изначально загаданное число на 10
                count += 1
                # указываем, что number изначально меньше predict
                less = 1

    return(count) # выход из цикла, если угадали 


# In[14]:


score_game(game_core_v3)


# In[ ]:




