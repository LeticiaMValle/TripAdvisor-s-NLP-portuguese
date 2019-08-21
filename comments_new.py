#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:09:05 2019

@author: LeticiaValle_Mac
"""


import pandas as pd 


def get_comments(csv_name):
    
    rating_new = []
    com_sen = []
        
    
    data = pd.read_csv(csv_name) 
    comments = data['Text_select']
    rating = data['Nota_element']
    
    for rat in range(len(rating)):
        bubble = rating[rat].split()[2].split('"')
        val = int(bubble[0].split("_")[1])/10
        rating_new.append(val)
        
        if rating_new[rat] <= 2:
            com_sen.append('Negativo')
        else:
            com_sen.append('Positivo')
    
    rating_new = pd.Series(rating_new, name='rating')
    com_sen = pd.Series(com_sen, name='sentiment')
    
    comments_as_df = pd.concat([comments,rating_new,com_sen], axis=1)
    return comments_as_df



   
if __name__ == "__main__":
   
    csv_name = "reviews_tripadvisor.csv"
    
    # Retorna o dataframe com o status (positivo, negativo) 
    comments_as_df = get_comments(csv_name) 
    print(comments_as_df )
    
