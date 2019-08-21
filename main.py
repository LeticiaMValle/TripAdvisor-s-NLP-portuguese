# -*- coding: utf-8 -*-

"""
Created on Tue May 21 18:12:56 2019

@author: LeticiaValle_Mac
"""

from tf_idf_pt import pre_process
from tf_idf_pt import get_stop_words
from tf_idf_pt import sort_coo
from tf_idf_pt import extract_topn_from_vector
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from comments_new import get_comments
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def treinamento (docs_train, stopwords):  

    cv = CountVectorizer(ngram_range=(1,2),max_df=0.85,stop_words=stopwords,
                         max_features=10000)
    word_count_vector = cv.fit_transform(docs_train)
    
    # Realiza a operaçao de IDF do conjunto de treinamento
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    return (cv, tfidf_transformer)  

def teste (cv, docs_test, tfidf_transformer):   
    # So precisa ser feito uma vez, é a indexaçao das features
    feature_names = cv.get_feature_names()

    for n in range(len(docs_test)):
        # Pega o documento que queremos usar, no caso o comentario
        doc = docs_test[n]
         
        # Gera o tf-idf para o documento em questao 
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
         
        # Ordena os vetores tf-idf vectors em ordem descendente de score
        sorted_items = sort_coo(tf_idf_vector.tocoo())
         
        # Extrai os top n, no caso n=10
        keywords = extract_topn_from_vector(feature_names,sorted_items,7)
#         
#        # now print the results
#        print("\n===== Comentario =====")
#        print(doc)
#        print("\n=== Palavras - chave ===")
#        
        for k in keywords:
            #print(k,keywords[k])
            list_words.append(k)
        
           
        # Guarda as principais palavras de cada comentario em uma lista
    return (list_words)   

def tf_idf(docs, stopwords):
    cv = CountVectorizer(ngram_range=(1,2),max_df=0.85,stop_words=stopwords,
                         max_features=10000)
    word_count_vector = cv.fit_transform(docs)
    
    # Realiza a operaçao de IDF do conjunto de treinamento
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)   
    
    feature_names = cv.get_feature_names()

    for n in range(len(docs)):
        # Pega o documento que queremos usar, no caso o comentario
        doc = docs[n]
         
        # Gera o tf-idf para o documento em questao 
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

        # Ordena os vetores tf-idf vectors em ordem descendente de score
        sorted_items = sort_coo(tf_idf_vector.tocoo())
         
        # Extrai os top n, no caso n=10
        keywords = extract_topn_from_vector(feature_names,sorted_items,10)
         
#        # now print the results
#        print("\n===== Comentario =====")
#        print(doc)
#        print("\n=== Palavras - chave ===")
#        
        for k in keywords:
#            print(k,keywords[k])
            list_words.append(k)
            
    return (list_words)
    
def count_methodos (list_words): 
    
    counter = collections.Counter(list_words) #Metodo de contagem
    most_common_list = counter.most_common(15) #Pega as n palavras mais citadas
    print(most_common_list)
    
    # draw a Word Cloud with word frequencies
    wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(counter)

    plt.figure(figsize=(17,14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return (wordcloud)
    

if __name__ == "__main__":
    # Declaraçao de variaveis
    list_words_pos = []
    list_words_neg = []
    list_words = []
    num_pags = 5
    #csv_name = "dataframe.csv"
    csv_name = "reviews_tripadvisor.csv"
    pos_com = []
    neg_com = []
    
    URL_base = 'http://www.tripadvisor.com.br'
    firstPage_URL= "https://www.tripadvisor.com.br/Airline_Review-d8729003-Reviews-Air-France/"
    before_URL = '/Airline_Review-d8729003-Reviews-or'
    after_URL = '-Air-France#REVIEWS/'
    
    
    # Busca os comentarios como dataframe, nesse caso vamos usar o arquivo csv
    list_comments = get_comments(csv_name) 
    #list_comments = get_comments_selenium(URL_base, firstPage_URL, before_URL, after_URL)
    
    
    # Realizao pre-processamento dos comentarios
    comments = list_comments['Text_select'].apply(lambda x:pre_process(x))
    classes = list_comments['sentiment']
    
    #Separa os comentarios negativos e os positivos a partir do rating
    
    for com in range(len(list_comments)):
        if(list_comments['sentiment'][com]) == 'Positivo':
            pos_com.append(list_comments['Text_select'][com])
        elif(list_comments['sentiment'][com]) == 'Negativo':
            neg_com.append(list_comments['Text_select'][com])
    
    # Carrega as stop-words
    stopwords = get_stop_words("stopwords_PT.txt")
     
 
    
    
#    ############ COMENTARIOS POSITIVOS ###########################
#    ###### separa conjunto de treinamento e testes usando metodo de validaçao cruzada
#    docs_train, docs_test = train_test_split(pos_com, test_size=0.25)
#    cv, tfidf_transformer = treinamento(docs_train, stopwords)
#    list_words_pos = teste(cv, docs_test, tfidf_transformer) 
#    wordcloud = count_methodos (list_words_pos)
#    
    ############ COMENTARIOS POSITIVOS ###########################
    ###### Calcula TF-IDF e faz a contagem das palavras para visualizaçao
    list_words_pos = tf_idf(pos_com, stopwords)
    wordcloud_pos = count_methodos (list_words_pos)
    wordcloud_pos.to_file('wordcloud_pos.png')
    
    
    list_words = []    
#    ##################### COMENTARIOS NEGATIVOS ###########################
#    docs_train, docs_test = train_test_split(neg_com, test_size=0.25)
#    cv, tfidf_transformer = treinamento(docs_train, stopwords)
#    list_words_neg = teste(cv, docs_test, tfidf_transformer) 
#    wordcloud = count_methodos (list_words_neg)

    ############ COMENTARIOS NEGATIVOS ###########################
    ###### Calcula TF-IDF e faz a contagem das palavras para visualizaçao
    list_words_neg = tf_idf(neg_com, stopwords)
    wordcloud_neg = count_methodos (list_words_neg)
    wordcloud_neg.to_file('wordcloud_neg.png')
 