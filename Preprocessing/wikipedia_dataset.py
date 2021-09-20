import nltk as nltk
import numpy as np
import pandas as pd
import os.path

def getDataLabelledSentences():
    # download nltk punkt package (to split text in sentences)
    nltk.download('punkt')

    # Get text of all articles
    articles_dataframe = pd.read_csv("Datasets/Wikipedia/1_Components_Detection/articles.txt", sep="	")

    # Get all claims
    claims_dataframe = pd.read_csv("Datasets/Wikipedia/1_Components_Detection/claims.txt", sep="	")

    print("Number of articles {}".format(len(articles_dataframe.Title)))
    print("Number of claims {}".format(len(claims_dataframe.get("Claim original text"))))

    number_located_claims = 0
    number_article_not_found = 0
    wiki_article = []

    directory = os.fsencode("Datasets/Wikipedia/articles")
    article_no_claims = []
    number_of_claims = 0
    X = []
    Y = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #print(filename)
        if filename.endswith(".txt") or filename.endswith(".py"):
            with open(os.path.join(directory, file), 'r') as txt_file:
                txt = txt_file.read().replace('\n', '')

                #print(txt)
                sentences = nltk.tokenize.sent_tokenize(txt)

                # Get topic from article id
                art_id = int(filename[6:-4])
                topic = articles_dataframe.loc[articles_dataframe['article Id'] == art_id, 'Topic']
                #print(topic)
                if len(topic) > 0:
                    #print(topic.item())
                    # Get all claim from this topic
                    claims_similar_topic = claims_dataframe.loc[claims_dataframe['Topic'] == topic.item()]

                    list_claims_ori = claims_similar_topic['Claim original text'].tolist()
                    list_claims_cor = claims_similar_topic['Claim corrected version'].tolist()

                    for sentence in sentences:
                        X.append(sentence)
                        if any(s in sentence for s in list_claims_ori) or any(s in sentence for s in list_claims_cor):
                            # Sentence include a claim
                            Y.append(1)
                            #print("claim")
                            number_of_claims = number_of_claims + 1
                            #print(sentence)
                        else:
                            # Sentence not include claim
                            Y.append(0)
                            # print("not claim")
                            # print(sentence)

                else:
                    article_no_claims.append(art_id)
                    continue
        else:
            continue

    return X,Y

    print("number_article_not_found")
    print(number_article_not_found)

    print("Number of claims {}".format(len(claims_dataframe.get("Claim original text"))))
    print(number_of_claims)
    print("Number of claims found in the articles: {}".format(number_located_claims))





    # for index, row in claims_dataframe.iterrows():
    #    print("Claim")
    #    print(row['Claim original text'])
    #    print(index)

    #    article = articles_dataframe.loc[articles_dataframe['article Id'] == index]

    #    print("Article title")
    #    print(article)
    #    article_title = article['Title']
    #    print(article_title)
