import pandas as pd
from  nltk.tokenize import sent_tokenize

data = pd.read_excel('fonts5.xlsx')
#print(data.index)
#print (data.loc['arial',:])
split_data = pd.DataFrame(['starts'],index=['font'],columns=['mission and vision'])
for index in data.index:
    list = sent_tokenize(data.ix[index,'Mission_statement'])

    for sentence in list:
        df = pd.DataFrame([sentence],index=[data.ix[index,0]],columns=['mission and vision'])
        split_data = pd.concat([split_data,df],axis=0)
print (split_data)

split_data.to_csv('split_mission_vision_statement.csv')
