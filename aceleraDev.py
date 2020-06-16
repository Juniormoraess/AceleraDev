import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import csv

base = pd.read_csv('train.csv')
base = base.fillna(0)
base = base.drop(['Unnamed: 0'], axis = 1)

X = base.iloc[:, 96:99].values
y = base.iloc[:, 99].values

##correlacao = np.corrcoef(X, y)
##X = X.reshape(-1, 1)


modelo = LinearRegression()
modelo.fit(X, y)
modelo.intercept_
modelo.coef_

modelo.score(X, y)
modelo_ajustado = sm.ols(formula = 'NU_NOTA_MT ~ NU_NOTA_CN + NU_NOTA_CH + NU_NOTA_LC', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

base2 = pd.read_csv('test.csv')
base2 = base2.fillna(0)
##base2 = base2.drop(['Unnamed: 0'], axis = 1)

X1 = base2.iloc[:, 28:31].values

notasMT = []

for i in range(0, len(X1)):
    previsao = np.array(X1[i])
    previsao = previsao.reshape(1, -1)
    notasMT.append(modelo.predict(previsao))
        

for i in range(0, len(X1)):
    if notasMT[i] < 0:
        notasMT[i] = 0

notas = np.asarray(notasMT, dtype = float)

for i in range(0, len(X1)):
    if notas[i] < 0:
        notas[i] = 0

numInscricao = base2.iloc[:, 0].values
numInscricao = np.array(numInscricao, dtype = object)

##df = {'NU_INSCRICAO': numInscricao,
      ##'NU_NOTA_MT': notasMT}


##answer = pd.DataFrame(df)
##csv = answer.to_csv(index=False)

with open('answer.csv', 'w') as arquivo_csv:
    colunas = ['NU_INSCRICAO', 'NU_NOTA_MT']
    escrever = csv.DictWriter(arquivo_csv, fieldnames = colunas, 
                              delimiter = ';', lineterminator='\n')
    
    escrever.writeheader()
    for i in range(0, len(X1)):
        escrever.writerow({'NU_INSCRICAO': numInscricao[i],
                           'NU_NOTA_MT': notas[i]})