# Script le prix de cloture de l'indice S&P500

import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Lire les données et convertir la colonne 'Date' en objet datetime
df = pd.read_csv("sphist.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Trier par date de la plus ancienne à la plus récente
df = df.sort_values("Date", ascending=True)

# Ajouter des caractéristiques utiles à notre algorithme de Machine Learning
df['5 Days Open'] = df['Open'].rolling(center=False, window=5).mean()
df['5 Days High'] = df['High'].rolling(center=False, window=5).mean()
df['5 Days Low'] = df['Low'].rolling(center=False, window=5).mean()
df['5 Days Volume'] = df['Volume'].rolling(center=False, window=5).mean()
df['Year'] = df['Date'].apply(lambda x: x.year)
# Ajouter les jours de la semaine et les fixer en colonnes de type catégorie
df['DOW'] = df['Date'].apply(lambda x: x.weekday())
dow_df = pd.get_dummies(df['DOW'])
df = pd.concat([df, dow_df], axis=1)
df = df.drop(['DOW'], axis=1)

# Déplacer la colonne d'un jour
df['5 Days Open'] = df['5 Days Open'].shift(1)
df['5 Days High'] = df['5 Days High'].shift(1)
df['5 Days Low'] = df['5 Days Low'].shift(1)
df['5 Days Volume'] = df['5 Days Volume'].shift(1)

df = df[df['Date'] >= datetime(year=1951, month=1, day=3)]
df.dropna(axis=0)

train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

features = ['5 Days Open', '5 Days Volume', '5 Days High', '5 Days Low', 'Year', 0, 1, 2, 3, 4]

lr = LinearRegression()
lr.fit(train[features], train['Close'])
predictions = lr.predict(test[features])

mae = mean_absolute_error(test['Close'] ,predictions)

print(df.tail(1))
print(mae)