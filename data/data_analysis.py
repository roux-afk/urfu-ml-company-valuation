from statistics import correlation

import pandas as pd                                     # для работы с таблицами (как Excel в коде)
import numpy as np                                      # для математических операций
import matplotlib.pyplot as plt                         # для графиков
import seaborn as sns                                   # для красивых графиков
from sklearn.model_selection import train_test_split    # для разделения данных
from sklearn.preprocessing import StandardScaler        # для нормализации данных
from sklearn.linear_model import LinearRegression       # простая модель
from sklearn.ensemble import GradientBoostingRegressor  # сложная модель
from sklearn.metrics import mean_squared_error, mean_absolute_error  # для оценки качества

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

print("\nИнформация о данных:")
print(train_csv.info())

print("\nОсновные статистики:")
print(train_csv.describe())

# Разделяем на признаки (X) и целевую переменную (y)
# Признаки - это то, на основе чего мы предсказываем
# Целевая переменная - то, что мы предсказываем (стоимость)

x = train_csv.drop("cost", axis=1)          # все колонки КРОМЕ cost
y = train_csv["cost"]                             # только колонка cost

# Проверяем пропущенные значения
print("\nПропущенные значения")
print(x.isnull().sum())                           # посчитает пропуски в каждой колонке

# 1. Смотрим на распределение стоимости
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, alpha=0.6, color="blue", edgecolor="gray", linewidth=1)
plt.title("Распределение стоимости компаний")
plt.xlabel("Стоимость")
plt.ylabel("Количество")
plt.show()

# 2. Смотрим корреляции (взаимосвязи между переменными)
correlation_matrix = train_csv.drop('id', axis=1).corr()    # Считаем корреляции кроме 'id'
plt.figure(figsize=(18, 14))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, linecolor="white", annot_kws={"size": 8})
plt.title("Матрица корреляций")
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.show()

# 3. Смотрим на самые важные признаки для стоимости
correlation_with_cost = correlation_matrix["cost"].sort_values(ascending=False)
print("\nКорреляция признаков со стоимостью:")
print(correlation_with_cost)