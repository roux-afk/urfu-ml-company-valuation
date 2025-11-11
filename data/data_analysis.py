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

print("Первые 5 строк тренировочных данных")
print(train_csv.head())

print("\nИнформация о данных")
print(train_csv.info())

print("\nОсновные статистики:")
print(train_csv.describe())