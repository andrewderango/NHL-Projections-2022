import pandas as pd
from sklearn import linear_model
import time

start_time = time.time()

big_board = pd.read_csv('/users/andrewderango/Documents/Python/NHL Projections/PP Shots60 Projections/Forwards/2021-22 PP Shots60 Projector - Forward Board.csv')
column_names = big_board.columns.values.tolist()
print(big_board)

X = big_board[['Model I', 'Model II']]
y = big_board['2021-22 Actual Score']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print()
print('--FORWARDS 3-YEAR REGRESSION--')
print('Model I Coefficient: ' + str(regr.coef_[0]))
print('Model II Coefficient: ' + str(regr.coef_[1]))
print('Results are adjusted for age variance')

end_time = time.time()
elapsed_time = end_time-start_time
print('\nLinear regression of ' + str(len(big_board.columns)) + ' instances performed in ' + str(round(elapsed_time,2)) + ' seconds.')