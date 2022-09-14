import pandas as pd
from sklearn import linear_model
import time

start_time = time.time()

#Must adjust after duplication!
models_tested_qty = 3

big_board = pd.read_csv('/users/andrewderango/Documents/Python/NHL Projections/PP Goals60 Projections/Defence/2021-22 PP Goals60 Projector - Defence Board.csv')
column_names = big_board.columns.values.tolist()
possible_models = ['Model I', 'Model II', 'Model III', 'Model IV', 'Model V', 'Model VI', 'Model VII', 'Model VIII', 'Model IX', 'Model X']
models_tested = possible_models[0:models_tested_qty]

X = big_board[models_tested]
y = big_board['2021-22 Actual Score']
regr = linear_model.LinearRegression()
regr.fit(X, y)

coefficient_sum = 0
print()
print('--DEFENCE 3-YEAR REGRESSION--')
for model in range(len(X.columns)):
    print(possible_models[model] + ' Coefficient: ' + str(regr.coef_[model]))
    coefficient_sum += regr.coef_[model]
print('\nSum of coefficients: ' + str(round(coefficient_sum,2)))

positive_coefficient_models = []
for model in range(len(X.columns)):
    if regr.coef_[model] > 0:
        positive_coefficient_models.append(possible_models[model])
    else:
        print('Removing ' + possible_models[model] + ' from regression analysis due to poor partial correlation...')

X = big_board[positive_coefficient_models]
y = big_board['2021-22 Actual Score']
regr = linear_model.LinearRegression()
regr.fit(X, y)

coefficient_sum = 0
positive_coefficient_model_counter = 0
print('\n--SIMPSON RE-EVALTION--')
for model in range(len(models_tested)):
    if models_tested[model] in X.columns:
        print(models_tested[model] + ' Coefficient: ' + str(regr.coef_[positive_coefficient_model_counter]))
        coefficient_sum += regr.coef_[positive_coefficient_model_counter]
        positive_coefficient_model_counter += 1
    else:
        print(models_tested[model] + ' Coefficient: 0')

print('\nSum of coefficients: ' + str(coefficient_sum))

end_time = time.time()
elapsed_time = end_time-start_time
print('\nMultiple linear regression of ' + str(len(big_board)) + ' instances performed in ' + str(round(elapsed_time,2)) + ' seconds.')