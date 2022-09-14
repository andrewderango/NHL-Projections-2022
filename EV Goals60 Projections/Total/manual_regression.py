import pandas as pd

df = pd.read_csv('/users/andrewderango/Documents/Python/NHL Projections/EV Shots60 Projections/Total/2021-22 EV Shots60 Projector - Filtered Board.csv')
print('Model I Correlation:  ' + str(df['Model I'].corr(df['2021-22 Actual Score'])))
print('Model II Correlation: ' + str(df['Model II'].corr(df['2021-22 Actual Score'])))

result_df = pd.DataFrame({'Coefficient 1': [], 'Coefficient 2': [], 'Correlation Value': []})

coefficient_1 = 1
coefficient_2 = 0

for x in range(102):
    comb_model = df['Model I']*coefficient_1 + df['Model II']*coefficient_2
    result_df = result_df.append({'Coefficient 1': coefficient_1, 'Coefficient 2': coefficient_2, 'Correlation Value': comb_model.corr(df['2021-22 Actual Score'])}, ignore_index=True)
    coefficient_1 -= 0.01
    coefficient_2 += 0.01

print(result_df.to_string())