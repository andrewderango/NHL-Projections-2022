import pandas as pd
from sklearn import linear_model
import time

start_time = time.time()

big_board = pd.read_csv('/users/andrewderango/Documents/Python/NHL Projections/EV MetaGoals60 Projections/Age Unadjusted/EV MetaGoals60 Projector - Forward Board.csv')
column_names = big_board.columns.values.tolist()
# print(big_board)
streaks = pd.DataFrame(columns=['Player','Range','Year I','Year II','Year III','Year IV'])
for index in range(len(big_board.index)):
    player_timeline = []

    for column in big_board:
        player_timeline.append(big_board[column][index])
    # print(player_timeline)
    
    for item in range(len(player_timeline)):
        try:
            if player_timeline[item] == float(player_timeline[item]) and player_timeline[item+1] == float(player_timeline[item+1]) and player_timeline[item+2] == float(player_timeline[item+2]) and player_timeline[item+3] == float(player_timeline[item+3]):
                # print(player_timeline[item], player_timeline[item+1], player_timeline[item+2], player_timeline[item+3])

                streak_dict = {'Player': player_timeline[0], 'Range': column_names[item][0:4] + column_names[item+3][4:7], 'Year I': player_timeline[item], 'Year II': player_timeline[item+1], 'Year III': player_timeline[item+2], 'Year IV': player_timeline[item+3]}
                streaks = streaks.append(streak_dict, ignore_index = True)
        except:
            pass

    # print()

# print(streaks)

X = streaks[['Year I', 'Year II', 'Year III']]
y = streaks['Year IV']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print()
print('--FORWARDS 3-YEAR REGRESSION--')
print('Year I Coefficient: ' + str(regr.coef_[0]))
print('Year II Coefficient: ' + str(regr.coef_[1]))
print('Year III Coefficient: ' + str(regr.coef_[2]))
print('NOTE: Results are unadjusted for age variance.')

end_time = time.time()
elapsed_time = end_time-start_time
print('\nLinear regression of ' + str(len(streaks.index)) + ' instances performed in ' + str(round(elapsed_time,2)) + ' seconds.')