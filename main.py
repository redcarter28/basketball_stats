import pandas
 
file_path = 'data/t_haliburton.csv'

stats = pandas.read_csv(file_path, parse_dates=['Date'])

stats.fillna(0, inplace=True)

avg_pts = stats['PTS'].mean()

stats['above_average'] = stats['PTS'] > avg_pts
stats['rebounds_assists_ratio'] = stats['TRB'] / stats['AST']
stats['pts_reb+ast_ratio'] = stats['PTS'] / (stats['TRB'] + stats['AST'])
stats.replace([float('inf'), -float('inf')], 0, inplace=True)

print(stats)
print(avg_pts)




