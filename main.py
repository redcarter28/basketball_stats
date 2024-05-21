import pandas

file_path = 'data/sportsref_download.xlsx'

stats = pandas.read_excel(file_path, parse_dates=['Date'])
#stats['MP'] = stats['MP'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
stats.fillna(0, inplace=True)

print(stats)