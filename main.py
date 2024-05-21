import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
pandas.options.display.float_format = '{:,.2f}'.format

file_path_t_haliburton_regszn = 'data/t_haliburton_23-24_regszn.csv'
file_path_t_haliburton_playoffs = 'data/t_haliburton_23-24_playoffs.csv'

#set1 = pandas.read_csv(file_path_t_haliburton_regszn, parse_dates=['Date'], dtype={'G': str})
#set2 = pandas.read_csv(file_path_t_haliburton_playoffs, parse_dates=['Date'], dtype={'G': str})

def convert_mp_to_minutes(mp_str):
    try:
        if isinstance(mp_str, pandas.Timestamp):
            time_str = mp_str.strftime('%H:%M:%S')
        else:
            time_str = mp_str

        # Split the time string into hours, minutes, and seconds
        time_parts = time_str.split(':')
        minutes = int(time_parts[0])
        seconds = int(time_parts[1])
        # Calculate total minutes
        total_minutes = minutes + seconds/60
        return total_minutes
    except Exception as e:
        print(f"Error converting {mp_str}: {e}")
        return 0
    
def preprocess(file_path):
    
    df = pandas.read_csv(file_path, parse_dates=['Date'], dtype={'MP': str})

    df = df.dropna(subset=['G', 'MP'])

    df['MP'] = df['MP'].apply(convert_mp_to_minutes)

    df.fillna(0, inplace=True)

    avg_pts = df['PTS'].mean()

    df['above_average'] = df['PTS'] > df['Line']
    df['rebounds_assists_ratio'] = df['TRB'] / df['AST']
    df['pts_reb+ast_ratio'] = df['PTS'] / (df['TRB'] + df['AST'])
    df.replace([float('inf'), -float('inf')], 0, inplace=True)

    return df

set1 = preprocess(file_path_t_haliburton_regszn)
#set1 = preprocess(file_path_t_haliburton_playoffs)

target = set1.iloc[:1]
set1 = set1.iloc[1:]

avg_pts = set1['PTS'].mean()
print(set1)
print(avg_pts)

# Histogram of points scored
plt.hist(set1['PTS'], bins=20, edgecolor='black')
plt.xlabel('Points Scored')
plt.ylabel('Frequency')
plt.title('Distribution of Points Scored')
plt.show()

# Scatter plot of points vs. minutes played
plt.scatter(set1['MP'], set1['PTS'])
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.title('Points vs. Minutes Played')
plt.show()

# Scatter plot of points vs. rebounds_assists_ratio
plt.scatter(set1['rebounds_assists_ratio'], set1['PTS'])
plt.xlabel('Rebounds to Assists Ratio')
plt.ylabel('Points Scored')
plt.title('Points vs. Rebounds to Assists Ratio')
plt.show()

# model
X = set1[['MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'PTS']]
y = set1['above_average'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

new_data = pandas.DataFrame({
    'MP': [35],
    'FG%': [0.45],
    '3P%': [0.38],
    'FT%': [0.85],
    'TRB': [4.5],
    'AST': [7.5],
    'STL': [1.5],
    'BLK': [0.5],
    'TOV': [2],
    'rebounds_assists_ratio': [5/7],
    'PTS': [17.5]
})

#first_row = set2.head(1).copy()
#print(set2.head(1))
first_row_prediction = model.predict(new_data[X.columns])
print(first_row_prediction)
print(f'Prediction: {"Above Line" if first_row_prediction[0] else "Below Line"}')
print(new_data)