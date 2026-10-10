from nba_api.stats.endpoints import leaguegamefinder
from datetime import date 
import pandas as pd 
import math
def pull_season_data(season):
    gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable= season,
    season_type_nullable='Regular Season'
)
    games_df = gamefinder.get_data_frames()[0]

# Split into home rows and away rows (you worked this logic out yourself)
    home_df = games_df[games_df['MATCHUP'].str.contains('vs.')]
    away_df = games_df[games_df['MATCHUP'].str.contains('@')]

    merged_df = home_df.merge(away_df, on='GAME_ID', suffixes= ('_home', '_away'))
    print(merged_df.head())
    print(merged_df.shape)
    print(merged_df.columns.tolist())


    final_df = merged_df[['GAME_ID', 'GAME_DATE_home', 'TEAM_ABBREVIATION_home', 'TEAM_ABBREVIATION_away', 'PTS_home', 'PTS_away']].rename(columns= {'GAME_ID':'Game ID', 'GAME_DATE_home':'Date', 'TEAM_ABBREVIATION_home':'Home Team', 'TEAM_ABBREVIATION_away':'Away Team', 'PTS_home':'Home Team Points','PTS_away':'Away Team Points'})
    final_df = final_df.sort_values(by='Date').reset_index(drop=True)
    print(final_df.head())
    print(final_df.shape)
    print(home_df.shape, away_df.shape)
    print(final_df['Date'].is_monotonic_increasing)
    return final_df


def get_past_seasons(n):
    
    today = date.today()
    if today.month >= 10: 
        anchor_year = today.year 
    else: 
        anchor_year = today.year - 1

    season_labels = []
    for i in range (1, n + 1):
        start_year = anchor_year - i
        end_suffix = start_year - 1999
        season_labels.append(f"{start_year}-{end_suffix}")

    all_seasons = []
    for season in season_labels:
        df = pull_season_data(season)
        all_seasons.append(df)

    combined_df = pd.concat(all_seasons)
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)

    return combined_df

pull_df = get_past_seasons(3)
print(pull_df.shape)
print(pull_df['Date'].min(), pull_df['Date'].max())



for season in ['2023-24', '2024-25', '2025-26']:
    df = pull_season_data(season)
    print(season, df.shape)


gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable= '2024-25',
    season_type_nullable='Regular Season'
)
games_df = gamefinder.get_data_frames()[0]

home_df = games_df[games_df['MATCHUP'].str.contains('vs.')]
away_df = games_df[games_df['MATCHUP'].str.contains('@')]

home_ids = set(home_df['GAME_ID'])
away_ids = set(away_df['GAME_ID'])
mystery_ids = away_ids - home_ids
print(len(mystery_ids))
print(games_df[games_df['GAME_ID'].isin(mystery_ids)][['GAME_ID', 'TEAM_ABBREVIATION', 'MATCHUP']])

home_win_rate = (pull_df['Home Team Points'] > pull_df['Away Team Points']).mean()
print(home_win_rate)

START = 1500
K = 20
HOME_BONUS = 32.75
CARRY = 0.75

def win_prob(rating_a, rating_b):
    # chance that A beats B, from the rating gap only
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def run_elo(games, k=K, home_bonus=HOME_BONUS, carry=CARRY):
    games = games.copy()
    games['Season'] = games['Game ID'].str[3:5]   # '23' in 0022300061 = season starting 2023

    teams = set(games['Home Team']) | set(games['Away Team'])
    ratings = {team: START for team in teams}
    current_season = games['Season'].iloc[0]
    predictions = []

    for _, g in games.iterrows():
        # new season: pull every rating part of the way back to 1500
        if g['Season'] != current_season:
            for team in ratings:
                ratings[team] = START + carry * (ratings[team] - START)
            current_season = g['Season']

        home, away = g['Home Team'], g['Away Team']

        # PREDICT: uses ratings only, never the result
        p_home = win_prob(ratings[home] + home_bonus, ratings[away])
        predictions.append(p_home)

        # LEARN: now look at what happened
        home_won = 1 if g['Home Team Points'] > g['Away Team Points'] else 0
        surprise = home_won - p_home
        ratings[home] += k * surprise
        ratings[away] -= k * surprise

    games['P_home'] = predictions
    return ratings, games

ratings, results = run_elo(pull_df)
print(sum(ratings.values()))
print(sorted(ratings.items(), key=lambda x: x[1], reverse=True))

results['Pick_home'] = results['P_home'] > 0.5
results['Home_won'] = results['Home Team Points'] > results['Away Team Points']
results['Correct'] = results['Pick_home'] == results['Home_won']

results['Game_num'] = results.groupby('Season').cumcount()
results['Early'] = results['Game_num'] < 150

print(results.groupby(['Season', 'Early'])['Correct'].mean())
print(results['Correct'].mean())
print(results.groupby('Season')['Correct'].mean())

brier = ((results['P_home'] - results['Home_won'].astype(int)) ** 2).mean()
print(brier)

results['Bucket'] = pd.cut(results['P_home'], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
print(results.groupby('Bucket', observed=True).agg(
    predicted=('P_home', 'mean'),
    actual=('Home_won', 'mean'),
    games=('P_home', 'size')))

def brier(df):
    return ((df['P_home'] - df['Home_won'].astype(int)) ** 2).mean()

# home bonus measured on the tuning seasons only
season_code = pull_df['Game ID'].str[3:5]
train_pull = pull_df[season_code.isin(['23', '24'])]
home_rate = (train_pull['Home Team Points'] > train_pull['Away Team Points']).mean()
bonus = 400 * math.log10(home_rate / (1 - home_rate))

rows = []
for k in [10, 20, 30, 40]:
    for carry in [0.5, 0.75, 0.9]:
        _, res = run_elo(pull_df, k=k, home_bonus=bonus, carry=carry)
        res['Home_won'] = res['Home Team Points'] > res['Away Team Points']
        train_res = res[res['Season'].isin(['23', '24'])]
        rows.append((k, carry, brier(train_res)))

for row in sorted(rows, key=lambda r: r[2]):
    print(row)


best_k, best_carry = 30, 0.5

_, tuned = run_elo(pull_df, k=best_k, home_bonus=bonus, carry=best_carry)
_, default = run_elo(pull_df, k=20, home_bonus=bonus, carry=0.75)

for res in (tuned, default):
    res['Home_won'] = res['Home Team Points'] > res['Away Team Points']

test_tuned = tuned[tuned['Season'] == '25']
test_default = default[default['Season'] == '25']

print('tuned:   ', brier(test_tuned))
print('default: ', brier(test_default))
print('always home rate:', ((home_rate - test_tuned['Home_won'].astype(int)) ** 2).mean())


