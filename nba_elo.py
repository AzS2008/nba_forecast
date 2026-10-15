"""
nba-forecast: data pipeline, Elo ratings, Monte Carlo season simulation.

Order of this file:
  1. settings
  2. functions (nothing runs here, they are just definitions)
  3. the run section at the bottom (only runs when you run this file directly)
"""
import math
from datetime import date

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, scheduleleaguev2

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
START = 1500          # every team's starting rating
K = 30                # how far one game moves a rating (tuned on 2023-25)
CARRY = 0.5           # share of last season's rating kept after the summer (tuned)
STRENGTH_SD = 120     # preseason uncertainty about team strength, in rating points
#                       (100 gave a spread of 10.7 wins in the 2025-26 backtest vs
#                       a real miss spread of 12.6, so it was nudged up to 120)
N_SIMS = 10000        # number of imagined seasons

# Used to work out playoff seed.
EAST = ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DET', 'IND',
        'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
WEST = ['DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN',
        'NOP', 'OKC', 'PHX', 'POR', 'SAC', 'SAS', 'UTA']


TEAM_FIX = {}


def pull_season_data(season):
    """One clean row per regular-season game, for a season like '2025-26'."""
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    games_df = gamefinder.get_data_frames()[0]

    # the API gives two rows per game: 'vs.' marks the home row, '@' the away row
    home_df = games_df[games_df['MATCHUP'].str.contains('vs.')]
    away_df = games_df[games_df['MATCHUP'].str.contains('@')]

    # glue the two rows of each game into one wide row, matched on GAME_ID
    merged_df = home_df.merge(away_df, on='GAME_ID', suffixes=('_home', '_away'))

    final_df = merged_df[['GAME_ID', 'GAME_DATE_home', 'TEAM_ABBREVIATION_home',
                          'TEAM_ABBREVIATION_away', 'PTS_home', 'PTS_away']].rename(
        columns={'GAME_ID': 'Game ID', 'GAME_DATE_home': 'Date',
                 'TEAM_ABBREVIATION_home': 'Home Team',
                 'TEAM_ABBREVIATION_away': 'Away Team',
                 'PTS_home': 'Home Team Points',
                 'PTS_away': 'Away Team Points'})
    return final_df.sort_values(by='Date').reset_index(drop=True)


def get_past_seasons(n):
    """The last n completed seasons, stacked into one table in date order."""
    today = date.today()
    # from October on, this calendar year's season is the current one
    anchor_year = today.year if today.month >= 10 else today.year - 1

    season_labels = []
    for i in range(1, n + 1):
        start_year = anchor_year - i
        season_labels.append(f"{start_year}-{(start_year + 1) % 100:02d}")

    all_seasons = [pull_season_data(season) for season in season_labels]
    combined_df = pd.concat(all_seasons)
    return combined_df.sort_values(by='Date').reset_index(drop=True)


def win_prob(rating_a, rating_b):
    """Chance that A beats B, from the rating gap only."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def home_bonus_from(df):
    """Home-court bonus in rating points, from the home win rate in df."""
    rate = (df['Home Team Points'] > df['Away Team Points']).mean()
    return 400 * math.log10(rate / (1 - rate))


def run_elo(games, k, home_bonus, carry, start=START):
    """Replay games in date order. Returns final ratings and the games table
    with the model's pre-game home win probability (P_home)."""
    games = games.copy()
    games['Season'] = games['Game ID'].str[3:5]   # '23' in 0022300061

    teams = set(games['Home Team']) | set(games['Away Team'])
    ratings = {team: start for team in teams}
    current_season = games['Season'].iloc[0]
    predictions = []

    for _, g in games.iterrows():
        # new season: pull every rating part of the way back to the average
        if g['Season'] != current_season:
            for team in ratings:
                ratings[team] = start + carry * (ratings[team] - start)
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
    games['Home_won'] = games['Home Team Points'] > games['Away Team Points']
    return ratings, games


def brier(df):
    """Average squared miss of the probabilities. Lower is better."""
    return ((df['P_home'] - df['Home_won'].astype(int)) ** 2).mean()


def scorecard(results):
    """Hit rate, Brier score and calibration of a run_elo results table."""
    results = results.copy()
    results['Correct'] = (results['P_home'] > 0.5) == results['Home_won']
    results['Early'] = results.groupby('Season').cumcount() < 150

    print('hit rate:', round(results['Correct'].mean(), 4))
    print(results.groupby('Season')['Correct'].mean().round(4))
    print('first ~10 games per team vs the rest:')
    print(results.groupby(['Season', 'Early'])['Correct'].mean().round(4))
    print('brier:', round(brier(results), 4))

    results['Bucket'] = pd.cut(results['P_home'], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    print(results.groupby('Bucket', observed=True).agg(
        predicted=('P_home', 'mean'),
        actual=('Home_won', 'mean'),
        games=('P_home', 'size')).round(3))


def preseason_ratings(end_ratings, carry, start=START):
    """Ratings at the start of a new season: shrink toward the average."""
    return {t: start + carry * (r - start) for t, r in end_ratings.items()}


def simulate_season(schedule, start_ratings, home_bonus, n_sims=N_SIMS,
                    strength_sd=0, neutral=None, extra_games=None):
    """Simulate a season n_sims times. Returns a table: one row per imagined
    season, one column per team, each cell = that team's wins.

    strength_sd: each imagined season starts every team with a random nudge
    to its rating (normal, this many rating points), so the simulation admits
    it is unsure how good each team really is.
    neutral: 1 for neutral-site games (no home bonus), else 0.
    extra_games: {team: number of games not on the schedule yet}. These are
    simulated against an average (1500) opponent with no home bonus."""
    teams = sorted(start_ratings)
    idx = {t: i for i, t in enumerate(teams)}
    n_games = len(schedule)

    base = np.array([start_ratings[t] for t in teams])
    ratings = base + np.random.normal(0, strength_sd, size=(n_sims, len(teams)))

    h = schedule['Home Team'].map(idx).to_numpy()
    a = schedule['Away Team'].map(idx).to_numpy()
    neutral = np.zeros(n_games) if neutral is None else np.asarray(neutral, dtype=float)

    # win probability of the home team in every game of every imagined season
    gap = ratings[:, h] + home_bonus * (1 - neutral) - ratings[:, a]
    p_home = 1 / (1 + 10 ** (-gap / 400))

    # tables that say which team hosted / travelled in each game
    H = np.zeros((n_games, len(teams)), dtype=np.float32)
    A = np.zeros((n_games, len(teams)), dtype=np.float32)
    H[np.arange(n_games), h] = 1
    A[np.arange(n_games), a] = 1

    # random-number idea, for every game at once: True = home team wins
    hw = (np.random.rand(n_sims, n_games) < p_home).astype(np.float32)
    wins = hw @ H + (1 - hw) @ A

    if extra_games:
        extra = np.array([extra_games.get(t, 0) for t in teams])
        p_extra = 1 / (1 + 10 ** ((START - ratings) / 400))   # vs an average team
        wins = wins + np.random.binomial(extra, p_extra)

    return pd.DataFrame(wins, columns=teams)


def seed_odds(sims, conference):
    """For each team in a conference: chance of a top-6 seed, a play-in spot
    (7-10) and missing the playoffs (11-15). Ties are broken at random."""
    wins = sims[conference]
    noise = np.random.rand(*wins.shape) * 0.001
    ranks = (-(wins + noise)).rank(axis=1, method='first')
    return pd.DataFrame({
        'top6_%': (ranks <= 6).mean() * 100,
        'playin_%': ((ranks >= 7) & (ranks <= 10)).mean() * 100,
        'miss_%': (ranks >= 11).mean() * 100,
    })


if __name__ == '__main__':

    # A. data: last 3 completed seasons 
    pull_df = get_past_seasons(3)
    pull_df.to_csv('games_last3.csv', index=False)   # saved copy of the data
    print('games:', pull_df.shape, pull_df['Date'].min(), pull_df['Date'].max())

    # B. tune on 2023-24 and 2024-25 only 
    season_code = pull_df['Game ID'].str[3:5]
    train_pull = pull_df[season_code.isin(['23', '24'])]
    bonus_train = home_bonus_from(train_pull)          # home bonus from tuning seasons only

    rows = []
    for k in [10, 20, 30, 40]:
        for carry in [0.5, 0.75, 0.9]:
            _, res = run_elo(pull_df, k, bonus_train, carry)
            rows.append((k, carry, brier(res[res['Season'].isin(['23', '24'])])))
    print('tuning (k, carry, brier on 2023-25):')
    for row in sorted(rows, key=lambda r: r[2]):
        print(row)

    #  C. held-out test on 2025-26 
    _, tuned = run_elo(pull_df, K, bonus_train, CARRY)
    _, default = run_elo(pull_df, 20, bonus_train, 0.75)
    home_rate_train = (train_pull['Home Team Points'] > train_pull['Away Team Points']).mean()
    test_tuned = tuned[tuned['Season'] == '25']
    test_default = default[default['Season'] == '25']
    print('held-out 2025-26 brier -> tuned:', round(brier(test_tuned), 4),
          '| default:', round(brier(test_default), 4),
          '| always home rate:',
          round(((home_rate_train - test_tuned['Home_won'].astype(int)) ** 2).mean(), 4))
    scorecard(tuned)

    #  D. backtest the simulation on 2025-26 
    sched25 = pull_df[season_code == '25']
    early = pull_df[season_code.isin(['23', '24'])]
    r_end, _ = run_elo(early, K, bonus_train, CARRY)
    start25 = preseason_ratings(r_end, CARRY)

    home_w = sched25['Home Team Points'] > sched25['Away Team Points']
    actual = sched25.loc[home_w, 'Home Team'].value_counts().add(
        sched25.loc[~home_w, 'Away Team'].value_counts(), fill_value=0)

    print('backtest: nudge | avg abs miss | spread of a team across sims | share of teams inside 10-90% range')
    for sd in [0, 100, 120, 140]:
        sims = simulate_season(sched25, start25, bonus_train, strength_sd=sd)
        lo, hi = sims.quantile(0.1), sims.quantile(0.9)
        inside = ((actual >= lo) & (actual <= hi)).mean()
        miss = (actual - sims.mean()).abs().mean()
        print(sd, round(miss, 1), round(sims.std().mean(), 1), round(inside, 2))
    print('know-nothing baseline miss:', round((actual - actual.mean()).abs().mean(), 1))

    #  E. fetch and check the 2026-27 schedule 
    sched_raw = scheduleleaguev2.ScheduleLeagueV2(season='2026-27')
    sched_df = sched_raw.get_data_frames()[0]

    # '002...' = regular season, '001...' = preseason, '006...' = Cup final
    sched26 = sched_df[sched_df['gameId'].str.startswith('002')][
        ['gameId', 'gameDate', 'homeTeam_teamTricode',
         'awayTeam_teamTricode', 'isNeutral']].copy()
    sched26.columns = ['Game ID', 'Date', 'Home Team', 'Away Team', 'Neutral']
    sched26['Home Team'] = sched26['Home Team'].replace(TEAM_FIX)
    sched26['Away Team'] = sched26['Away Team'].replace(TEAM_FIX)
    sched26['Date'] = pd.to_datetime(sched26['Date'], errors='coerce')

    # Cup quarter/semi-final games are listed with the teams still to be decided
    # (blank team names). They cannot be simulated yet, so they are dropped here.
    n_tbd = int(sched26[['Home Team', 'Away Team']].isna().any(axis=1).sum())
    sched26 = sched26.dropna(subset=['Home Team', 'Away Team']).reset_index(drop=True)

    print('rows with teams still to be decided (dropped):', n_tbd)
    print('2026-27 regular-season games with known teams:', len(sched26))
    print('first / last game:', sched26['Date'].min(), sched26['Date'].max())
    print('duplicate game IDs:', sched26['Game ID'].duplicated().sum())
    print('neutral-site games:', int(sched26['Neutral'].astype(bool).sum()))
    games_per_team = pd.concat([sched26['Home Team'], sched26['Away Team']]).value_counts()
    # games not scheduled yet: they depend on how the NBA Cup group stage ends
    extra_games = (82 - games_per_team).clip(lower=0).to_dict()
    print('games still to be scheduled per team:', sorted(set(extra_games.values())))
    print('total unscheduled team-games:', sum(extra_games.values()), '(= 2 per team if all 30 teams show 2)')

    #  F. final ratings for the live forecast (all 3 seasons) 
    bonus_all = home_bonus_from(pull_df)
    ratings_end, _ = run_elo(pull_df, K, bonus_all, CARRY)
    print('sum of ratings (should be 45000):', round(sum(ratings_end.values()), 1))
    print('top 5 / bottom 5 ratings:')
    ranked = sorted(ratings_end.items(), key=lambda x: x[1], reverse=True)
    print(ranked[:5])
    print(ranked[-5:])

    schedule_teams = set(sched26['Home Team']) | set(sched26['Away Team'])
    mismatch = schedule_teams ^ set(ratings_end)
    print('team abbreviations that do not match between schedule and ratings:', mismatch)

    #  G. the 2026-27 forecast
    if mismatch:
        print('FORECAST SKIPPED: fix the team abbreviations first (see TEAM_FIX).')
    else:
        if set(extra_games.values()) - {0, 2}:
            print('WARNING: some teams are missing a number of games other than 2.',
                  'Check games_per_team before publishing.')
        start26 = preseason_ratings(ratings_end, CARRY)
        sims26 = simulate_season(sched26, start26, bonus_all, strength_sd=STRENGTH_SD,
                                 neutral=sched26['Neutral'].astype(bool),
                                 extra_games=extra_games)

        forecast = pd.DataFrame({
            'exp_wins': sims26.mean(),
            'wins_10th_pct': sims26.quantile(0.1),
            'wins_90th_pct': sims26.quantile(0.9),
        })
        odds = pd.concat([seed_odds(sims26, EAST), seed_odds(sims26, WEST)])
        forecast = forecast.join(odds).sort_values('exp_wins', ascending=False).round(1)
        print(forecast)

        out = f'forecast_{date.today()}.csv'
        forecast.to_csv(out)
        print('saved', out)