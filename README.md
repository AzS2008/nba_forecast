# nba-forecast

A pre-season forecast of the 2026-27 NBA regular season, built from an Elo rating model and a Monte Carlo simulation. Forecasts are committed to this repo **before** the games are played, so they can be checked against what actually happens.

## Forecast 1: 2026-27 (committed 15 Oct 2026, before opening night on 20 Oct)

`forecast_2026-10-15.csv` has, for every team: expected wins, the 10th and 90th percentile of wins, and the chance of a top-6 seed, a play-in spot (seeds 7-10) and missing the playoffs. All figures come from 10,000 simulated seasons. This forecast will not be edited after it is committed.

## How it works

1. **Data.** Game results for 2023-24 to 2025-26 from `nba_api` (3,680 regular-season games). 10 games were dropped because the source data did not mark a home team; the 2025 NBA Paris Games are among them.
2. **Elo ratings.** Every team starts at 1500. After each game a rating moves by `K x (result - expected)`, where the expected result comes from the rating gap plus a home-court bonus of about 33 rating points (measured from the 54.7% home win rate).
3. **Between seasons.** Ratings are pulled halfway back toward 1500 (carry = 0.5).
4. **Tuning and test.** K = 30 and carry = 0.5 were chosen using 2023-24 and 2024-25 only, then tested once on 2025-26: Brier score 0.2078, against 0.2472 for always predicting the home win rate.
5. **Simulation.** The 2026-27 schedule is simulated 10,000 times. At the start of each simulated season every team's rating gets a random nudge (standard deviation 120 rating points) to reflect uncertainty about its true strength. The size was set so the ranges matched the 2025-26 backtest, where the 10th-90th percentile range captured 83% of teams.
6. **Unscheduled games.** Two games per team depend on how the NBA Cup group stage ends and are not on the schedule yet. They are simulated against an average opponent until the schedule is complete.

## What the model does not know

Injuries, trades, the draft, coaching changes or anything else that happened after a team's last game. Last season's results are its only evidence, so teams whose 2025-26 results were distorted by injuries or tanking (for example IND and WAS) are the likeliest to be wrong.

## How this forecast will be scored (decided in advance)

1. Average absolute gap between predicted and actual wins, compared with a know-nothing baseline of league-average wins for every team.
2. Share of teams whose actual wins land inside their 10th-90th percentile range (target: about 80%).
3. Calibration of the top-6, play-in and miss-the-playoffs probabilities.

Results will be published at the end of the season, whether good or bad.

## Files

- `nba_elo.py`: data pipeline, Elo model, tuning, backtest, simulation and forecast.
- `games_last3.csv`: the game data used.
- `forecast_2026-10-15.csv`: the committed forecast.
