import pandas as pd

df = pd.read_csv("pl_matches_raw.csv")

def clean_and_feature_engineer(df):
    df = df[df['status'] == 'FINISHED'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    def determine_result(row):
        if row['home_score'] > row['away_score']:
            return 0
        elif row['home_score'] == row['away_score']:
            return 1
        else:
            return 2

    df['target'] = df.apply(determine_result, axis=1)

    df['home_code'] = df['home_team'].astype('category').cat.codes
    df['away_code'] = df['away_team'].astype('category').cat.codes

    # Points earned per game (used for rolling form)
    df['h_pts'] = df.apply(
        lambda x: 3 if x['target'] == 0 else (1 if x['target'] == 1 else 0), axis=1
    )
    df['a_pts'] = df.apply(
        lambda x: 3 if x['target'] == 2 else (1 if x['target'] == 1 else 0), axis=1
    )

    # Rolling avg points and goals over last 5 games (no data leakage — only past games)
    def get_rolling_stats(team_name, current_date):
        past_games = df[
            ((df['home_team'] == team_name) | (df['away_team'] == team_name)) &
            (df['date'] < current_date)
        ].tail(5)

        if len(past_games) < 5:
            return 0, 0

        pts = 0
        goals = 0
        for _, game in past_games.iterrows():
            if game['home_team'] == team_name:
                pts += game['h_pts']
                goals += game['home_score']
            else:
                pts += game['a_pts']
                goals += game['away_score']

        return pts / 5, goals / 5

    print("Calculating rolling form (this takes 1-2 minutes)...")

    home_stats = df.apply(
        lambda x: get_rolling_stats(x['home_team'], x['date']), axis=1
    )
    df['home_form'], df['home_avg_goals'] = zip(*home_stats)

    away_stats = df.apply(
        lambda x: get_rolling_stats(x['away_team'], x['date']), axis=1
    )
    df['away_form'], df['away_avg_goals'] = zip(*away_stats)

    # Days rest since last match
    def get_rest_days(team_name, current_date):
        past_games = df[
            ((df['home_team'] == team_name) | (df['away_team'] == team_name)) &
            (df['date'] < current_date)
        ]
        if past_games.empty:
            return 7
        return (current_date - past_games['date'].max()).days

    df['home_rest'] = df.apply(
        lambda x: get_rest_days(x['home_team'], x['date']), axis=1
    )
    df['away_rest'] = df.apply(
        lambda x: get_rest_days(x['away_team'], x['date']), axis=1
    )

    # Drop early-season rows with no form history yet
    df = df[df['home_form'] > 0]

    df.to_csv("pl_matches_processed.csv", index=False)
    print("Phase 2 Complete! Saved 'pl_matches_processed.csv'")
    return df

if __name__ == "__main__":
    processed_df = clean_and_feature_engineer(df)
    print(processed_df.head())
