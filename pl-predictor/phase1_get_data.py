import requests
import pandas as pd
import time  # Fix 1: was missing

API_KEY = 'YOUR_API_KEY_HERE'  # Get a free key at football-data.org
BASE_URL = 'https://api.football-data.org/v4/competitions/PL/matches'

headers = {'X-Auth-Token': API_KEY}

def fetch_premier_league_data():
    seasons = [2022, 2023, 2024, 2025]
    match_list = []

    for year in seasons:
        print(f"Fetching season {year}...")
        url = f"{BASE_URL}?season={year}"
        response = requests.get(url, headers=headers)  # Fix 2: removed duplicate line

        if response.status_code == 200:
            data = response.json()
            matches = data['matches']

            for match in matches:  # Fix 3: now correctly inside the if block
                match_info = {
                    'season': year,  # Fix 4: added missing comma
                    'date': match['utcDate'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                    'status': match['status']
                }
                match_list.append(match_info)
        else:
            print(f"Error fetching {year}: Status Code {response.status_code}")

        time.sleep(6)  # Fix 5: inside season loop, after each API call

    if match_list:
        df = pd.DataFrame(match_list)
        df.to_csv('pl_matches_raw.csv', index=False)
        print(f"Success! Saved {len(df)} matches to pl_matches_raw.csv")
    else:
        print("No matches were collected.")

if __name__ == "__main__":
    fetch_premier_league_data()
