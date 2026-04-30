import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("pl_matches_processed.csv")

def train_predictor(df):
    features = [
        "home_code", "away_code",
        "home_form", "away_form",
        "home_avg_goals", "away_avg_goals",
        "home_rest", "away_rest"
    ]

    X = df[features]
    y = df["target"]

    # Split by time: older matches train, recent matches test (more realistic than random split)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Training on {len(X_train)} matches, testing on {len(X_test)}...")

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Prevents model from ignoring rare draws
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Breakdown:")
    print(classification_report(
        y_test, preds,
        target_names=["Home Win", "Draw", "Away Win"]
    ))

    importance = pd.Series(model.feature_importances_, index=features)
    print("\nFeature Importance (higher = more influential):")
    print(importance.sort_values(ascending=False))

    # Save model
    joblib.dump(model, "football_model.pkl")

    # Save team stats lookup table for the web app
    team_stats = df.groupby('home_team').agg(
        home_code=('home_code', 'last'),
        away_code=('away_code', 'last'),
        home_form=('home_form', 'mean'),
        away_form=('away_form', 'mean'),
        home_avg_goals=('home_avg_goals', 'mean'),
        away_avg_goals=('away_avg_goals', 'mean')
    ).reset_index()
    team_stats.to_csv("team_stats.csv", index=False)

    print("\nPhase 3 Complete! Saved 'football_model.pkl' and 'team_stats.csv'")

if __name__ == "__main__":
    train_predictor(df)
