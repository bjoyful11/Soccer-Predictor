import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report

model = joblib.load("football_model.pkl")
team_stats = pd.read_csv("team_stats.csv")

FEATURES = [
    "home_code", "away_code",
    "home_form", "away_form",
    "home_avg_goals", "away_avg_goals",
    "home_rest", "away_rest"
]

RESULT_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def get_team(name):
    row = team_stats[team_stats["home_team"] == name]
    if row.empty:
        raise ValueError(f"Team not found: '{name}'. Check team_stats.csv for exact names.")
    return row.iloc[0]


def predict_match(home, away, home_rest=5, away_rest=5):
    h = get_team(home)
    a = get_team(away)

    row = pd.DataFrame([{
        "home_code":      h["home_code"],
        "away_code":      a["away_code"],
        "home_form":      h["home_form"],
        "away_form":      a["away_form"],
        "home_avg_goals": h["home_avg_goals"],
        "away_avg_goals": a["away_avg_goals"],
        "home_rest":      home_rest,
        "away_rest":      away_rest,
    }])

    probs = model.predict_proba(row)[0]
    predicted = model.predict(row)[0]
    return predicted, probs


#Feature Importance

print("\n" + "=" * 50)
print("  FEATURE IMPORTANCE")
print("  (what the model weighs most heavily)")
print("=" * 50)

importance = pd.Series(model.feature_importances_, index=FEATURES)
importance = importance.sort_values(ascending=False)

print()
for feature, score in importance.items():
    bar = "█" * int(score * 200)
    print(f"  {feature:<20} {bar:<30} {score:.3f}")

print("=" * 50)


#Accuracy & Precision

print("\n" + "=" * 58)
print("  MODEL ACCURACY & PRECISION (held-out test set)")
print("=" * 58)

df = pd.read_csv("pl_matches_processed.csv")
X = df[FEATURES]
y = df["target"]

train_size = int(len(df) * 0.8)
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")

print(f"\n  Test set size : {len(y_test)} matches")
print(f"  Overall Accuracy  : {acc*100:.1f}%")
print(f"  Weighted Precision: {precision*100:.1f}%")

print("\n  Per-class breakdown:")
print(f"\n  {'Outcome':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Matches':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

report = classification_report(
    y_test, preds,
    target_names=["Home Win", "Draw", "Away Win"],
    output_dict=True
)
for label in ["Home Win", "Draw", "Away Win"]:
    r = report[label]
    print(f"  {label:<12} {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% {r['f1-score']*100:>9.1f}% {int(r['support']):>10}")

print("=" * 58)

# Live Predictions
print("\n" + "=" * 50)
print("  LIVE PREDICTION")
print("=" * 50)

teams = sorted(team_stats["home_team"].tolist())
team_map = {i: name for i, name in enumerate(teams, 1)}

print("\n  Available teams:")
for i, t in team_map.items():
    print(f"    {i:>2}. {t}")

def pick_team(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            n = int(raw)
            if n in team_map:
                return team_map[n]
            print(f"  Please enter a number between 1 and {len(team_map)}.")
        except ValueError:
            print("  Please enter a number.")

print()
while True:
    home_input = pick_team("  Home team number: ")
    away_input = pick_team("  Away team number: ")

    result, probs = predict_match(home_input, away_input)
    print(f"\n  {home_input}  vs  {away_input}\n")
    print(f"  Home Win  : {probs[0]*100:.1f}%")
    print(f"  Draw      : {probs[1]*100:.1f}%")
    print(f"  Away Win  : {probs[2]*100:.1f}%")
    print(f"\n  Prediction: {RESULT_LABELS[result]}")

    print()
    again = input("  Predict another match? (y/n): ").strip().lower()
    if again != "y":
        break

print("=" * 58)
