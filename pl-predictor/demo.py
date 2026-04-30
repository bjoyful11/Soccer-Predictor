"""
demo.py — Presentation demo for the PL Match Predictor
Each section is self-contained and can be screenshotted for slides.
"""

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


# ─────────────────────────────────────────────
# SLIDE 1 — Single Match Prediction
# ─────────────────────────────────────────────
print("=" * 50)
print("  SINGLE MATCH PREDICTION")
print("=" * 50)

home_team = "Arsenal FC"
away_team = "Chelsea FC"

result, probs = predict_match(home_team, away_team)

print(f"\n  {home_team}  vs  {away_team}\n")
print(f"  Home Win  : {probs[0]*100:.1f}%")
print(f"  Draw      : {probs[1]*100:.1f}%")
print(f"  Away Win  : {probs[2]*100:.1f}%")
print(f"\n  Prediction: {RESULT_LABELS[result]}")
print("=" * 50)


# ─────────────────────────────────────────────
# SLIDE 2 — Batch Predictions (classic matchups)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BATCH PREDICTIONS — CLASSIC PL MATCHUPS")
print("=" * 60)

fixtures = [
    ("Manchester City FC",  "Liverpool FC"),
    ("Arsenal FC",          "Tottenham Hotspur FC"),
    ("Chelsea FC",          "Manchester United FC"),
    ("Newcastle United FC", "Aston Villa FC"),
    ("Liverpool FC",        "Manchester City FC"),
]

print(f"\n  {'Home':<28} {'Away':<28} {'Prediction'}")
print(f"  {'-'*28} {'-'*28} {'-'*15}")

for home, away in fixtures:
    result, _ = predict_match(home, away)
    print(f"  {home:<28} {away:<28} {RESULT_LABELS[result]}")

print("=" * 60)


# ─────────────────────────────────────────────
# SLIDE 3 — Full Probability Breakdown (one match)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  PROBABILITY BREAKDOWN")
print("=" * 50)

home_team = "Liverpool FC"
away_team = "Manchester City FC"

result, probs = predict_match(home_team, away_team)

print(f"\n  {home_team}  vs  {away_team}\n")

for label, prob in zip(["Home Win", "Draw", "Away Win"], probs):
    bar = "█" * int(prob * 30)
    print(f"  {label:<10} {bar:<30} {prob*100:.1f}%")

print(f"\n  Prediction: {RESULT_LABELS[result]}")
print("=" * 50)


# ─────────────────────────────────────────────
# SLIDE 4 — Feature Importance
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# SLIDE 5 — What goes into a prediction (raw input)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("  WHAT THE MODEL SEES (raw input features)")
print("=" * 50)

home_team = "Arsenal FC"
away_team = "Manchester City FC"

h = get_team(home_team)
a = get_team(away_team)

print(f"\n  Match: {home_team} vs {away_team}\n")
print(f"  {'Feature':<20} {'Home':>10} {'Away':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10}")
print(f"  {'Team code':<20} {int(h['home_code']):>10} {int(a['away_code']):>10}")
print(f"  {'Form (pts/game)':<20} {h['home_form']:>10.2f} {a['away_form']:>10.2f}")
print(f"  {'Avg goals/game':<20} {h['home_avg_goals']:>10.2f} {a['away_avg_goals']:>10.2f}")
print(f"  {'Rest days':<20} {'5':>10} {'5':>10}")

result, probs = predict_match(home_team, away_team)
print(f"\n  Prediction → {RESULT_LABELS[result]}  "
      f"({probs[0]*100:.1f}% / {probs[1]*100:.1f}% / {probs[2]*100:.1f}%)")
print("=" * 50)


# ─────────────────────────────────────────────
# SLIDE 6 — Accuracy & Precision on Test Set
# ─────────────────────────────────────────────
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
