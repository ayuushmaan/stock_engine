# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 -- Sponsored Content Classifier Validation
#
# Audits the LightGBM classifier trained in pipeline step 04.
# Checks label distributions, feature importance circularity,
# tone separation, source-level sanity, and the uncertain zone.

# %% tags=["setup"]
import sys, pickle, hashlib, json, platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path(".").resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_PROCESSED, MODELS_DIR, OUTPUTS_FIGURES,
    SPONSORED_PROB_HIGH, SPONSORED_PROB_LOW,
    NUMPY_SEED, seed_everything,
)

seed_everything(NUMPY_SEED)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

FIG_DIR = OUTPUTS_FIGURES / "classifier_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plotting defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
COLORS = {"organic": "#2563eb", "uncertain": "#9333ea", "sponsored": "#dc2626"}

# %% tags=["load"]
# ── Load data and model ──────────────────────────────────────────
df = pd.read_parquet(DATA_PROCESSED / "sponsored_scores.parquet")
print(f"Loaded {len(df):,} articles  |  columns: {len(df.columns)}")

with open(MODELS_DIR / "sponsored_classifier.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = model.feature_name()
print(f"Model features ({len(feature_names)}): {feature_names}")

# %% [markdown]
# ---
# ## Section 1 — Label Distribution

# %%
n_sponsored = (df["weak_label"] == 1).sum()
n_organic   = (df["weak_label"] == 0).sum()
n_uncertain = df["weak_label"].isna().sum()
n_total     = len(df)

pct_labeled = 100 * (n_sponsored + n_organic) / n_total

print("=== Weak-Label Distribution ===")
print(f"  SPONSORED (1):  {n_sponsored:>8,}  ({100*n_sponsored/n_total:.1f}%)")
print(f"  ORGANIC   (0):  {n_organic:>8,}  ({100*n_organic/n_total:.1f}%)")
print(f"  UNCERTAIN (NaN):{n_uncertain:>8,}  ({100*n_uncertain/n_total:.1f}%)")
print(f"  ------------------------")
print(f"  TOTAL:          {n_total:>8,}")
print(f"  % with definite label: {pct_labeled:.1f}%")

# %%
# Bar chart of weak label counts by source tier
tier_label_counts = (
    df.assign(label_str=df["weak_label"].map({1.0: "Sponsored", 0.0: "Organic"}).fillna("Uncertain"))
    .groupby(["source_tier", "label_str"]).size().unstack(fill_value=0)
)
tier_label_counts = tier_label_counts.reindex(columns=["Organic", "Uncertain", "Sponsored"], fill_value=0)

fig, ax = plt.subplots(figsize=(8, 5))
tier_label_counts.plot(
    kind="bar", stacked=True, ax=ax,
    color=[COLORS["organic"], COLORS["uncertain"], COLORS["sponsored"]],
    edgecolor="white", linewidth=0.5,
)
ax.set_xlabel("Source Tier")
ax.set_ylabel("Article Count")
ax.set_title("Weak-Label Distribution by Source Tier")
ax.legend(title="Weak Label")
ax.set_xticklabels([f"Tier {int(t)}" for t in tier_label_counts.index], rotation=0)
fig.savefig(FIG_DIR / "label_distribution_by_tier.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'label_distribution_by_tier.png'}")

# %% [markdown]
# ---
# ## Section 2 — Circularity Check
#
# If `IS_PR_WIRE` and URL-pattern features dominate (>60% total gain),
# the classifier is just *memorizing* the weak-label rules, not learning
# any generalizable signal from tone/textual features.

# %%
importances = model.feature_importance(importance_type="gain")
imp_df = (
    pd.DataFrame({"feature": feature_names, "gain": importances})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
total_gain = imp_df["gain"].sum()
imp_df["pct"] = 100 * imp_df["gain"] / total_gain if total_gain > 0 else 0

# ── Recreate Train/Validation splits for Permutation & Ablation ──
labelled = df[df["weak_label"].notna()].copy()
train_mask = (labelled["effective_date"] >= "2020-01-01") & (labelled["effective_date"] <= "2022-12-31")
valid_mask = (labelled["effective_date"] >= "2023-01-01") & (labelled["effective_date"] <= "2023-12-31")

train_df = labelled[train_mask]
valid_df = labelled[valid_mask]

if len(train_df) < 100:
    # random split fallback
    shuffled = labelled.sample(frac=1.0, random_state=42)
    split_idx = int(len(shuffled) * 0.7)
    train_df = shuffled.iloc[:split_idx].copy()
    valid_df = shuffled.iloc[split_idx:].copy()

X_train = train_df[feature_names].values
y_train = train_df["weak_label"].values
X_valid = valid_df[feature_names].values
y_valid = valid_df["weak_label"].values

# ── Permutation Importance on Validation Slice ──
def get_permutation_importance(model, X, y, features, n_repeats=5, seed=42):
    np.random.seed(seed)
    baseline_pred = model.predict(X)
    baseline_auc = roc_auc_score(y, baseline_pred) if len(np.unique(y)) > 1 else 0.5
    
    importances_list = []
    for i, feat in enumerate(features):
        auc_drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            perm_pred = model.predict(X_perm)
            perm_auc = roc_auc_score(y, perm_pred) if len(np.unique(y)) > 1 else 0.5
            auc_drops.append(baseline_auc - perm_auc)
        importances_list.append({
            "feature": feat,
            "perm_importance": float(np.mean(auc_drops)),
            "perm_std": float(np.std(auc_drops))
        })
    return pd.DataFrame(importances_list).sort_values("perm_importance", ascending=False).reset_index(drop=True), baseline_auc

perm_df, auc_full = get_permutation_importance(model, X_valid, y_valid, feature_names)

# ── Feature Ablation Study ───────────────────────────────────────
def train_lgb_subset(train_data, valid_data, features_to_use, early_stopping=50):
    X_tr = train_data[features_to_use].values
    y_tr = train_data["weak_label"].values
    X_va = valid_data[features_to_use].values
    y_va = valid_data["weak_label"].values
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": 42
    }
    
    # Balance classes
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    if n_pos > 0 and n_neg > 0:
        lgb_params["scale_pos_weight"] = n_neg / n_pos
        
    d_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=features_to_use)
    d_va = lgb.Dataset(X_va, label=y_va, feature_name=features_to_use)
    
    m = lgb.train(
        lgb_params,
        d_tr,
        num_boost_round=300,
        valid_sets=[d_tr, d_va],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)]
    )
    
    pred_va = m.predict(X_va)
    auc = roc_auc_score(y_va, pred_va) if len(np.unique(y_va)) > 1 else 0.5
    return auc, m

rule_features = ["is_pr_wire", "has_promo_ratio", "has_risk_words"]
no_rules_features = [f for f in feature_names if f not in rule_features]
text_only_features = [f for f in feature_names if f not in rule_features + ["source_tier"]]

auc_no_rules, _ = train_lgb_subset(train_df, valid_df, no_rules_features)
auc_text_only, _ = train_lgb_subset(train_df, valid_df, text_only_features)

print("=== Feature Importance (Gain) ===")
print(imp_df.to_string(index=False))

print("\n=== Feature Importance (Permutation on Validation Slice) ===")
print(perm_df.to_string(index=False))

print("\n=== Feature Ablation Analysis ===")
print(f"  Full Model AUC (with all features):    {auc_full:.4f}")
print(f"  Ablated AUC (no rules/PR/URL features): {auc_no_rules:.4f}")
print(f"  Text-only AUC (no rules & no domain):  {auc_text_only:.4f}")

# Circularity flag rules
rule_pct = imp_df.loc[imp_df["feature"].isin(rule_features), "pct"].sum()
source_tier_pct = imp_df.loc[imp_df["feature"] == "source_tier", "pct"].sum()

# Permutation importance rules
rule_perm_val = perm_df.loc[perm_df["feature"].isin(rule_features), "perm_importance"].sum()
total_perm_val = perm_df["perm_importance"].sum()
rule_perm_pct = 100 * rule_perm_val / total_perm_val if total_perm_val > 0 else 0.0

ablation_drop = auc_full - auc_no_rules
ablation_collapsed = (auc_no_rules < 0.65) or (ablation_drop > 0.15)
gain_dominated = (rule_pct > 60.0)
perm_dominated = (rule_perm_pct > 60.0)

is_circular = gain_dominated or perm_dominated or ablation_collapsed

print(f"\nCircularity Diagnostics:")
print(f"  Rule-based features gain importance: {rule_pct:.1f}%  (Dominate > 60%: {gain_dominated})")
print(f"  Rule-based features perm importance: {rule_perm_pct:.1f}%  (Dominate > 60%: {perm_dominated})")
print(f"  source_tier gain importance:         {source_tier_pct:.1f}%")
print(f"  Ablation AUC Drop (Full - Ablated):  {ablation_drop:.4f}")
print(f"  Ablation collapsed (AUC < 0.65 or Drop > 0.15): {ablation_collapsed}")
print(f"[!!] CIRCULARITY FLAG: {'YES -- memorization, not generalization' if is_circular else 'No -- classifier learned beyond rules'}")

# %%
# Feature importance bar charts (Gain and Permutation side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Gain
bars1 = ax1.barh(imp_df["feature"], imp_df["pct"], color="#3b82f6", edgecolor="white", linewidth=0.5)
for bar, feat in zip(bars1, imp_df["feature"]):
    if feat in rule_features or feat == "source_tier":
        bar.set_color("#ef4444")
ax1.set_xlabel("Importance (% of Total Gain)")
ax1.set_title("Feature Importance (Gain)\n(Red = rule/domain features)")
ax1.invert_yaxis()

# Permutation
perm_sorted = perm_df.sort_values("perm_importance", ascending=True)
bars2 = ax2.barh(perm_sorted["feature"], perm_sorted["perm_importance"], color="#10b981", edgecolor="white", linewidth=0.5)
for bar, feat in zip(bars2, perm_sorted["feature"]):
    if feat in rule_features or feat == "source_tier":
        bar.set_color("#ef4444")
ax2.set_xlabel("Mean AUC Drop on Shuffle")
ax2.set_title("Permutation Feature Importance\n(Validation Slice)")

plt.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_audit.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'feature_importance_audit.png'}")

# %%
# Strict "No IS_PR_WIRE and No URL Signal" Filter
from config.settings import SPONSORED_URL_PATTERNS, PR_WIRE_DOMAINS

def has_pr_wire_domain(url: str) -> bool:
    url_lower = str(url).lower()
    return any(d in url_lower for d in PR_WIRE_DOMAINS)

def has_sponsored_url_pattern(url: str) -> bool:
    url_lower = str(url).lower()
    return any(pat in url_lower for pat in SPONSORED_URL_PATTERNS)

print("\n=== Strict Filter Step-by-Step Breakdown for Ambiguous Subset ===")
print(f"  Initial total articles: {len(df):,}")

# Step-by-step masks
mask_no_pr_feat = df["is_pr_wire"] == 0
print(f"  1. After is_pr_wire == 0 (feature):           {mask_no_pr_feat.sum():,} ({100*mask_no_pr_feat.sum()/len(df):.1f}%)")

mask_no_pr_domain = ~df["source_url"].fillna("").apply(has_pr_wire_domain)
mask_combined = mask_no_pr_feat & mask_no_pr_domain
print(f"  2. After filtering out PR wire URL domains:   {mask_combined.sum():,} ({100*mask_combined.sum()/len(df):.1f}%)")

mask_no_promo = df["has_promo_ratio"] == 0.0
mask_combined = mask_combined & mask_no_promo
print(f"  3. After filtering out promo URL keywords:    {mask_combined.sum():,} ({100*mask_combined.sum()/len(df):.1f}%)")

mask_no_risk = df["has_risk_words"] == 0
mask_combined = mask_combined & mask_no_risk
print(f"  4. After filtering out risk URL keywords:     {mask_combined.sum():,} ({100*mask_combined.sum()/len(df):.1f}%)")

mask_no_spon_pat = ~df["source_url"].fillna("").apply(has_sponsored_url_pattern)
strict_mask = mask_combined & mask_no_spon_pat
print(f"  5. After filtering out sponsored URL patterns:{strict_mask.sum():,} ({100*strict_mask.sum()/len(df):.1f}%)")

ambiguous = df[strict_mask].copy()

print(f"\n=== Strictly Ambiguous Articles (No PR-wire, No URL keywords, No URL patterns) ===")
print(f"Count: {len(ambiguous):,} / {len(df):,} ({100*len(ambiguous)/len(df):.1f}%)")
print(f"sponsored_prob distribution:")
print(ambiguous["sponsored_prob"].describe().to_string())

# Breakdown of ambiguous articles by tier
print("\n=== sponsored_prob on Strictly Ambiguous Articles by Source Tier ===")
tier_ambiguous = (
    ambiguous.groupby("source_tier")["sponsored_prob"]
    .agg(["count", "mean", "std", "min", "max"])
)
print(tier_ambiguous.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ambiguous["sponsored_prob"], bins=50, color="#6366f1", edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(SPONSORED_PROB_HIGH, color="#dc2626", ls="--", lw=1.5, label=f"High cutoff ({SPONSORED_PROB_HIGH})")
ax.axvline(SPONSORED_PROB_LOW, color="#2563eb", ls="--", lw=1.5, label=f"Low cutoff ({SPONSORED_PROB_LOW})")
ax.set_xlabel("sponsored_prob")
ax.set_ylabel("Count")
ax.set_title("Predicted Probabilities for STRICTLY Ambiguous Articles\n(No PR-wire, no URL keyword features/patterns)")
ax.legend()
fig.savefig(FIG_DIR / "ambiguous_articles_prob_dist.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'ambiguous_articles_prob_dist.png'}")


# %% [markdown]
# ---
# ## Section 3 — Tone Distribution Validation
#
# If the classifier is working, high-sponsored articles should
# cluster at high positive GDELT tone (+5 to +15), while organic
# articles should have a wider distribution centered near 0.

# %%
df["prob_group"] = pd.cut(
    df["sponsored_prob"],
    bins=[-0.01, SPONSORED_PROB_LOW, SPONSORED_PROB_HIGH, 1.01],
    labels=["high_organic", "uncertain", "high_sponsored"],
)
df["prob_group"] = pd.Categorical(
    df["prob_group"],
    categories=["high_organic", "uncertain", "high_sponsored"],
    ordered=True
)

group_stats = (
    df.groupby("prob_group", observed=True)["tone_score"]
    .agg(["count", "mean", "std", "median"])
    .rename(columns={"count": "n", "mean": "mean_tone", "std": "std_tone", "median": "median_tone"})
)
print("=== Tone by Probability Group ===")
print(group_stats.to_string())

tone_sep_delta = (
    group_stats.loc["high_sponsored", "mean_tone"] if "high_sponsored" in group_stats.index else np.nan
) - (
    group_stats.loc["high_organic", "mean_tone"] if "high_organic" in group_stats.index else np.nan
)
print(f"\nTone separation delta (high_sponsored - high_organic): {tone_sep_delta:.3f}")
if np.isnan(tone_sep_delta):
    print("[!!] Cannot compute delta -- one or both groups are empty")
elif tone_sep_delta > 2:
    print("[OK] Strong tone separation -- classifier captures real behavioral difference")
elif tone_sep_delta > 0:
    print("[!!] Weak tone separation -- classifier may be partially circular")
else:
    print("[X] No tone separation -- classifier is NOT capturing tone differences")

# %%
# Violin plot overall
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(
    data=df,
    x="prob_group",
    y="tone_score",
    hue="prob_group",
    palette={"high_organic": COLORS["organic"], "uncertain": COLORS["uncertain"], "high_sponsored": COLORS["sponsored"]},
    legend=False,
    inner="quart",
    ax=ax
)
ax.axhline(0, color="gray", ls="--", alpha=0.4)
ax.set_xlabel("Classifier Probability Group")
ax.set_ylabel("GDELT Tone Score")
ax.set_title("Tone Distribution by Classifier Probability Group")
ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG_DIR / "tone_violin_by_prob_group.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'tone_violin_by_prob_group.png'}")

# %%
# Stratified Tone Violin Plot by Source Tier (Secondary View)
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(
    data=df,
    x="source_tier",
    y="tone_score",
    hue="prob_group",
    palette={"high_organic": COLORS["organic"], "uncertain": COLORS["uncertain"], "high_sponsored": COLORS["sponsored"]},
    inner="quart",
    ax=ax
)
ax.axhline(0, color="gray", ls="--", alpha=0.4)
ax.set_xlabel("Source Tier")
ax.set_ylabel("GDELT Tone Score")
ax.set_title("Tone Distribution by Classifier Group, Stratified by Source Tier")
ax.legend(title="Classifier Group")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "tone_violin_stratified_by_tier.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'tone_violin_stratified_by_tier.png'}")

# %% [markdown]
# ---
# ## Section 4 — Source Sanity Check
#
# PR-wire domains should rank highest by mean `sponsored_prob`.
# Tier-1 outlets (ET, Mint, BS) should be low. Any tier-1 domain
# with mean > 0.5 is a bug.

# %%
domain_stats = (
    df.groupby("source_domain")
    .agg(
        n=("sponsored_prob", "size"),
        mean_prob=("sponsored_prob", "mean"),
        mean_tone=("tone_score", "mean"),
        tier=("source_tier", "first"),
    )
    .sort_values("mean_prob", ascending=False)
)
domain_stats["mean_prob_with_count"] = domain_stats.apply(
    lambda r: f"{r['mean_prob']:.4f} (n={int(r['n'])})", axis=1
)

print("=== Top 20 Domains by Mean sponsored_prob (min 5 articles, all tiers) ===")
print(domain_stats[domain_stats["n"] >= 5][["mean_prob_with_count", "mean_tone", "tier"]].head(20).to_string())

# %%
tier1_stats = domain_stats[domain_stats["tier"] == 1].sort_values("mean_prob", ascending=False)
print("\n=== Top 20 Tier-1 Domains by Mean sponsored_prob (min 5 articles) ===")
print(tier1_stats[tier1_stats["n"] >= 5][["mean_prob_with_count", "mean_tone", "tier"]].head(20).to_string())

tier1_bugs = tier1_stats[tier1_stats["mean_prob"] > 0.5]
if len(tier1_bugs) > 0:
    print(f"\n[BUG] {len(tier1_bugs)} tier-1 domains have mean sponsored_prob > 0.5:")
    print(tier1_bugs[["mean_prob_with_count", "mean_tone"]].to_string())
else:
    print("\n[OK] No tier-1 domains with mean sponsored_prob > 0.5")

# %%
# Horizontal bar chart — top 20 domains (min 5 articles)
top20 = domain_stats[domain_stats["n"] >= 5].head(20).copy()
tier_colors = {1: COLORS["organic"], 2: COLORS["uncertain"], 3: COLORS["sponsored"]}
bar_colors = [tier_colors.get(int(t), "#999") for t in top20["tier"]]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(top20)), top20["mean_prob"], color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f"{d}  (n={int(top20.loc[d, 'n'])})" for d in top20.index], fontsize=9)
ax.set_xlabel("Mean sponsored_prob")
ax.set_title("Top 20 Domains by Mean Sponsored Probability (min 5 articles)\n(Blue=Tier1, Purple=Tier2, Red=Tier3)")
ax.invert_yaxis()
ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
fig.savefig(FIG_DIR / "top_domains_by_sponsored_prob.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'top_domains_by_sponsored_prob.png'}")

# %% [markdown]
# ---
# ## Section 5 — Uncertain Zone Analysis
#
# Articles in the 0.3 – 0.7 probability band represent the grey area
# where the classifier is least confident.

# %%
uncertain_mask = (df["sponsored_prob"] > SPONSORED_PROB_LOW) & (df["sponsored_prob"] < SPONSORED_PROB_HIGH)
uncertain_df = df[uncertain_mask].copy()
print(f"=== Uncertain Zone (0.3 < prob < 0.7) ===")
print(f"Count: {len(uncertain_df):,} / {len(df):,} ({100*len(uncertain_df)/len(df):.1f}%)")

uncertain_domains = (
    uncertain_df.groupby("source_domain")
    .agg(
        n=("sponsored_prob", "size"),
        mean_prob=("sponsored_prob", "mean"),
        mean_tone=("tone_score", "mean"),
        tier=("source_tier", "first"),
    )
    .sort_values("n", ascending=False)
)
print(f"\nTop 20 domains in the uncertain zone:")
print(uncertain_domains.head(20).to_string())

print(f"\nAverage tone score in uncertain zone: {uncertain_df['tone_score'].mean():.3f}")
print(f"Average tone score overall:           {df['tone_score'].mean():.3f}")

# %% [markdown]
# ---
# ## Summary

# %%
pct_high_sponsored = 100 * (df["sponsored_prob"] > SPONSORED_PROB_HIGH).sum() / n_total
pct_high_organic   = 100 * (df["sponsored_prob"] < SPONSORED_PROB_LOW).sum() / n_total
pct_uncertain_zone = 100 * uncertain_mask.sum() / n_total

is_pr_wire_imp = imp_df.loc[imp_df["feature"] == "is_pr_wire", "pct"].values
is_pr_wire_imp = float(is_pr_wire_imp[0]) if len(is_pr_wire_imp) > 0 else 0.0

summary = {
    "total_articles": int(n_total),
    "pct_labeled": round(float(pct_labeled), 2),
    "pct_high_sponsored": round(float(pct_high_sponsored), 2),
    "pct_high_organic": round(float(pct_high_organic), 2),
    "pct_uncertain": round(float(pct_uncertain_zone), 2),
    "tone_separation_delta": round(float(tone_sep_delta), 4) if not np.isnan(tone_sep_delta) else None,
    "classifier_is_circular": bool(is_circular),
    "ablation_collapsed": bool(ablation_collapsed),
    "auc_full": round(float(auc_full), 4),
    "auc_no_rules": round(float(auc_no_rules), 4),
    "auc_text_only": round(float(auc_text_only), 4),
    "is_pr_wire_importance_pct": round(float(is_pr_wire_imp), 2),
    "rule_features_importance_pct": round(float(rule_pct), 2),
    "source_tier_importance_pct": round(float(source_tier_pct), 2),
}

print("=== CLASSIFIER VALIDATION SUMMARY ===")
print(json.dumps(summary, indent=2))

# ── Reproducibility manifest ─────────────────────────────────────
data_bytes = (DATA_PROCESSED / "sponsored_scores.parquet").read_bytes()
manifest = {
    "notebook": "01_classifier_validation",
    "run_timestamp": datetime.now().isoformat(),
    "dataset_hash": hashlib.sha256(data_bytes).hexdigest()[:16],
    "dataset_rows": int(n_total),
    "random_seed": NUMPY_SEED,
    "python_version": platform.python_version(),
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
    "summary": summary,
}
manifest_path = FIG_DIR / "repro_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
print(f"\nSaved repro manifest: {manifest_path}")
