# --- 1) Recreate MA3 baseline mape per key (Jul-Dec 2025) ---
ma_df = so_fcst_df.with_columns(
  pl.col("so_nw_ct").shift(3).alias("m-0"),
  pl.col("so_nw_ct").shift(4).alias("m-1"),
  pl.col("so_nw_ct").shift(5).alias("m-2"),
).filter(pl.col("periods") >= "2023 01").with_columns(
  ma3=(pl.col("m-0")+pl.col("m-1")+pl.col("m-2"))/3
).with_columns(
  re = pl.col("ma3") - pl.col("so_nw_ct"),
  ae = (pl.col("ma3") - pl.col("so_nw_ct")).abs(),
  mape = pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae")/pl.col("so_nw_ct")).otherwise(1)
).with_columns(
  mape = pl.when(pl.col("mape") > 1).then(1).otherwise(pl.col("mape"))
)

# Keep Jul-Dec 2025 for evaluation
eval_df = ma_df.filter((pl.col("periods") >= "2025 07") & (pl.col("periods") <= "2025 12"))

# Pivot to get mape per period per key, then compute mape_avg
eval_perf = eval_df.pivot(
  on="periods",
  index="key",
  values="mape",
  aggregate_function="sum",
  sort_columns=True
).with_columns(
  mape_avg = pl.col(pl.exclude("key")).mean_horizontal()
)

# Also compute 2025 sales sum per key (for Pareto)
sales_2025 = so_fcst_df.filter(pl.col("periods").str.contains("2025")).group_by("key").agg(
  pl.col("so_nw_ct").sum().alias("sales_2025")
)

# --- 2) Pareto 80 selection based on 2025 sales ---
pareto_df = sales_2025.sort("sales_2025", descending=True).with_columns(
  cumsum = pl.col("sales_2025").cumsum(),
  cumsum_pct = pl.col("sales_2025").cumsum() / pl.col("sales_2025").sum()
).filter(pl.col("cumsum_pct") < 0.8).select(["key", "sales_2025"])

# Merge pareto flag into eval_perf
eval_perf = eval_perf.join(pareto_df, on="key", how="left").with_columns(
  pareto80_flag = pl.when(pl.col("sales_2025").is_null()).then(0).otherwise(1)
)

# Recover zone, soldto, shipto, ac fields from key
split_df = eval_perf.with_columns(
  pl.col("key").str.split_exact("_", 4).struct.rename_fields(["zone","soldto","shipto","ac"])
).unnest("key")

# Filter to Pareto only for zone ranking etc
pareto_perf = split_df.filter(pl.col("pareto80_flag")==1)

# --- 3) Zone-level stats (Pareto only) ---
zone_stats = pareto_perf.groupby("zone").agg([
  pl.col("mape_avg").mean().alias("mean_mape"),
  pl.col("mape_avg").median().alias("median_mape"),
  pl.col("mape_avg").max().alias("max_mape"),
  pl.col("sales_2025").sum().alias("sales_2025_total"),
  pl.col("mape_avg").count().alias("pareto_count")
]).sort("mean_mape", descending=False)

# Determine best and worst zones
best_zone = zone_stats.sort("mean_mape").select("zone").to_pandas()["zone"].iloc[0]
worst_zone = zone_stats.sort("mean_mape", descending=True).select("zone").to_pandas()["zone"].iloc[0]

print("Zone ranking (Pareto only):")
print(zone_stats.to_pandas())

print(f"\nBest zone (lowest mean): {best_zone}")
print(f"Worst zone (highest mean): {worst_zone}")

# --- 4) Worst zone: top 3 worst error (Pareto keys) ---
worst_zone_keys = pareto_perf.filter(pl.col("zone")==worst_zone).sort("mape_avg", descending=True)
top3_worst = worst_zone_keys.head(3).select(["key","mape_avg","sales_2025"]).to_pandas()
print("\nTop 3 worst Pareto keys in worst zone (by mape_avg):")
print(top3_worst)

# Plot function: time series actual vs ma3 and monthly mape
def plot_key_trends(key_list):
  for k in key_list:
    # extract monthly series from ma_df
    tmp = ma_df.filter(pl.col("key")==k).select(["periods","so_nw_ct","m-0","m-1","m-2","ma3","mape"]).to_pandas().sort_values("periods")
    tmp['periods_dt'] = pd.to_datetime(tmp['periods'].str.replace(" ", "-")+"-01")
    # plot actual vs ma3
    plt.figure(figsize=(10,4))
    plt.plot(tmp['periods_dt'], tmp['so_nw_ct'], label='actual')
    plt.plot(tmp['periods_dt'], tmp['ma3'], label='ma3')
    plt.title(f"Actual vs MA3 - {k}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plot monthly mape
    plt.figure(figsize=(10,3))
    plt.bar(tmp['periods_dt'], tmp['mape'])
    plt.title(f"Monthly MAPE - {k}")
    plt.tight_layout()
    plt.show()

# Plot the top3 worst
plot_key_trends(top3_worst['key'].tolist())



# --- 5) Best zone: top 3 by sales (largest Pareto combos) ---
best_zone_keys = pareto_perf.filter(pl.col("zone")==best_zone).sort("sales_2025", descending=True)
top3_best_by_sales = best_zone_keys.head(3).select(["key","mape_avg","sales_2025"]).to_pandas()
print("\nTop 3 Pareto keys in best zone (by sales_2025):")
print(top3_best_by_sales)

# Plot them
plot_key_trends(top3_best_by_sales['key'].tolist())

# --- 6) Zone 2B special: find biggest average error but exclude mape_avg == 1.0; use >=0.9 threshold,
#   fallback to top6 or top9 if not enough hits. ---
zone_target = "ZONE02B"
z2b = pareto_perf.filter(pl.col("zone")==zone_target).sort("mape_avg", descending=True).select(["key","mape_avg","sales_2025"]).to_pandas()

# drop perfect-100% rows (mape_avg == 1.0)
z2b = z2b[z2b['mape_avg'] < 1.0]

threshold = 0.9
sel = z2b[z2b['mape_avg'] >= threshold].copy()

if len(sel) < 3:
  # fallback: top 6
  sel = z2b.head(6)
if len(sel) < 3:
  # final fallback: top 9
  sel = z2b.head(9)

print(f"\nZone {zone_target} special selection (threshold 0.9 fallback to top6/top9): count={len(sel)}")
print(sel)

# Plot the special selection keys
if len(sel)>0:

  plot_key_trends(sel['key'].tolist())






ma_df = (
    so_fcst_df
    .with_columns(
        pl.col("so_nw_ct").shift(3).alias("m-0"),
        pl.col("so_nw_ct").shift(4).alias("m-1"),
        pl.col("so_nw_ct").shift(5).alias("m-2"),
    )
    .filter(pl.col("periods") >= "2023 01")
    .with_columns(
        ma3 = (pl.col("m-0") + pl.col("m-1") + pl.col("m-2")) / 3
    )
    .with_columns(
        re = pl.col("ma3") - pl.col("so_nw_ct"),
        ae = (pl.col("ma3") - pl.col("so_nw_ct")).abs()
    )
    .with_columns(
        mape = pl.when(pl.col("so_nw_ct") > 0)
                 .then(pl.col("ae") / pl.col("so_nw_ct"))
                 .otherwise(1)
    )
    .with_columns(
        mape = pl.when(pl.col("mape") > 1).then(1).otherwise(pl.col("mape"))
    )
)

# Evaluation window
eval_df = ma_df.filter(
    (pl.col("periods") >= "2025 07") &
    (pl.col("periods") <= "2025 12")
)



















eval_perf = (
    eval_df
    .pivot(
        on="periods",
        index="key",
        values="mape",
        aggregate_function="sum",
        sort_columns=True
    )
    .with_columns(
        mape_avg = pl.mean_horizontal(pl.exclude("key"))
    )
)


