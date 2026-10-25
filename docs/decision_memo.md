# Which in-season slumps, hot starts and breakouts should you believe?

*Evidence from backtesting rest-of-season forecasts, 2018–2026 (2020 excluded). Method: [model card](model_card.md). The shares in section 1 use the 2026 fitted values.*

## Bottom line

1. **Don't act on season-to-date numbers alone.** Taken at face value, a hitter's wOBA so far predicts the rest of the season *worse than assuming everyone is league average*: RMSE .0742 vs .0471.
2. **Slumps mostly fix themselves; hot starts are where judgment pays off.** Even a naive list of the worst wOBA hitters on June 1 improves 96% of the time. A naive list of the *best* hitters holds up far less well, and the model's edge in spotting hot starts that won't last grows as the season goes on.
3. **"Breakouts" are close to coin flips.** Whether a hitter beats their *preseason* projection the rest of the way is a much harder call, and the model's edge here is real but modest.
4. **Weigh contact quality and park changes more than results.** The best read of a hitter's current wOBA is about two-thirds xwOBA. Hitters moving to better hitting parks are under-forecast by about 11 points of wOBA without a park adjustment.

## 1. How much should this season count?

The model treats a hitter's track record (their preseason projection) as if it were a set of extra plate appearances, and blends it with this season's plate appearances according to how many of each there are. It learns from past seasons how many plate appearances the track record is worth for each stat. For an established regular with **~200 plate appearances by June 1** (the median), that gives this season the following share of the forecast:

| Stat | Track record worth | This season's share on June 1 |
|---|---|---|
| K% | ~335 PA | **37%** |
| wOBA | ~600 PA | **23%** |
| BB% | ~700 PA | **22%** |

**Example:** a hitter projected for a .340 wOBA has a .400 wOBA through June 1. The forecast is 23% × .400 + 77% × .340 ≈ **.354**: the hot start moves it only about a quarter of the way. A hitter projected for a 20% K rate who is striking out 28% of the time gets 37% × 28% + 63% × 20% ≈ **23%**, more than a third of the way, because K% reflects a hitter's skill faster than wOBA, which depends heavily on whether balls in play find holes.

For a hitter with no MLB history, the track record is worth far less (about 250 PA for wOBA), so this season counts for about 42%. **K% is the stat to believe first:** with 300+ PA, a hitter's season-to-date K% beats their preseason projection (RMSE .0412 vs .0447). wOBA and BB% never get there.

## 2. Slumps and hot starts

On each date, for each stat, the model lists the 10 hitters most likely to beat their season-to-date number from here ("slumps to ignore") and the 10 most likely to fall short ("hot starts to doubt"). For K%, beating it means striking out less. Naive lists are simply the 10 worst and the 10 best hitters so far.

All numbers below are the share that moved as called, pooled over 8 seasons (80 hitters per list and date), for hitters with 75+ PA so far.

**All three stats, averaged over the five forecast dates:**

| | wOBA | K% | BB% |
|---|---|---|---|
| Slumps that improved: model | 95% | 88% | 92% |
| Slumps that improved: naive | 90% | 80% | 87% |
| Hot starts that declined: model | 94% | 89% | 94% |
| Hot starts that declined: naive | 85% | 74% | 86% |

**One stat in detail, wOBA by date:**

| wOBA | May 1 | Jun 1 | Jul 1 | Aug 1 | Sep 1 |
|---|---|---|---|---|---|
| Slumps that improved: model | 100% | 100% | 98% | 94% | 82% |
| Slumps that improved: naive | 92% | 96% | 91% | 89% | 81% |
| Hot starts that declined: model | 100% | 99% | 95% | 90% | **84%** |
| Hot starts that declined: naive | 95% | 90% | 91% | 84% | **66%** |

- **For every stat and every date, the model's lists match or beat the naive lists.**
- **For wOBA and BB%, regression to the mean does most of the work on slumps.** Bad starts bounce back whether or not a model picks them. The naive wOBA list's misses on May 1 were hitters who really were that bad, and the model avoided all six of them.
- **K% slumps are different.** A high-strikeout start is more often real, so the naive list does worse (66% on September 1) and the model's edge is larger (78%).
- **Hot starts are where the model adds the most,** with edges of 15–19 points from July on: wOBA 84% vs 66% on September 1, K% 91% vs 72% on July 1, and BB% 85% vs 70% on September 1. A hot streak from a hitter with a weaker track record and lucky contact is the classic case to doubt.

## 3. Has the outlook really changed?

The harder question is whether a hitter will beat their **preseason projection** from here: is the breakout real? The model's "outlooks raised" list does better than taking hot starts at face value at every date:

| wOBA, outlooks raised that beat preseason | May 1 | Jun 1 | Jul 1 | Aug 1 | Sep 1 |
|---|---|---|---|---|---|
| Model | 59% | 65% | 74% | 74% | 68% |
| Naive (biggest gain over projection so far) | 51% | 59% | 66% | 56% | 61% |

For **lowered** outlooks the model shows no consistent edge: 70–75%, against 61–81% for the naive list. The model's stated chances are also **optimistic in this view**: wOBA hitters given 66% beat their projection 61% of the time. **Treat a believed breakout as a lean, not a lock.**

## 4. Contact quality, playing time and parks

- **xwOBA over wOBA.** The model's learned blend puts 56–71% of the weight on xwOBA. Hitting the ball hard without results is a strong sign that a slump will end.
- **Playing time is information.** Hitters with little MLB history or a small share of plate appearances are below average as a group. Without accounting for this, forecasts for hitters with fewer than 50 PA ran about 10 points of wOBA too high. Teams give plate appearances to the hitters they believe in.
- **Adjust for parks when a hitter changes parks.** For hitters whose parks barely changed, park adjustment makes no difference. For those whose parks changed:

| Hitters whose parks… | Hitters | wOBA bias without parks | With parks |
|---|---|---|---|
| got better for hitting (≥ +3%) | 201 | −.011 | −.006 |
| got tougher (≤ −3%) | 190 | +.006 | +.000 |
| stayed similar (within 1%) | 2,930 | +.000 | +.000 |

## Caveats

- These results are for **hitters who kept playing** (50+ PA left). Injuries and demotions aren't forecast.
- **K% forecasts ran high in 2021–2023,** by up to 1.6 pts in 2021.
- The lists are **pooled over 8 seasons:** 80 picks per cell is enough to show a pattern, not to rank individual months precisely.

Reproduce every number here with `make reports`. The tables are in [`reports/`](../reports).
