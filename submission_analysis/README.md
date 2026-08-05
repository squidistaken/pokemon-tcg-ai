# Submission Analysis

Kaggle submission-status tooling for the `pokemon-tcg-ai-battle` competition:
live submission status, played-episode outcomes (with an optional replay
download), and a deck-refinement report built from those downloaded replays.
Mirrors `scraper/analysis`'s package shape (`console.py` / `reporting.py` /
`plots.py`, `outputs/<name>/` for generated artifacts) so both read as one
family.

## Usage

```bash
# Live status for this competition's submissions, filtered/sorted.
# --log-history appends a row per successful submission to
# logs/kaggle_rating_history.csv. Also fetches our team's current
# leaderboard rank by default (--no-rank to skip).
uv run python -m submission_analysis status --most-recent-n 2 --log-history

# Win/loss/draw outcomes by opponent for a submission's played episodes.
# --download-replays additionally saves each episode's full replay JSON
# under logs/replays/<submission ref>/.
uv run python -m submission_analysis episodes --download-replays --most-recent-n 2

# Deck-refinement report (card/attack/loss-cause stats + plots) built from
# downloaded replays. One report per submission - see "Output layout" below.
uv run python -m submission_analysis deck-report --deck decks/example.csv

# Scout the top-8 leaderboard teams' decks via their public replays, and
# diff against our own deck. Looks outward (the competition), unlike
# deck-report which only ever looks inward at our own deck.
uv run python -m submission_analysis scout --deck decks/example.csv
```

Each subcommand also has its own `main(argv) -> int`; `__main__.py` is a
thin router over `submissions.py` / `episodes.py` / `scout.py` / itself, not
a reimplementation.

## Output layout

```
outputs/submission_analysis/
  rating_history.png                       # Kaggle live rating over time, all submissions
  leaderboard_rank_over_time.png           # our team's live leaderboard rank over time (lower = better)
  win_rate_by_submission.png               # our local win rate, across analyzed submissions
  average_ko_margin_by_submission.png
  average_first_attack_turn_by_submission.png
  <submission ref>[-<message slug>]/       # one folder per analyzed submission
    report_<timestamp>.txt                 # opens with a "worth looking into" section
    card_play_rate.png
    win_rate_when_played.png
    ko_rate.png
    attack_utilization.png                 # played but never/rarely attacked with (bench-warmers)
    play_rate_vs_win_rate.png              # scatter: combines play rate + win rate, one point/card
    ko_rate_vs_win_rate.png                # scatter: combines KO rate + win rate, one point/card
    attack_usage.png
    damage_by_card.png                     # total damage, summed across each card's attacks
    opponent_attack_usage.png              # threat intel: enemy attacks/cards responsible for our KOs
    evolution_conversion_rate.png
    loss_causes.png
    first_attack_turn_distribution.png
    ko_margin_distribution.png
    game_length_distribution.png           # final turn reached, by result - fast blowouts vs long grinds
  scout/                                   # written by `scout`, not `deck-report`
    <deck CSV stem>/                       # one folder per --deck, e.g. "ns-zoroark" - gap_analysis.png
      report_<timestamp>.txt               # is a diff against that specific deck, so it can't be shared
      card_frequency.png                   # how common each card is across the scouted teams
      gap_analysis.png                     # cards they run that our own --deck doesn't
    no-deck/                               # used when --deck "" disables gap analysis entirely

logs/
  replays/<submission ref>/               # downloaded replay JSON + manifest.json
  scouted_replays/<team id>/              # other teams' replays, from `scout` (kept separate from ours)
  kaggle_rating_history.csv               # ref/label/status/score snapshots (status --log-history)
```
