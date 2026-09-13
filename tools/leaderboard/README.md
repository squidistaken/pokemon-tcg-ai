# Elo progression figure

Plots the Kaggle rating of every agent we submitted, against the field it plays in.

```bash
uv run python tools/leaderboard/elo_progression.py
uv run python tools/leaderboard/elo_progression.py --output docs/elo.pdf
uv run python tools/leaderboard/elo_progression.py --dump-json outputs/elo.json --open
```

Every run calls the Kaggle API again, so a rerun is how you refresh the numbers.

| Flag | Meaning |
| --- | --- |
| `--competition SLUG` | Competition to read; defaults to `pokemon-tcg-ai-battle`. |
| `--output PATH` | Image to write; the suffix picks the format (`.png`, `.pdf`, `.svg`). Defaults to `outputs/leaderboard/elo_progression.png`. |
| `--dpi N` | Resolution for bitmap formats; defaults to 200. |
| `--style NAME` | Seaborn axes style; defaults to `whitegrid`. |
| `--context NAME` | Seaborn context, which scales every font and line: `paper`, `notebook`, `talk` (default), `poster`. |
| `--figsize WxH` | Figure size in inches; defaults to `14x8`. |
| `--dump-json PATH` | Also write the raw payload, for a diff or a different plot. |
| `--team-name NAME` | Match our team by name when the account lookup finds nothing. |
| `--no-leaderboard` | Skip the leaderboard download; the figure then has no reference lines. |
| `--open` | Open the figure after writing it. |

What the figure shows: the rating of each submission over its upload date, the best rating reached
so far as a step line with a light wash under it, and the current field #1 and field median as
dashed reference lines. Submissions that raised the bar carry a larger orange marker, so the runs
of failed experiments between two records read at a glance. The peak is annotated with its agent
name; the subtitle carries the team rank and the fetch time.

Two things the figure cannot show, because Kaggle does not publish them:

- The score is the rating the agent holds **now**, not the rating it had on its submission date.
  Simulation competitions keep replaying active agents, so points move between two runs of this
  tool. Only the date is fixed.
- The leaderboard part is a snapshot at fetch time. The reference lines are today's field, drawn
  across the whole history.

Authentication comes from the usual Kaggle places: `KAGGLE_USERNAME` / `KAGGLE_KEY`, or
`~/.kaggle/credentials.json` after `kaggle auth login`. The account name also decides which
leaderboard row counts as ours.

## Layout

| File | Contents |
| --- | --- |
| `elo_progression.py` | Command line entry point: fetch, plot, write. |
| `kaggle_client.py` | `KaggleCompetitionClient`, the two API calls and their CSV parsing. |
| `submission_record.py` | `SubmissionRecord`, one submission. |
| `leaderboard_row.py` | `LeaderboardRow`, one team on the public leaderboard. |
| `elo_history.py` | `EloHistory`, the running best, the field summary and the JSON payload. |
| `elo_figure.py` | `EloProgressionFigure`, the seaborn plot. |
