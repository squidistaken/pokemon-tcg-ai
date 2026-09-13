"""
Draw the Elo progression as a seaborn figure.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INK = "#1f2328"
SUBTLE_INK = "#5c6169"
FAINT_INK = "#9aa0a6"
BLUE = "#3d6fb4"
ORANGE = "#e07b39"


class EloProgressionFigure:
    """
    Plot the rating of every submitted agent against the field it plays in.

    One axis, two series: the rating each submission holds now, and the best
    rating reached up to that date as a step line with a light wash under it.
    Submissions that raised the bar carry a larger marker, so the run of failed
    experiments between two records is visible without reading the table. The
    current leader and the field median are dashed reference lines, so a point
    can be read against the ladder and not only against our own past.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        style: str = "whitegrid",
        context: str = "talk",
        figsize: tuple[float, float] = (14.0, 8.0),
    ) -> None:
        """
        :param payload: Output of `EloHistory.to_payload`.
        :param style: Seaborn axes style, for example `whitegrid` or `ticks`.
        :param context: Seaborn context, which scales every font and line: `paper`,
            `notebook`, `talk` or `poster`.
        :param figsize: Figure size in inches.
        """
        self.payload = payload
        self.style = style
        self.context = context
        self.figsize = figsize

    @staticmethod
    def _pyplot():
        """
        Import matplotlib with the non-interactive `Agg` backend.

        :return: The `matplotlib.pyplot` module.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _title(self) -> str:
        """
        Build the figure title from the team name the leaderboard reports.
        """
        team = self.payload["field"].get("team_name")
        return (
            f"{team}: Kaggle Elo by submission" if team else "Kaggle Elo by submission"
        )

    def _subtitle(self) -> str:
        """
        Build the line under the title: the field context and the fetch time.
        """
        field = self.payload["field"]
        parts = [f"{len(self.payload['points'])} scored submissions"]
        if "team_rank" in field:
            parts.append(f"rank #{field['team_rank']} of {field['team_count']} teams")
        parts.append(f"fetched {self.payload['fetched_at']}")
        return " · ".join(parts)

    def render(self, output: Path, dpi: int = 200) -> Path:
        """
        Draw the figure and write it to disk.

        The file format follows the suffix of the output path, so `.png`, `.pdf`
        and `.svg` all work.

        :param output: Image path to write.
        :param dpi: Raster resolution for bitmap formats.
        :return: The path written.
        """
        import matplotlib.dates as mdates
        import seaborn as sns
        from matplotlib import transforms

        plt = self._pyplot()
        points = self.payload["points"]
        if not points:
            raise RuntimeError("no scored submissions to plot")

        dates = [
            datetime.strptime(point["date"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            for point in points
        ]
        scores = [point["score"] for point in points]
        best = [point["best"] for point in points]
        records = [
            index
            for index, point in enumerate(points)
            if point["score"] >= point["best"]
        ]
        field = self.payload["field"]

        sns.set_theme(style=self.style, context=self.context)
        figure, axes = plt.subplots(figsize=self.figsize)
        figure.patch.set_facecolor("white")
        axes.set_facecolor("white")

        axes.plot(
            dates,
            best,
            drawstyle="steps-post",
            color=ORANGE,
            linewidth=3.0,
            solid_capstyle="round",
            label="Best so far",
            zorder=2,
        )
        axes.plot(
            dates,
            scores,
            color=BLUE,
            linewidth=2.4,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.6,
            solid_capstyle="round",
            label="Submission Elo",
            zorder=3,
        )
        axes.plot(
            [dates[index] for index in records],
            [scores[index] for index in records],
            linestyle="none",
            marker="o",
            markersize=13,
            markerfacecolor=ORANGE,
            markeredgecolor="white",
            markeredgewidth=2.0,
            label="Raised the bar",
            zorder=4,
        )

        references = [
            (field[key], label)
            for key, label in (
                ("top_score", "field #1"),
                ("median_score", "field median"),
            )
            if field.get(key) is not None
        ]
        low = min(scores)
        high = max(scores + [value for value, _ in references])
        axes.set_ylim(low - 0.10 * (high - low), high + 0.12 * (high - low))
        axes.fill_between(
            dates,
            best,
            axes.get_ylim()[0],
            step="post",
            color=ORANGE,
            alpha=0.06,
            linewidth=0,
            zorder=0,
        )

        right_edge = transforms.blended_transform_factory(
            axes.transAxes, axes.transData
        )
        for value, label in references:
            axes.axhline(
                value, color=FAINT_INK, linestyle=(0, (6, 5)), linewidth=1.4, zorder=1
            )
            axes.text(
                0.995,
                value,
                f"{label} {value:.0f}",
                transform=right_edge,
                color=SUBTLE_INK,
                fontsize=13,
                va="center",
                ha="right",
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3.0},
            )

        peak = max(range(len(points)), key=lambda index: scores[index])
        axes.annotate(
            f"{scores[peak]:.0f}  {points[peak]['label']}",
            xy=(dates[peak], scores[peak]),
            xytext=(-16, 30),
            textcoords="offset points",
            ha="right",
            fontsize=14,
            fontweight="medium",
            color=INK,
            arrowprops={"arrowstyle": "-", "color": FAINT_INK, "linewidth": 1.2},
        )
        if peak != len(points) - 1:
            axes.annotate(
                f"{scores[-1]:.0f}",
                xy=(dates[-1], scores[-1]),
                xytext=(0, -26),
                textcoords="offset points",
                ha="center",
                fontsize=13,
                color=SUBTLE_INK,
            )

        axes.set_ylabel("Elo", fontsize=15, color=SUBTLE_INK, labelpad=12)
        axes.set_xlabel("")
        axes.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes.tick_params(axis="both", labelsize=13, colors=SUBTLE_INK, length=0)
        axes.margins(x=0.03)
        axes.set_xlim(right=dates[-1] + (dates[-1] - dates[0]) * 0.14)
        axes.grid(axis="y", color="#e6e8eb", linewidth=1.1)
        axes.grid(axis="x", visible=False)
        axes.set_axisbelow(True)
        for spine in axes.spines.values():
            spine.set_visible(False)

        legend = axes.legend(
            loc="lower right", frameon=False, fontsize=14, handlelength=1.8
        )
        for entry in legend.get_texts():
            entry.set_color(SUBTLE_INK)

        figure.tight_layout(rect=(0, 0, 1, 0.88))
        left = axes.get_position().x0
        figure.text(
            left,
            0.965,
            self._title(),
            fontsize=23,
            fontweight="semibold",
            color=INK,
            ha="left",
            va="top",
        )
        figure.text(
            left,
            0.905,
            self._subtitle(),
            fontsize=14,
            color=FAINT_INK,
            ha="left",
            va="top",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
        return output
