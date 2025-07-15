"""Automatically post to Reddit when new README graphics are made."""

import logging
import shutil
import time

import praw

from f1_visualization.consts import ROOT_PATH

logging.basicConfig(level=logging.INFO, format="%(filename)s\t%(levelname)s\t%(message)s")
logger = logging.getLogger(__name__)


VISUALS_PATH = ROOT_PATH / "Docs" / "visuals"
COMMENTS_PATH = ROOT_PATH / "Comments"


def main() -> None:
    """Submit posts and make one comment."""
    reddit = praw.Reddit("armchair-strategist")
    r_formula1 = reddit.subreddit("formula1")
    r_f1technical = reddit.subreddit("f1technical")
    formula1_flairs = r_formula1.flair.link_templates.user_selectable()
    f1technical_flairs = r_f1technical.flair.link_templates.user_selectable()
    formula1_flair_id = next(
        flair for flair in formula1_flairs if "Statistics" in flair["flair_text"]
    )["flair_template_id"]
    f1technical_flair_id = next(
        flair for flair in f1technical_flairs if "Strategy" in flair["flair_text"]
    )["flair_template_id"]

    with open(ROOT_PATH / "tmp" / "event_name.txt", "r", encoding="utf-8") as fin:
        event_name = fin.read().strip()

    dashboard_link = "Check out more at armchair-strategist.dev!"
    images = [
        {
            "image_path": VISUALS_PATH / "strategy.png",
            "caption": f"Tyre strategy recap. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "position.png",
            "caption": f"Race position history. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "laptime.png",
            "caption": f"Point finishers' lap times. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "fuel_adjusted.png",
            "caption": f"Point finishers' fuel-adjusted lap times. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "podium_gap.png",
            "caption": f"Podium finishers' gaps to winners. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "team_pace.png",
            "caption": f"Team pace ranking. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "teammate_violin.png",
            "caption": (
                "Driver pace ranking (teammates vs teammates). Largest gap on the left. "
                f"{dashboard_link}"
            ),
        },
        {
            "image_path": VISUALS_PATH / "driver_pace.png",
            "caption": (
                "Driver pace ranking (finishing order). Highest finisher on the left. "
                f"{dashboard_link}"
            ),
        },
    ]

    formula1_post = r_formula1.submit_gallery(
        title=f"{event_name} Strategy & Performance Recap",
        images=images,
        flair_id=formula1_flair_id,
    )
    with open(COMMENTS_PATH / "formula1_comment.md", "r", encoding="utf-8") as fin:
        formula1_post.reply(fin.read())

    logger.info("Finished posting to r/formula1")

    time.sleep(5)

    f1technical_post = r_f1technical.submit_gallery(
        title=f"{event_name} Strategy & Performance Recap",
        images=images,
        flair_id=f1technical_flair_id,
    )
    with open(COMMENTS_PATH / "f1technical_comment.md", "r", encoding="utf-8") as fin:
        f1technical_post.reply(fin.read())
    logger.info("Finished posting to r/f1technical")

    # clean up temp directory
    shutil.rmtree(ROOT_PATH / "tmp")


if __name__ == "__main__":
    main()
