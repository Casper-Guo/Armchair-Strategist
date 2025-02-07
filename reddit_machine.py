"""Automatically post to Reddit when new README graphics are made."""

import logging
import shutil
import time

import praw

from f1_visualization._consts import ROOT_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(filename)s\t%(message)s")
logger = logging.getLogger(__name__)


VISUALS_PATH = ROOT_PATH / "Docs" / "visuals"


def main():
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

    with open(ROOT_PATH / "tmp" / "event_name.txt", "r") as fin:
        event_name = fin.read().strip()

    dashboard_link = "Check out more at armchair-strategist.dev!"
    images = [
        {
            "image_path": VISUALS_PATH / "strategy.png",
            "caption": (
                "Tyre strategy recap. Stripped bar sections represent used tyre stints. "
                f"{dashboard_link}"
            ),
        },
        {
            "image_path": VISUALS_PATH / "podium_gap.png",
            "caption": f"Podium finishers' gaps to winners. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "position.png",
            "caption": f"Race position history. {dashboard_link}",
        },
        {
            "image_path": VISUALS_PATH / "laptime.png",
            "caption": (
                "Point finishers' lap times. White vertical bars represent pitstops. "
                f"{dashboard_link}"
            ),
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
    formula1_post.reply(
        (
            "What other graphics do you want to see and "
            "how can these existing graphics be improved, quesion."
        )
    )
    logger.info("Finished posting to r/formula1")

    time.sleep(5)

    f1technical_post = r_f1technical.submit_gallery(
        title=f"{event_name} Strategy & Performance Recap",
        images=images,
        flair_id=f1technical_flair_id,
    )
    f1technical_post.reply(
        (
            "Check out the interactive version of these graphics and more "
            "at my [strategy dashboard](https://armchair-strategist.dev/)"
            "\n\n"
            "Please let me know if you have suggestions for improving these graphics "
            "or ideas for other graphics!"
        )
    )
    logger.info("Finished posting to r/f1technical")

    # clean up temp directory
    shutil.rmtree(ROOT_PATH / "tmp")


if __name__ == "__main__":
    main()
