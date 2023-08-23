import argparse
import os
import sys
from dotenv import load_dotenv

from src.podcast_backend import save_podcast_output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Podcast processing script")
    
    parser.add_argument(
        "--run_backend",
        help="Process podcast and save output",
        action="store_true"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("No arguments provided. Please add arguments.")
    else:
        load_dotenv()
        print("Loading OpenAI API key...")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if args.run_backend:
            rss_url = "https://tenminutepodcast.libsyn.com/rss"
            local_path = 'podcasts/'
            save_podcast_output(rss_url, local_path, openai_api_key)


# rss_url = "https://tenminutepodcast.libsyn.com/rss"
# local_path = '/content/drive/My Drive/streamlit/'
# # Conan Needs a Friend
# rss_url = 'https://feeds.simplecast.com/dHoohVNH'

# Unexplainable
# rss_url = 'https://feeds.megaphone.fm/VMP9331026707'