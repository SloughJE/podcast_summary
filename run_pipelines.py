import sys 
#import yaml
import argparse

from podcast_backend import save_podcast_output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_backend",
        help="process podcast and save output",
        action="store_true"
    )
     
    parser.add_argument(
        "--make_features",
        help="make features",
        action="store_true"
    )
    # etc

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("No arguments, please add arguments")
    else:
        #with open("params.yaml") as f:
        #    params = yaml.safe_load(f)

        if args.run_backend:
            rss_url = "https://tenminutepodcast.libsyn.com/rss"
            local_path = 'test/'
            save_podcast_output(
                rss_url,
                local_path
            )

# rss_url = "https://tenminutepodcast.libsyn.com/rss"
# local_path = '/content/drive/My Drive/streamlit/'
# # Conan Needs a Friend
# rss_url = 'https://feeds.simplecast.com/dHoohVNH'

# Unexplainable
# rss_url = 'https://feeds.megaphone.fm/VMP9331026707'