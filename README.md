Check out the app at https://podcastsummary-jslough.streamlit.app/

# Podcast Newsletter Processing and Summary Tool

This tool processes podcast episodes, transcribes them, extracts important details like guest information, generates summaries, highlights, and even creates an image inspired by the podcast content using DALL-E.

The results can be visualized using the associated Streamlit app.

## Usage

You can run the backend processing tool using the `run_pipelines.py` script. Note the filepath in `run_pipelines.py`

### Command-line Arguments:

- `--run_backend`: Process the podcast and save output.

### Examples:

To process a podcast and save its output:
```bash
python run_pipelines.py --run_backend
```

## Streamlit App

You can visualize the processed results and summaries using the Streamlit app.

[https://podcastsummary-jslough.streamlit.app/](https://podcastsummary-jslough.streamlit.app/)

or to view locally on processed podcasts:

```bash
streamlit run src/podcast_frontend.py
```