import json
import random
import time
from pathlib import Path
from typing import Tuple, Dict, Union, Optional, Any, List

import feedparser
import openai
import requests
import tiktoken
import whisper
import wikipedia


def sanitize_shorten_filename(filename: str, max_length: int = 15) -> str:
    """
    Sanitize the filename to ensure only alphanumeric characters are present 
    and truncate if its length exceeds a specified max length.

    Args:
    - filename (str): Original filename to be sanitized.
    - max_length (int, optional): Maximum length for the sanitized filename. Defaults to 15.

    Returns:
    - str: Sanitized filename.
    """
    # Retain only alphanumeric characters
    sanitized_filename = ''.join(char for char in filename if char.isalnum())

    # Truncate episode title if it's too long
    if len(sanitized_filename) > max_length:
        sanitized_filename = sanitized_filename[:max_length]
    
    return sanitized_filename


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text based on a specific model's tokenization.

    Args:
    - text (str): The text whose tokens are to be counted.
    - model_name (str, optional): The name of the model for token counting. Defaults to "gpt-3.5-turbo".

    Returns:
    - int: Number of tokens in the text.
    """
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def trim_text(text: str, max_tokens: int = 12000) -> str:
    """
    Trim the text if it exceeds a certain token count.

    Args:
    - text (str): The original text to trim.
    - max_tokens (int, optional): Maximum number of tokens allowed. Defaults to 12000.

    Returns:
    - str: Trimmed text.
    """
    tokens = text.split()  # Splitting by whitespace
    
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    
    return text


def get_transcribe_podcast(rss_url: str, local_path: str) -> dict:
    """
    Extract podcast details from an RSS URL, download the latest episode, transcribe it 
    and return podcast details along with the transcribed data.

    Args:
    - rss_url (str): The URL of the RSS feed to parse.
    - local_path (str): Path to save the downloaded podcast episode.

    Returns:
    - dict: Details of the podcast and the transcribed data.
    """
    print(f"Feed URL: {rss_url}")
    print(f"Local Path: {local_path}")

    intelligence_feed = feedparser.parse(rss_url)

    if not intelligence_feed.entries:
        raise ValueError("The RSS feed doesn't have any entries.")

    print("Extracting Podcast Information and downloading latest episode")
    podcast_title = intelligence_feed['feed'].get('title', 'Unknown Podcast')
    episode_title = intelligence_feed.entries[0].get('title', 'Unknown Episode')

    sanitized_podcast_title = sanitize_shorten_filename(podcast_title)
    sanitized_episode_title = sanitize_shorten_filename(episode_title)

    episode_name = f"{sanitized_podcast_title}_{sanitized_episode_title}.mp3"
    episode_image = intelligence_feed['feed'].get('image', {}).get('href', None)

    episode_url = next(
        (item.get('href') for item in intelligence_feed.entries[0].get('links', []) if item.get('type') == 'audio/mpeg'),
        None
    )

    if not episode_url:
        episode_url = next(
            (item.get('href') for item in intelligence_feed.entries[0].get('links', []) if '.mp3' in item.get('href', '')),
            None
        )

    if not episode_url:
        raise ValueError("No suitable audio link found in the RSS feed.")

    episode_directory = Path(local_path).joinpath(sanitized_podcast_title, sanitized_episode_title)
    episode_directory.mkdir(parents=True, exist_ok=True)

    print(f"RSS URL read and episode URL: {episode_url}")
    print("Downloading the podcast episode")

    with requests.get(episode_url, stream=True) as response:
        response.raise_for_status()
        episode_path = episode_directory.joinpath(episode_name)
        with open(episode_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    print("Podcast Episode downloaded")
    print("Load the Whisper model")

    model = whisper.load_model("small.en", device='cuda')

    start_time = time.time()
    print("Starting podcast transcription")
    result = model.transcribe(str(episode_path))
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Transcription completed in {int(minutes)} minutes and {int(seconds)} seconds.")

    output = {
        'podcast_details': intelligence_feed['feed'],
        'first_episode': intelligence_feed.entries[0],
        'transcribed_data': {
            'podcast_title': podcast_title,
            'episode_title': episode_title,
            'episode_image': episode_image,
            'episode_transcript': result['text']
        }
    }

    return output


def extract_information_from_podcast(transcript: str, prompt: str) -> str:
    """
    Extract information from the podcast transcript based on the given prompt using an OpenAI model.

    Args:
    - transcript (str): The podcast transcript to extract information from.
    - prompt (str): The prompt to guide the model for extracting information.

    Returns:
    - str: Extracted information, or None if an exception occurred.
    """
    request = prompt + transcript

    # Choose the model based on token count
    total_tokens = count_tokens(request)
    if total_tokens <= 3500:
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-3.5-turbo-16k"

    try:
        chat_output = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request}
            ]
        )
        return chat_output.choices[0].message.content
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None
    

def get_podcast_summary(podcast_transcript: str) -> str:
    """
    Generate a summary for the podcast transcript suitable for an email newsletter.

    Args:
    - podcast_transcript (str): The transcript of the podcast to summarize.

    Returns:
    - str: Summarized content of the podcast in a formatted manner.
    """
    print("Generating summary...")
    
    summary_prompt = (
        "Please provide a summary of this podcast transcript suitable for an email newsletter in the form of bullet points. "
        "Highlight the main points, any surprising or unexpected information, and conclude with an inspiring takeaway from the podcast. "
        "Format as follows:\n"
        "- TL;DR: [Brief overall summary]\n"
        "- Bullet Point 1: [Main point]\n"
        "- Bullet Point 2: [Another main point]\n"
        "- Bullet Point 3: [Another main point]\n"
        "- Surprising Fact: [Unexpected information]\n"
        "- Inspiring Takeaway: [Inspiring point from the podcast]\n\n"
        "The transcript:\n"
    )
    
    podcast_summary = extract_information_from_podcast(podcast_transcript, summary_prompt)
    return podcast_summary


def get_single_subject(podcast_summary: str) -> str:
    """
    Extract the most important subject or idea discussed from the podcast summary.

    Args:
    - podcast_summary (str): Summary of the podcast.

    Returns:
    - str: Most important subject or idea (5 words or less).
    """
    print("Generating single subjects...")

    single_subject_prompt = (
        "Please extract 5 words or less of the most important subject or idea discussed from this summary of a podcast. "
        "Return only 5 words or less."
    )
    
    single_subject = extract_information_from_podcast(podcast_summary, single_subject_prompt)
    return single_subject


def get_podcast_guest(podcast_transcript: str) -> str:
    """
    Extract the guest's name from the podcast transcript using the OpenAI API.

    Args:
    - podcast_transcript (str): The transcript of the podcast.

    Returns:
    - str: Name of the guest if found; None otherwise.
    """
    print("Getting podcast guest...")

    request = podcast_transcript[:5000]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request}],
        functions=[
            {
                "name": "get_podcast_guest_information",
                "description": "Extract the name of the guest from the provided podcast transcript. \
                    The guest is typically introduced in the beginning of the podcast.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guest_name": {
                            "type": "string",
                            "description": "The name of the guest who appeared in the podcast episode.",
                        },
                        "unit": {"type": "string"},
                    },
                    "required": ["guest_name"],
                },
            }
        ],
        function_call={"name": "get_podcast_guest_information"}
    )

    response_message = completion["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        return function_args.get("guest_name")

    raise ValueError("Failed to extract guest information from the podcast transcript.")


def get_podcast_highlights(podcast_transcript: str) -> str:
    """
    Divide the podcast transcript into chapters and provide a title or theme for each.

    Args:
    - podcast_transcript (str): The transcript of the podcast.

    Returns:
    - str: Highlights in the form of chapter titles or themes.
    """
    print("Generating highlights...")

    chapters_prompt = """
    Divide the following podcast transcript into chapters and provide a title or theme for each chapter:
    """
    podcast_highlights = extract_information_from_podcast(podcast_transcript, chapters_prompt)
    
    return podcast_highlights


def extract_detailed_guest_info(transcript: str) -> Tuple[str, str, str]:
    """
    Extract detailed information about the podcast guest using the OpenAI API.

    Args:
    - transcript (str): The transcript of the podcast.

    Returns:
    - Tuple[str, str, str]: A tuple containing guest's name, organization, and title.
    """
    request = transcript[:10000]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request}],
        functions=[
            {
                "name": "get_podcast_guest_information",
                "description": "Get information on the podcast guest using their full name and the name of the \
                                organization they are part of to search for them on Wikipedia or Google.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guest_name": {
                            "type": "string",
                            "description": "The full name of the guest who is speaking in the podcast.",
                        },
                        "guest_organization": {
                            "type": "string",
                            "description": "The full name of the organization that the podcast guest belongs to or runs.",
                        },
                        "guest_title": {
                            "type": "string",
                            "description": "The title, designation, or role of the podcast guest in their organization.",
                        },
                    },
                    "required": ["guest_name"],
                },
            }
        ],
        function_call={"name": "get_podcast_guest_information"}
    )

    # Extracting the detailed information from the API response
    response_message = completion["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        guest_name = function_args.get("guest_name", "")
        guest_org = function_args.get("guest_organization", "")
        guest_title = function_args.get("guest_title", "")
        return guest_name, guest_org, guest_title
    return None, None, None


def summarize_guest_info(detailed_info: str) -> str:
    """
    Summarize the detailed guest information into no more than 3 sentences using the OpenAI API.

    Args:
    - detailed_info (str): Detailed information about the podcast guest.

    Returns:
    - str: Summarized information.
    """
    prompt = f"Summarize the following information about the podcast guest in no more than 3 sentences: {detailed_info}"
    
    summarized_info = extract_information_from_podcast(detailed_info, prompt)
    return summarized_info.strip()


def get_detailed_podcast_guest_info(transcript: str) -> Optional[str]:
    """
    Get detailed information about the podcast guest from Wikipedia and then summarize it.

    Args:
    - transcript (str): Podcast transcript.

    Returns:
    - Optional[str]: Summarized information about the podcast guest or None if information isn't found.
    """
    guest_name, guest_org, guest_title = extract_detailed_guest_info(transcript)
    print("Guest Name:", guest_name)
    print("Guest Organization:", guest_org)
    print("Guest Title:", guest_title)

    # Querying Wikipedia for more information about the guest
    try:
        print(f"Searching wikipedia for {guest_name},  {guest_org},  {guest_title}")
        search_query = f"{guest_name} {guest_org} {guest_title}"
        detailed_info = wikipedia.page(search_query, auto_suggest=True).summary
    except Exception as e:
        print(f"Couldn't fetch information for {guest_name} from Wikipedia due to: {e}")
        detailed_info = None

    if detailed_info:
        summarized_info = summarize_guest_info(detailed_info)
        return summarized_info
    else:
        return None


def generate_dalle_image(prompt: str) -> Optional[bytes]:
    """Generate an image using DALL-E based on the given prompt.
    
    Args:
    - prompt (str): The text prompt based on which DALL-E will generate an image.

    Returns:
    - Optional[bytes]: Image data in bytes if successful, None otherwise.
    """
    print(f"Creating image with prompt: {prompt}")
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        image_data = requests.get(image_url).content
        return image_data
    except Exception as e:
        print(f"An error occurred during DALL-E image generation: {e}")
        return None
    

def generate_podcast_image(podcast_summary: str, podcast_title: str, episode_title: str, single_subject: str) -> Optional[bytes]:
    """Generate an image using DALL-E based on the podcast summary and titles.

    Args:
    - podcast_summary (str): Podcast summary text.
    - podcast_title (str): Podcast title.
    - episode_title (str): Episode title.
    - single_subject (str): Single subject from the podcast.

    Returns:
    - Optional[bytes]: Image data in bytes if successful, None otherwise.
    """
    def is_valid_image_response(response: Union[bytes, None]) -> bool:
        """Check if the response from DALL-E is valid."""
        return isinstance(response, bytes)

    print("Generating image with DALL-E")
    # List of random activities
    activities = [
        'exercising at the gym',
        'working at the office',
        'relaxing on a hammock',
        'cooking in the kitchen',
        'driving in a car',
        'jogging in a park',
        'sipping coffee at a café',
        'walking their dog in the park',
        'painting in a studio',
        'gardening in the backyard',
        'doing yoga on a beach',
        'shopping at a grocery store',
        'hiking up a mountain',
        'fishing by a lakeside',
        'lying on a couch at home',
        'traveling in a subway train',
        'cycling through the countryside',
        'reading a book in a library',
        'sunbathing by a pool',
        'camping in the woods',
        'fixing a car in a garage'
    ]

    styles = [
        "a Renaissance painting",
        "a Picasso artwork",
        "a Japanese ukiyo-e woodblock print",
        "a Van Gogh painting",
        "a 1980s retro poster",
        "a noir film scene",
        "an Art Deco poster",
        "a graffiti mural",
        "a 1960s psychedelic poster",
        "a pop art piece",
        "a Cubist artwork",
        "an Impressionist painting",
        "a Surrealist dream scene",
        "a Gothic art piece",
        "a Baroque painting",
        "a Dadaist collage",
        "an Abstract Expressionist artwork",
        "a Minimalist design",
        "a Futurist scene",
        "a Romantic painting",
        "a Byzantine mosaic",
        "an Art Nouveau poster",
        "a Fauvist artwork",
        "a Neoclassical scene",
        "a Medieval tapestry",
        "a Bauhaus design",
        "a Pointillist painting",
        "a Russian Constructivist poster",
        "a Pre-Raphaelite artwork",
        "an Op art piece"
        "a photo-realistic image"
    ]

    activity = random.choice(activities)
    style = random.choice(styles)

    prompt = (f"An image of someone {activity} in the style of {style} while deeply engrossed in listening to the podcast titled '{podcast_title}' "
              f"The background should be inspired from this phrase: {single_subject}.")
    image_data = generate_dalle_image(prompt)
    
    if is_valid_image_response(image_data):
        return image_data
    else:
        print("Error with detailed prompt. Trying with a simplified prompt...")
        prompt = (f"An image of someone {activity} in the style of {style} while deeply engrossed in listening to the podcast titled '{podcast_title}' "
                  f"Use the podcast title as inspiration for the background.")
        return generate_dalle_image(prompt)
        

def process_podcast(url: str, path: str) -> Tuple[Dict[str, Union[str, Dict[str, Any], List[str]]], Optional[bytes]]:
    """Process the podcast based on its URL and path.
    
    Args:
    - url (str): The podcast URL.
    - path (str): The path to the podcast file or details.
    
    Returns:
    - tuple: A tuple containing the output dictionary and image data.
    """
    output = {}
    image_data = None  # Initialize here

    try:
        # 1. Transcribe and retrieve basic podcast details.
        podcast_details = get_transcribe_podcast(url, path)  
        print("Extracting information from podcast using GPT")

        # 2. Trim and preprocess the transcript.
        trimmed_transcript = trim_text(podcast_details['transcribed_data']['episode_transcript'])

        # 3. Extract key information from the podcast.
        podcast_summary = get_podcast_summary(trimmed_transcript)
        podcast_single_subject = get_single_subject(podcast_summary)
        podcast_guest = get_podcast_guest(trimmed_transcript)
        podcast_highlights = get_podcast_highlights(trimmed_transcript)

        # 4. Get podcast and episode titles.
        podcast_title = podcast_details['transcribed_data'].get('podcast_title', 'Unknown Podcast')
        episode_title = podcast_details['transcribed_data'].get('episode_title', 'Unknown Episode')

        # 5. Generate DALL-E image for the podcast.
        image_data = generate_podcast_image(podcast_summary, podcast_title, episode_title, podcast_single_subject)

        # 6. Consolidate results into the output dictionary.
        output['podcast_details'] = podcast_details
        output['transcribed_data'] = podcast_details['transcribed_data']
        output['podcast_summary'] = podcast_summary
        output['podcast_guest'] = podcast_guest
        output['podcast_highlights'] = podcast_highlights

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}, None
    
    return output, image_data


def save_podcast_output(rss_url: str, local_path: str, openai_api_key: str) -> None:
    """Save the podcast output to local storage based on the RSS URL and local path.
    
    Args:
    - rss_url (str): The RSS URL of the podcast.
    - local_path (str): The local directory path to save the processed data.
    - openai_api_key (str): The API key to authenticate with OpenAI.
    
    Returns:
    - None
    """
    
    # Set the OpenAI API Key
    openai.api_key = openai_api_key

    # Call the process_podcast function directly
    output, image_data = process_podcast(rss_url, local_path)

    # Extract podcast and episode titles from the output
    transcribed_data = output.get('transcribed_data', {})
    podcast_title = transcribed_data.get('podcast_title', 'Unknown Podcast')
    episode_title = transcribed_data.get('episode_title', 'Unknown Episode')

    sanitized_podcast_title = sanitize_shorten_filename(podcast_title)
    sanitized_episode_title = sanitize_shorten_filename(episode_title)

    # Create the directory path and filename for the JSON output
    from pathlib import Path
    p = Path(local_path).joinpath(sanitized_podcast_title, sanitized_episode_title)
    p.mkdir(parents=True, exist_ok=True)

    json_filename = f"{sanitized_podcast_title}_{sanitized_episode_title}.json"
    json_path = p.joinpath(json_filename)

    # Save the output as a JSON file
    import json
    with open(json_path, "w") as outfile:
        json.dump(output, outfile)

    # Save the DALL-E generated image
    if image_data:
        image_filename = f"{sanitized_podcast_title}_{sanitized_episode_title}_image.jpg"
        image_path = p.joinpath(image_filename)
        with open(image_path, 'wb') as img_file:
            img_file.write(image_data)

    print(f"Podcast output saved to {json_path}")
