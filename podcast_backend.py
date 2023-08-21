import feedparser
import requests
from pathlib import Path
import pickle
import openai
import json
import wikipedia 

# Load the Whisper model
import whisper
whisper._download(whisper._MODELS["medium"], '/content/podcast/', False)

# Sanitize and shorten the titles for use in filenames and directories
def sanitize_shorten_filename(filename, max_length = 15):
        """Sanitize the filename to remove invalid characters."""
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Truncate episode title if it's too long
        if len(filename) > max_length:
            filename = filename[:max_length]
        return filename

def count_tokens(text, model_name="gpt-3.5-turbo"):
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def trim_text(text, max_tokens=14000):
    tokens = text.split()  # Splitting by whitespace for simplicity
    if len(tokens) > max_tokens:
        print(f"transcription length too long, trimming text to length: {max_tokens}")
        return text
    return ' '.join(tokens[:max_tokens])

def get_transcribe_podcast(rss_url, local_path):
    import feedparser
    import requests
    from pathlib import Path
    import pickle
    
    print("Feed URL: ", rss_url)
    print("Local Path:", local_path)

    # Read from the RSS Feed URL
    intelligence_feed = feedparser.parse(rss_url)
    
    if not intelligence_feed.entries:
        raise ValueError("The RSS feed doesn't have any entries.")
    
    print("Extracting Podcast Information and downloading latest episode")
    # Safely extract podcast and episode titles
    podcast_title = intelligence_feed['feed'].get('title', 'Unknown Podcast')
    episode_title = intelligence_feed.entries[0].get('title', 'Unknown Episode')

    sanitized_podcast_title = sanitize_shorten_filename(podcast_title)
    sanitized_episode_title = sanitize_shorten_filename(episode_title)

    episode_name = f"{sanitized_podcast_title}_{sanitized_episode_title}.mp3"

    # Safely extract episode image
    episode_image = intelligence_feed['feed'].get('image', {}).get('href', None)

    # Safely extract episode URL
    episode_url = None
    for item in intelligence_feed.entries[0].get('links', []):
        if item.get('type') == 'audio/mpeg':
            episode_url = item.get('href')
            break

    # If no audio/mpeg type link is found, try to find any link with '.mp3' in the URL
    if not episode_url:
        for item in intelligence_feed.entries[0].get('links', []):
            if '.mp3' in item.get('href', ''):
                episode_url = item.get('href')
                break

    if not episode_url:
        raise ValueError("No suitable audio link found in the RSS feed.")

   # Download the podcast episode by parsing the RSS feed
    p = Path(local_path).joinpath(sanitized_podcast_title, sanitized_episode_title)
    p.mkdir(parents=True, exist_ok=True)

    print("RSS URL read and episode URL: ", episode_url)
    
    print("Downloading the podcast episode")
    with requests.get(episode_url, stream=True) as r:
        r.raise_for_status()
        episode_path = p.joinpath(episode_name)
        with open(episode_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Podcast Episode downloaded")

    # Load the Whisper model
    import whisper

    # Load model from saved location
    print("Load the Whisper model")
    model = whisper.load_model('medium', device='cuda', download_root='/content/podcast/')

    # Perform the transcription
    print("Starting podcast transcription")
    result = model.transcribe(str(episode_path))

   # Return the transcribed text along with the entire feed and first episode details
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


def extract_information_from_podcast(transcript, prompt):
    """Extract information from the podcast transcript based on the given prompt."""
    import openai
    import tiktoken 
    request = prompt + transcript

    # Choose the model based on token count
    total_tokens = count_tokens(request)
    if total_tokens <= 4096:
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-3.5-turbo-16k"

    try:
        chatOutput = openai.ChatCompletion.create(model=model_name,
                                                  messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                            {"role": "user", "content": request}
                                                            ]
                                                  )
        return chatOutput.choices[0].message.content
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None
    

def get_podcast_summary(podcast_transcript):
    
    summaryPrompt = """
    Please provide a summary suitable for an email newsletter in the form of bullet points.
    The reader should be able to go through it in 30 seconds or less.
    Highlight the main points, any surprising or unexpected information, and conclude with an inspiring takeaway from the podcast.
    Format as follows:
    - TL;DR: [Brief overall summary]
    - Bullet Point 1: [Main point]
    - Bullet Point 2: [Another main point]
    - Bullet Point 3: [Another main point]
    - Surprising Fact: [Unexpected information]
    - Inspiring Takeaway: [Inspiring point from the podcast]
    """
    podcastSummary = extract_information_from_podcast(podcast_transcript, summaryPrompt)

    return podcastSummary

def get_single_subject(podcast_summary):
    single_subjectPrompt = """
    Please extract one word or acronym of the most important subject or idea discussed from this summary of a podcast.
    Only return one word.
    """
    single_subject = extract_information_from_podcast(podcast_summary, single_subjectPrompt)

    return single_subject

def get_podcast_guest(podcast_transcript):
    import openai
    import json
    
    """Extract the guest's name from the podcast transcript using the OpenAI API."""
    
    request = podcast_transcript[:5000]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request}],
        functions=[
            {
                "name": "get_podcast_guest_information",
                "description": "Extract the name of the guest from the provided podcast transcript. The guest is typically introduced in the beginning of the podcast.",
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

    # Extracting the guest's name from the API response
    response_message = completion["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        return function_args.get("guest_name")
    if not response_message.get("function_call"):
        raise ValueError("Failed to extract guest information from the podcast transcript.")

    return None


def get_podcast_highlights(podcast_transcript):
    import openai
    chapters_prompt = """
    Divide the following podcast transcript into chapters and provide a title or theme for each chapter:
    """
    podcastHighlights = extract_information_from_podcast(podcast_transcript, chapters_prompt)
    return podcastHighlights


def extract_detailed_guest_info(transcript):
    """Extract detailed information about the podcast guest using the OpenAI API."""
    request = transcript[:10000]
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print("Number of tokens in input prompt:", len(enc.encode(request)))

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request}],
        functions=[
            {
                "name": "get_podcast_guest_information",
                "description": "Get information on the podcast guest using their full name and the name of the organization they are part of to search for them on Wikipedia or Google",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guest_name": {
                            "type": "string",
                            "description": "The full name of the guest who is speaking in the podcast",
                        },
                        "guest_organization": {
                            "type": "string",
                            "description": "The full name of the organization that the podcast guest belongs to or runs",
                        },
                        "guest_title": {
                            "type": "string",
                            "description": "The title, designation or role of the podcast guest in their organization",
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

def summarize_guest_info(detailed_info):
    """Summarize the detailed guest information into no more than 3 sentences using the OpenAI API."""
    prompt = f"Summarize the following information about the podcast guest in no more than 3 sentences: {detailed_info}"
    
    summarized_info = extract_information_from_podcast(detailed_info, prompt)
    return summarized_info.strip()

def get_detailed_podcast_guest_info(transcript):
    """Get detailed information about the podcast guest from Wikipedia and then summarize it."""
    guest_name, guest_org, guest_title = extract_detailed_guest_info(transcript)
    print("Guest Name:", guest_name)
    print("Guest Organization:", guest_org)
    print("Guest Title:", guest_title)

    # Querying Wikipedia for more information about the guest
    try:
        print(f"Searching wikipedia for {guest_name},  {guest_org},  {guest_title}")
        detailed_info = wikipedia.page(guest_name + " " + guest_org + " " + guest_title, auto_suggest=True).summary
    except:
        print(f"Couldn't fetch information for {guest_name} from Wikipedia.")
        detailed_info = None

    if detailed_info:
        summarized_info = summarize_guest_info(detailed_info)
        return summarized_info
    else:
        return None


def generate_dalle_image(prompt):
    """Generate an image using DALL-E based on the given prompt."""
    import openai
    import requests
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


import random

def generate_podcast_image(podcast_summary, podcast_title, episode_title, single_subject):
    """Generate an image using DALL-E based on the podcast summary and titles."""
    
    def is_valid_image_response(response):
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

    try:
        prompt = (f"An image of someone {activity} in the style of {style} while deeply engrossed in listening to the podcast titled '{podcast_title}' "
          f"The background should look like {single_subject}.")
        image_data = generate_dalle_image(prompt)
        if not is_valid_image_response(image_data):
            raise ValueError("Invalid DALL-E response")
        return image_data
    except Exception as e:
        print(f"Error with detailed prompt: {e}")
        print("Trying with a simplified prompt...")
        prompt = (f"An image of someone {activity} in the style of {style} while deeply engrossed in listening to the podcast titled '{podcast_title}' "
                  f"Use the podcast title  as inspiration for the background.")
        return generate_dalle_image(prompt)

def process_podcast(url, path):
    output = {}
    image_data = None  # Initialize here

    try:
        podcast_details = get_transcribe_podcast(url, path)  
        print("Extracting information from podcast using GPT")

        trimmed_transcript = trim_text(podcast_details['transcribed_data']['episode_transcript'])
        podcast_summary = get_podcast_summary(trimmed_transcript)
        podcast_single_subject = get_single_subject(podcast_summary)

        podcast_guest = get_podcast_guest(trimmed_transcript)  
        podcast_highlights = get_podcast_highlights(trimmed_transcript)  

        podcast_title = podcast_details['transcribed_data'].get('podcast_title', 'Unknown Podcast')
        episode_title = podcast_details['transcribed_data'].get('episode_title', 'Unknown Episode')

        # Extracting detailed guest information
        detailed_guest_info = get_detailed_podcast_guest_info(trimmed_transcript)
        if detailed_guest_info:
            output['detailed_guest_info'] = detailed_guest_info

        image_data = generate_podcast_image(podcast_summary, podcast_title, episode_title, podcast_single_subject)


        output['podcast_details'] = podcast_details
        output['transcribed_data'] = podcast_details['transcribed_data']
        output['podcast_summary'] = podcast_summary
        output['podcast_guest'] = podcast_guest
        output['podcast_highlights'] = podcast_highlights

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}, None
    
    return output, image_data


def save_podcast_output(rss_url, local_path):
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
