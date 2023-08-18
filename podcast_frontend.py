import streamlit as st
#import modal
import json
import os

def display_podcast_info(podcast_info):
    # Display the podcast title and episode title
    
    # Check if 'first_episode' exists and then display its title
    if 'first_episode' in podcast_info['podcast_details']:
        st.subheader(podcast_info['podcast_details']['first_episode']['title'])

    display_podcast_summary(podcast_info)
    # Check if there's a guest and display their information
    # NOTE: 'podcast_guest' doesn't exist in the provided JSON. You might want to adjust this.

    display_podcast_guest(podcast_info)

    # Display the podcast highlights/chapters
    display_podcast_highlights(podcast_info)



def display_podcast_summary(podcast_info):
    """Display the podcast episode summary and cover image."""
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Episode Summary")
        st.write(podcast_info['podcast_summary'])
    with col2:
        st.image(podcast_info['podcast_details']['transcribed_data']['episode_image'], caption="Podcast Cover", width=300, use_column_width=True)

def display_podcast_guest(podcast_info):
    """Display the podcast guest and their details."""
    col3, col4 = st.columns([3, 7])
    with col3:
        st.subheader("Podcast Guest")
        # Using get() to provide a default value if 'podcast_guest' doesn't exist
        st.write(podcast_info.get('podcast_guest', 'Guest not available'))
    with col4:
        st.subheader("Summary of Podcast Guest Details")
        # Check if 'detailed_guest_info' exists, if not, provide a default message
        st.write("Guest information is currently not reliable for lesser known guests")
        guest_details = podcast_info.get('detailed_guest_info', 'Details not available.')
        st.write(guest_details)


def display_podcast_highlights(podcast_info):
    """Display the key moments or highlights of the podcast episode."""
    st.subheader("Key Moments")
    for moment in podcast_info['podcast_highlights'].split('\n'):
        st.markdown(f"<p style='margin-bottom: 5px;'>{moment}</p>", unsafe_allow_html=True)

def main():
    st.title("Newsletter Dashboard")

    available_podcast_info = create_dict_from_json_files('./podcasts')

    # Left section - Input fields
    st.sidebar.header("Podcast RSS Feeds")

    # Dropdown box
    st.sidebar.subheader("Available Podcasts Feeds")
    selected_podcast = st.sidebar.selectbox("Select Podcast", options=available_podcast_info.keys())

    if selected_podcast:
        podcast_data = available_podcast_info[selected_podcast]
        podcast_info = podcast_data['info']
        st.title(podcast_info['podcast_details']['podcast_details']['title'])

        st.header("DALL-E Podcast Inspired Image")
        
        # Display the DALL-E generated image
        image_path = podcast_data['image_path']
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="DALL-E Generated Image", width=400)
        else:
            st.write(f"DALL-E Generated Image not found!")

        # Call the comprehensive display function
        display_podcast_info(podcast_info)

    # User Input box
    st.sidebar.subheader("Add and Process New Podcast Feed")
    url = st.sidebar.text_input("Link to RSS Feed")

    process_button = st.sidebar.button("Process Podcast Feed")
    st.sidebar.markdown("**Note**: Podcast processing can take up to 5 mins, please be patient.")

    # Explanatory text
    st.sidebar.markdown("### Explanations:")
    st.sidebar.markdown("- The **top image** is inspired by the podcast TLDR, generated using a random choice of activity and styles.")
    st.sidebar.markdown("- The **TLDR summary** and **Key Moments** are generated using GPT.")
    st.sidebar.markdown("- The **guest info** pulled from Wikipedia is summarized by GPT. This information is not always the actual guest (work in progress).")

    if process_button:
        podcast_info = process_podcast_info(url)
        st.header("Newsletter Content")
        st.subheader("Episode Title")
        st.write(podcast_info['podcast_details']['episode_title'])
        display_podcast_summary(podcast_info)
        display_podcast_guest(podcast_info)
        display_podcast_highlights(podcast_info)


def create_dict_from_json_files(folder_path):
    data_dict = {}

    # Recursively find all JSON files in the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    podcast_info = json.load(file)
                    
                    # Extract podcast title based on the new structure
                    podcast_name = podcast_info['podcast_details']['podcast_details']['title']
                    
                    # Look for a .jpg file in the same directory
                    jpg_files = [f for f in os.listdir(root) if f.endswith('.jpg')]
                    image_path = os.path.join(root, jpg_files[0]) if jpg_files else None
                    
                    data_dict[podcast_name] = {
                        'info': podcast_info,
                        'image_path': image_path
                    }

    return data_dict





#def process_podcast_info(url):
#    f = modal.Function.lookup("corise-podcast-project", "process_podcast")
#    output = f.call(url, '/content/podcast/')
#    return output

if __name__ == '__main__':
    main()