import streamlit as st
import torch
from PIL import Image
from transformers import pipeline
import tempfile
import os
from gtts import gTTS

# Set the page configuration
st.set_page_config(
    page_title="🖼️ Enhanced Image Captioning App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pipelines
@st.cache_resource
def load_pipelines():
    try:
        # Image Captioning Pipeline
        caption_image = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Error loading image captioning model: {e}")
        st.stop()
    
    try:
        # Translation Pipeline
        # Using Helsinki-NLP's models for translation
        # Define a pipeline for each supported language
        translation_pipelines = {}
        translation_models = {
            "Spanish": "Helsinki-NLP/opus-mt-en-es",
            "French": "Helsinki-NLP/opus-mt-en-fr",
            "German": "Helsinki-NLP/opus-mt-en-de",
            "Chinese": "Helsinki-NLP/opus-mt-en-zh",
            "Japanese": "Helsinki-NLP/opus-mt-en-ja",
            "Hindi": "Helsinki-NLP/opus-mt-en-hi",
            # Add more models as needed
        }
        for lang, model_name in translation_models.items():
            translation_pipelines[lang] = pipeline("translation", model=model_name, device=0 if device == "cuda" else -1)
    except Exception as e:
        st.error(f"Error initializing translation pipelines: {e}")
        st.stop()
    
    return caption_image, translation_pipelines

caption_image, translation_pipelines = load_pipelines()

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Hindi": "hi",
    # Add more languages as needed
}

# Helper functions
def generate_audio(text, language='en'):
    try:
        # Initialize gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def truncate_text(text, max_length=200):
    return text[:max_length] + '...' if len(text) > max_length else text

def translate_text(text, target_language):
    if target_language == "English":
        return text  # No translation needed

    try:
        translator = translation_pipelines.get(target_language)
        if not translator:
            st.error(f"Translation for {target_language} is not supported.")
            return text
        translated = translator(text, max_length=400)[0]['translation_text']
        return translated
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Fallback to original text if translation fails

def caption_my_image(pil_image, language):
    try:
        # Generate the caption from the image
        caption_result = caption_image(images=pil_image)[0]  # Get a single caption
        caption = caption_result['generated_text']
    except Exception as e:
        st.error(f"Error generating captions: {e}")
        return None, None, None  # Return None for all outputs
    
    # Translate the caption if needed
    translated_caption = translate_text(caption, language)
    
    # Optionally truncate the text
    truncated_caption = truncate_text(translated_caption, max_length=200)

    # Generate audio for the translated caption
    audio_lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    audio_path = generate_audio(truncated_caption, language=audio_lang_code)

    return caption, truncated_caption, audio_path

def download_caption(text):
    return text.encode('utf-8')

# Custom CSS for background hover effect and button colors
st.markdown(""" 
    <style>
        body {
            background: linear-gradient(135deg, rgba(255,0,0,0.7), rgba(0,0,255,0.7));
            transition: background-color 0.5s ease;
        }
        .stButton > button:hover {
            background-color: rgba(255, 165, 0, 0.8);
            color: white;
        }
        .stButton > button {
            background-color: rgba(0, 255, 0, 0.7);
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 16px;
            transition: background-color 0.3s, transform 0.3s;
        }
        .stButton > button:active {
            transform: scale(0.95);
        }
        .creator-link {
            color: black;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            display: block;  /* Make the link a block element */
            text-align: center;  /* Center text */
            margin-top: 10px;   /* Add some space above */
        }
        .creator-link:hover {
            background-color: white;  /* Background color on hover */
            color: red;              /* Change text color on hover */
            padding: 5px;           /* Add some padding on hover */
            border-radius: 5px;     /* Rounded corners */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Interface
def main():
   
    # Add an image to the sidebar
    st.sidebar.image("imagec.jpg", use_column_width=True)
    st.sidebar.title("🖼️ Enhanced Image Caption Generator App")
    
    # Creator link at the top of the sidebar
    st.sidebar.markdown(
        '<a href="https://www.linkedin.com/in/sm980/" class="creator-link">Created by SHASHWAT MISHRA</a>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        """
        **Welcome to the Enhanced Image Caption Generator App!**  
        This application generates captions for images and translates them into your preferred language. 

        **Features:**
        - Upload an image to get an automatically generated caption.
        - Translate the caption into multiple languages.
        - Listen to the audio version of the caption.
        - Download the caption as a text file.
        """
    )
    st.sidebar.header("Instructions : ")
    st.sidebar.markdown(
        """
        **1. Upload an Image:**  
        Click on the 'Select Image' button in the sidebar to upload an image (JPEG, PNG).

        **2. Select Language:**  
        Choose your preferred language for the caption from the dropdown menu.

        **3. Generate Caption:**  
        Click on the 'Generate' button to create the caption. The app will process the image and display the caption along with the translated version.

        **4. Listen to the Caption:**  
        An audio version of the caption will be generated. Click the play button to listen to it.

        **5. Download the Caption:**  
        After generating the caption, you can download it as a text file by clicking the 'Download Caption' button.

        **Supported Languages:**
        - English
        - Spanish
        - French
        - German
        - Chinese
        - Japanese
        - Hindi
        """
    )

    # Instructions for using the additional app version
    st.header("Instructions for Another Version : ")
    st.markdown(
        """
        If you wish to try another version of the app, you can access it by clicking the link above.  
        This alternative version may offer different functionalities and models for image captioning.

        **Steps to use:**
        1. Click on **"First Version of this App"** link.
        2. Follow the on-screen instructions provided in that version.
        3. Upload your image and generate captions as needed.
        """
    )
    
    # Additional link for another version of the app
    st.markdown(
        '<a href="https://huggingface.co/spaces/kingsm997/Image-Caption-Generator" class="creator-link"> First Version of this App</a>',
        unsafe_allow_html=True
    )

    # Sidebar Inputs
    image_input = st.file_uploader("📸 Select Image", type=["png", "jpg", "jpeg"])
    language_input = st.selectbox("🌐 Language", options=list(SUPPORTED_LANGUAGES.keys()), index=0)
    generate_button = st.button("✨ Generate")
    
    if generate_button:
        if image_input is not None:
            try:
                # Read the image
                pil_image = Image.open(image_input).convert("RGB")
                st.image(pil_image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error opening image: {e}")
                st.stop()
            
            with st.spinner("Generating caption..."):
                caption, translated_caption, audio_path = caption_my_image(
                    pil_image, language_input
                )
            
            if caption is not None:
                # Display results
                st.subheader("📝 Generated Caption")
                st.write(caption)

                st.subheader("🌐 Translated Caption")
                st.write(translated_caption)

                if audio_path and os.path.exists(audio_path):
                    st.subheader("🔊 Audio Caption")
                    try:
                        # Streamlit can play MP3 files directly
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Error loading audio: {e}")

                # Download button for text caption
                st.download_button(
                    label="📥 Download Caption",
                    data=download_caption(translated_caption),
                    file_name="caption.txt",
                    mime="text/plain",
                )
        else:
            st.warning("Please upload an image to generate a caption.")

if __name__ == "__main__":
    main()
