import streamlit as st
import torch
from PIL import Image
from transformers import pipeline
from googletrans import Translator
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
        caption_image = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Error loading image captioning model: {e}")
        st.stop()
    
    try:
        translator = Translator()
    except Exception as e:
        st.error(f"Error initializing translator: {e}")
        st.stop()
    
    return caption_image, translator

caption_image, translator = load_pipelines()

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Hindi": "hi",
    "Punjabi": "pa",
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
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Fallback to original text if translation fails

def caption_my_image(pil_image, language):
    try:
        # Generate the caption from the image
        caption_result = caption_image(images=pil_image)
        semantics = caption_result[0]['generated_text']  # Get the first caption
        confidence = caption_result[0].get('score', 'N/A')  # Assuming 'score' key exists
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None, None, None
    
    # Translate the caption if needed
    if language != "English":
        target_lang_code = SUPPORTED_LANGUAGES.get(language, "en")
        translated_semantics = translate_text(semantics, target_lang_code)
    else:
        translated_semantics = semantics
    
    # Optionally truncate the text
    truncated_semantics = truncate_text(translated_semantics, max_length=200)
    
    # Generate the corresponding audio
    audio_path = generate_audio(truncated_semantics, language=SUPPORTED_LANGUAGES.get(language, "en"))
    
    return semantics, translated_semantics, confidence, audio_path

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
    st.sidebar.image("imagec.jpg",use_column_width=True)
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
    st.sidebar.header("Instructions :")
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
        1. Click on **"First Version of the App"** link.
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
                semantics, translated_semantics, confidence, audio_path = caption_my_image(
                    pil_image, language_input
                )
            
            if semantics is not None:
                # Display results
                st.subheader("📝 Generated Caption")
                st.write(semantics)
                
                st.subheader("🌐 Translated Caption")
                st.write(translated_semantics)

                if audio_path and os.path.exists(audio_path):
                    st.subheader("🔊 Audio Caption")
                    try:
                        # Streamlit can play MP3 files directly
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")
                    
                    # Download Button for text caption
                    st.subheader("💾 Download Caption")
                    st.download_button(
                        label="Download Caption as Text",
                        data=download_caption(translated_semantics),
                        file_name="caption.txt",
                        mime="text/plain"
                    )
                    
if __name__ == "__main__":
    main()
