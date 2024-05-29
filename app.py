import streamlit as st
import pandas as pd
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sentencepiece
import base64
from langdetect import detect, LangDetectException

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title='Welcome!', layout='centered')

# Function to convert an image file to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Paths for the images
background_image_path = "/Uni Notes/GP/GP2/streamlit 2/streamlit/back.webp"
logo_image_path = "D:/Uni Notes/GP/GP2/streamlit 2/streamlit/logo4.png"
# Convert images to base64
background_image_base64 = get_base64_of_bin_file(background_image_path)
logo_image_base64 = get_base64_of_bin_file(logo_image_path)

# Custom CSS to enhance UI
def set_custom_ui():
    css = """
    <style>
    @import url('https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css');

    body, html {
        height: 100%;
        margin: 0;
        font-family: 'Arial', sans-serif;
    }

    .stApp {
        font-family: 'Helvetica';
    }

    .big-font {
        font-size: 40px; /* Increased size for more prominence */
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 5px #000000;
    }

    .medium-font {
        font-size: 30px; /* Slightly smaller than big-font */
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 5px #000000;
    }

    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 40px;
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        width: 300px;
    }

    .login-title {
        font-size: 24px;
        color: #333;
    }

    .login-subtitle {
        color: #888;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .form-container {
        width: 100%;
    }

    input[type="email"], input[type="password"] {
        width: 100%;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }

    .action-container {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px.
    }

    .forgot-password {
        color: #ff5c8d;
        text-decoration: none.
    }

    .forgot-password:hover {
        text-decoration: underline.
    }

    button {
        background-color: #ff5c8d.
        color: white.
        border: none.
        padding: 12px 20px.
        border-radius: 5px.
        cursor: pointer.
    }

    button:hover {
        background-color: #e44d79.
    }

    .social-login {
        font-size: 14px.
        color: #666.
    }

    .social-login a {
        color: #ff5c8d.
        text-decoration: none.
    }

    .social-login a:hover {
        text-decoration: underline.
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_custom_ui()

# File path for the CSV file
file_path = 'user_credentials.csv'

# Load user data if the file exists, otherwise create an empty DataFrame
def load_user_data():
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['username', 'password'])

# Save new user to CSV
def save_user(username, password, user_data):
    new_data = pd.DataFrame([[username, password]], columns=['username', 'password'])
    updated_data = pd.concat([user_data, new_data], ignore_index=True)
    updated_data.to_csv(file_path, index=False)

# Validate the user credentials
def validate_login(username, password, user_data):
    user_record = user_data[(user_data['username'] == username) & (user_data['password'] == password)]
    return not user_record.empty

# Define the path to the saved model and tokenizer
save_dir = r"D:\Uni Notes\GP\GP2\streamlit 2\streamlit\new_model_configs"  # Using raw string literal

# Check if the directory contains necessary files
required_files = ['config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(save_dir, f))]

if missing_files:
    raise FileNotFoundError(f"The directory '{save_dir}' is missing the following required files: {', '.join(missing_files)}")

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(save_dir)
model = T5ForConditionalGeneration.from_pretrained(save_dir)

# Text summarization function with enhanced error handling
def summarize_text(input_text):
    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""

# Main application
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        user_data = load_user_data()

        st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/webp;base64,{background_image_base64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
        """, unsafe_allow_html=True)

        with st.form(key='login_form'):
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
                <div style="background-color: rgba(211, 211, 211, 0.8); padding: 50px; border-radius: 10px; text-align: center; box-shadow: 0 10px 20px rgba(11, 24, 133, 0.64); width: 350px;">
                    <div style="margin-bottom: 20px;">
                        <img src="data:image/png;base64,{logo_image_base64}" alt="ARABRIEF Logo" style="width: 200px;">
                    </div>
            """, unsafe_allow_html=True)
            
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            st.markdown("</div></div>", unsafe_allow_html=True)
            login_button = st.form_submit_button(label='Login')
        
        if login_button:
            if validate_login(login_username, login_password, user_data):
                st.session_state.logged_in = True
                st.session_state.logged_in_username = login_username
                st.success("Login Successful!")
                st.experimental_rerun()
            else:
                st.error("Login Failed: Incorrect username or password.")
    else:
        st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/webp;base64,{background_image_base64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="navbar">
            <a href="#" onClick="javascript:window.location.reload();">Logout</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; margin-top: 150px;">
            <div style="background-color: rgba(211, 211, 211, 0.8); padding: 50px; border-radius: 10px; text-align: center; box-shadow: 0 10px 20px rgba(11, 24, 133, 0.64); width: 350px;">
                <div style="margin-bottom: 20px;">
                    <img src="data:image/png;base64,{logo_image_base64}" alt="ARABRIEF Logo" style="width: 200px;">
                </div>
        """, unsafe_allow_html=True)
        
        original_text = st.text_area("Enter text here:", height=300, placeholder="Enter text to summarize", key="input_text")

        if st.button('Submit'):
            if not original_text.strip():
                st.error("Please enter a text.")
            else:
                try:
                    language = detect(original_text)
                    if language != 'ar':
                        st.error("Error: The text must be in Arabic.")
                    else:
                        summarized_text = summarize_text(original_text)
                        st.text_area("Summarized Text:", value=summarized_text, height=150, key="output_text", help="This is the summarized text.")
                        with open('summarized_texts.csv', 'a', encoding='utf-8') as f:
                            f.write(f"{original_text},{summarized_text}\n")
                except LangDetectException:
                    st.error("Error: Unable to detect the language of the text. Please enter a valid text.")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
        
        st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()