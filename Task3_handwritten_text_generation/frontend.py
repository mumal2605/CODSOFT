# frontend.py
import streamlit as st
import requests

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/generate"

# --- Page Styling ---
def set_page_style():
    parchment_image_url = "https://www.publicdomainpictures.net/pictures/40000/velka/parchment-paper.jpg"
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("{parchment_image_url}");
        background-size: cover;
    }}
    /* Add other styles from previous answer... */
    .title {{ text-align: center; font-family: 'Garamond', serif; color: #382d21; }}
    .output-box {{ background-color: rgba(255, 250, 235, 0.8); border: 2px solid #5c4b3c; border-radius: 10px; padding: 20px; font-family: 'Georgia', serif; color: #333333; max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Main Application ---
def main():
    set_page_style()

    st.markdown("<h1 class='title'>🎭 Shakespearean Text Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #5c4b3c;'>Hark! Lend thy prompt to the digital bard, and witness verse unfold.</p>", unsafe_allow_html=True)

    with st.form("generation_form"):
        start_string = st.text_input("Starting Prompt", placeholder="Enter thy poetic musings...", value="ROMEO:", label_visibility="collapsed")
        col1, col2 = st.columns(2)
        with col1:
            num_chars = st.slider("Length of Verse (characters)", 100, 2000, 500)
        with col2:
            temperature = st.slider("Creativity (temperature)", 0.1, 1.5, 0.8, 0.05)
        submit_button = st.form_submit_button(label="Summon Text")

    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = ""

    if submit_button:
        if not start_string:
            st.warning("Pray, provide a starting phrase to inspire the muse!")
        else:
            with st.spinner("The bard is composing..."):
                payload = {"start_string": start_string, "num_generate": num_chars, "temperature": temperature}
                try:
                    response = requests.post(API_URL, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    st.session_state.generated_text = data.get("generated_text", "Alas, no text was returned.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Alas, the connection to the backend failed. Is it running? Error: {e}")
    
    if st.session_state.generated_text:
        st.markdown(f"<div class='output-box'>{st.session_state.generated_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()