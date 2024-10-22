import os
import cohere
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Cohere API key from the .env file
cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize Cohere Client
co = cohere.Client(cohere_api_key)

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try fetching the manual transcript
    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code  # Save the detected language
    except:
        # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code  # Save the detected language
        except:
            # If no auto-generated transcript is found, raise an exception
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code  # Return both the transcript and detected language

def summarize_with_langchain_and_cohere(transcript, model='command-r-plus'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4])  # Adjust this as needed

    # Prepare the prompt for a detailed summarization
    prompt = f'''Summarize the following text in detail in English:
    Text: {text_to_summarize}

    Please generate a detailed summary with:
    - An introduction that briefly explains the overall topic.
    - Key points and subtopics discussed in the text.
    - Bullet points covering all the essential aspects and subtopics.
    - A conclusion summarizing the main takeaways.
    - Additional insights or background information where relevant.

    The summary should be comprehensive, clear, and cover the text's main ideas.'''

    # Start summarizing using Cohere
    response = co.generate(
        model=model,
        prompt=prompt,
        max_tokens=800,  # Increased token limit for more detailed output
        temperature=0.8  # Lower temperature for focused output
    )
    
    return response.generations[0].text

def answer_question_with_cohere(question, transcript_or_summary, model='command-r-plus'):
    # Prepare the prompt for answering questions
    prompt = f'''Based on the following information, please provide a detailed answer to the user's question:
    Information: {transcript_or_summary}

    Question: {question}

    Answer in a concise yet detailed manner.'''

    # Generate an answer using Cohere
    response = co.generate(
        model=model,
        prompt=prompt,
        max_tokens=300,  # Adjust max_tokens for a reasonable answer length
        temperature=0.7  # Slightly lower temperature for focused answers
    )
    
    return response.generations[0].text

def main():
    st.title('Advanced YouTube Video Summarizer')
    
    # Maintain session state for the summary
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    
    link = st.text_input('Enter the link of the YouTube video you want to summarize:')

    if st.button('Start'):
        if link:
            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text('Loading the transcript...')
                progress.progress(25)

                # Getting both the transcript and language_code
                transcript, _ = get_transcript(link)

                status_text.text(f'Creating detailed summary...')
                progress.progress(75)

                # Getting the detailed summary
                summary = summarize_with_langchain_and_cohere(transcript)

                # Store the summary and transcript in session state
                st.session_state.summary = summary
                st.session_state.transcript = transcript

                status_text.text('Summary:')
                st.markdown(st.session_state.summary)
                progress.progress(100)

            except Exception as e:
                st.write(str(e))
        else:
            st.write('Please enter a valid YouTube link.')

    # Show the question input box and handle the answer generation if the summary is already generated
    if st.session_state.summary:
        st.write("You can now ask questions based on the video.")
        user_question = st.text_input("Ask a question about the video:")

        if st.button("Get Answer"):
            if user_question:
                with st.spinner('Generating an answer...'):
                    answer = answer_question_with_cohere(user_question, st.session_state.transcript)
                    st.markdown(f"**Answer:** {answer}")
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
