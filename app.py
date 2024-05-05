import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from spacy import displacy
from transformers import pipeline
import nltk
from nltk import ngrams, pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from snownlp import SnowNLP
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    return plt

# Function to generate word cloud for Chinese text
def generate_wordcloud_chinese(text):
    wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    return plt

# Function for text summarization
def text_summarization(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function for translation from English to Chinese
def translate_to_chinese(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='zh-cn')
    return translation.text

# Function for Chinese text summarization using SnowNLP
def text_summarization_chinese(text):
    s = SnowNLP(text)
    summary = s.summary(3)  # Summarize into 3 sentences
    return '\n'.join(summary)

# Function for Chinese sentiment analysis using SnowNLP
def sentiment_analysis_chinese(text):
    s = SnowNLP(text)
    sentiment_score = s.sentiments
    return sentiment_score

# Function for n-gram analysis
def ngram_analysis(text, n=2):
    tokens = nltk.word_tokenize(text)
    n_grams = ngrams(tokens, n)
    return list(n_grams)

# Function for sentiment analysis
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

# Function for parts of speech analysis
def pos_analysis(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags

# Function for named entity recognition
def ner_analysis(text):
    doc = nlp(text)
    return doc

# Streamlit UI
st.title("Text Analysis Tool")
st.write("CS49 project by Georgia Tai! Inspired by the project of Prakhar Rathi.")

# Sidebar navigation
page = st.sidebar.selectbox("Select a function:", ("Home", "Word Cloud", "Text Summarization", "N-gram Analysis", "Sentiment Analysis", "Parts of Speech Analysis", "Named Entity Recognition", "Translate English to Chinese"))

# Home page
if page == 'Home':
	
	st.write(
				"""
				### Project Description
				This CS49 final project explores the usage of NLP libraries and streamlit. The app includes the following features:\n
				1. Word Cloud\n
				2. Text Summarization\n
				3. N-gram Analysis (visualized by Word Cloud)\n
				4. Sentiment Analysis (visualized using bar charts)\n
				5. Part of Speech Analysis\n
				6. Named Entity Recognition\n
				7. Chinese Text Analysis (including translation, summarization, and Word Cloud)\n
				P.S. There are some notes included in some of the features (Text Summarization, N-gram Analysis, Chinese Translation)!
				"""
		)
	st.image("cat_smile.jpg", width=300)

# Display selected function
if page == "Word Cloud":
    st.header("Word Cloud")
    text = st.text_area("Enter your text here:")
    if st.button("Generate Word Cloud"):
        if text:
            wc_plot = generate_wordcloud(text)
            st.pyplot(wc_plot)
        else:
            st.error("Please enter some text.")

elif page == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter your text here:")
    if st.button("Summarize"):
        if text:
            summary = text_summarization(text)
            st.write(summary)
            st.subheader("Notes and Thoughts from Georgia")
            st.write("As users can probably tell, this summary may not be the best, depening on the content and not to mention the pretty long run time. On top of that, there might be some grammatical, including punctuation, errors. The original app used gensim, who deleted their summary functions after version 4.0.0, and I couldn't get version 3.8.3 to work. Therefore, I had to take an alternative path and use the Transformers library. I also tried using the ChatGPT API to summarize the text, but I wasn't able to make it work yet. I will continue to look into that in order to make this function better.")
        else:
            st.error("Please enter some text.")

elif page == "N-gram Analysis":
    st.header("N-gram Analysis")
    text = st.text_area("Enter your text here:")
    n_value = st.slider("Select the value of 'n' for n-grams:", 2, 5, 2)
    if st.button("Analyze"):
        if text:
            n_grams = ngram_analysis(text, n_value)
            st.write(n_grams)
            # Generate word cloud from n-gram result
            n_gram_text = ' '.join([' '.join(ngram) for ngram in n_grams])
            wc_plot = generate_wordcloud(n_gram_text)
            st.pyplot(wc_plot)
            st.subheader("Notes and Thoughts from Georgia")
            st.write("I would say I am not really happy with this feature, since the Word Cloud did not turn out the way I expected it to. The original app did not have a good way of visualizing the N-gram analysis, that's why I chose to integrate the Word Cloud here too. The libraries are able to separate the texts using the N-gram analysis, but instead of showing phrases, only words were shown in the Word Cloud. However, I haven't really had the chance to dive deeper and fix this, and thus a future job for project would be to fix this problem.")
        else:
            st.error("Please enter some text.")

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter your text here:")
    if st.button("Analyze Sentiment"):
        if text:
            sentiment_scores = sentiment_analysis(text)
            # Create bar chart for sentiment analysis
            categories = list(sentiment_scores.keys())
            scores = list(sentiment_scores.values())
            fig, ax = plt.subplots()
            ax.bar(categories, scores, color=['red', 'gray', 'green', 'gray'])
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Sentiment Analysis')
            ax.set_ylim(0, 1)  # Set y-axis limit to range from 0 to 1
            for i, v in enumerate(scores):
                ax.text(i, v + 0.02, str(round(v, 2)), ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.error("Please enter some text.")

elif page == "Parts of Speech Analysis":
    st.header("Parts of Speech Analysis")
    text = st.text_area("Enter your text here:")
    if st.button("Analyze POS"):
        if text:
            pos_tags = pos_analysis(text)
            st.write(pos_tags)
        else:
            st.error("Please enter some text.")

elif page == "Named Entity Recognition":
    st.header("Named Entity Recognition")
    text = st.text_area("Enter your text here:")
    if st.button("Recognize Entities"):
        if text:
            doc = ner_analysis(text)
            
            # Extract named entities
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Visualize named entities using spaCy's displacy
            html = displacy.render(doc, style="ent", page=True)
            st.components.v1.html(html, height=10000)

        else:
            st.error("Please enter some text.")

elif page == "Translate English to Chinese":
    st.header("Translate English to Chinese")
    text = st.text_area("Enter English text to translate:")
    if st.button("Translate"):
        if text:
            chinese_text = translate_to_chinese(text)
            st.subheader("Translated Chinese Text:")
            st.write(chinese_text)
			# Generate word cloud for translated Chinese text
            translated_wc_plot = generate_wordcloud_chinese(chinese_text)
            st.subheader("Word Cloud of the Translated Text:")
            st.pyplot(translated_wc_plot)
			# Perform text summarization on translated Chinese text
            summary = text_summarization_chinese(chinese_text)
            st.subheader("Summary of Translated Text:")
            st.write(summary)
            # Perform sentiment analysis on translated Chinese text
            sentiment_score = sentiment_analysis_chinese(chinese_text)
            st.subheader("Sentiment Analysis for Translated Text:")
            st.write(sentiment_score)
			# Notes and Reflections
            st.subheader("Notes and Thoughts from Georgia")
            st.write("I have not yet found a way of to quantify how good the translation is and to compare the similarity of two texts. However, there are some interesting findings here. First, the Word Cloud did not exactly work the way I wanted it to, since it's showing more of sentences rather than words. Secondly, I feel like snowNLP did a decent job in sentiment analyzing and summarizing Chinese texts, despite not having punctuations in its summary. Lastly, the googletrans library is really easy to use and did a fair job translating.")
        else:
            st.error("Please enter some English text.")