from flask import Flask, render_template, request
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS
from spacy.lang.hi.stop_words import STOP_WORDS as HINDI_STOP_WORDS
from spacy.lang.kn.stop_words import STOP_WORDS as KANNADA_STOP_WORDS
from spacy.lang.ml.stop_words import STOP_WORDS as MALAYALAM_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOP_WORDS
from spacy.lang.zh.stop_words import STOP_WORDS as CHINESE_STOP_WORDS
from spacy.lang.ko.stop_words import STOP_WORDS as KOREAN_STOP_WORDS
from string import punctuation
from heapq import nlargest
from rouge import Rouge

app = Flask(__name__)

def summarizer(rawdocs, language="english"):
    if language == "english":
        stopwords = ENGLISH_STOP_WORDS
        nlp = spacy.load("en_core_web_sm")
    elif language == "hindi":
        stopwords = HINDI_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm")
    elif language == "kannada":
        stopwords = KANNADA_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm")
    elif language == "malayalam":
        stopwords = MALAYALAM_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm")
    elif language == "french":
        stopwords = FRENCH_STOP_WORDS
        nlp = spacy.load("fr_core_news_sm")
    elif language == "german":
        stopwords = GERMAN_STOP_WORDS
        nlp = spacy.load("de_core_news_sm")
    elif language == "chinese":
        stopwords = CHINESE_STOP_WORDS
        nlp = spacy.load("zh_core_web_sm")
    elif language == "korean":
        stopwords = KOREAN_STOP_WORDS
        nlp = spacy.load("ko_core_news_sm")
    else:
        raise ValueError("Unsupported language.")
    
    nlp.add_pipe('sentencizer')
    doc = nlp(rawdocs)
    tokens = [token.text for token in doc]
    word_freq={}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
    max_freq=max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word]=word_freq[word]/max_freq
    sent_tokens=[sent for sent in doc.sents]
    sent_scores={}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent]=word_freq[word.text]
                else:
                    sent_scores[sent]+=word_freq[word.text]
    select_len=int(len(sent_tokens)*0.3)
    summary=nlargest(select_len,sent_scores,key=sent_scores.get)
    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))

def calculate_rouge(summary, rawdocs):
    rouge = Rouge()
    scores = rouge.get_scores(summary, rawdocs)
    return scores[0]['rouge-1']['f']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/testcases')
def testcases():
    return render_template('testcases.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == "POST":
        rawtext = request.form['rawtext']
        language = request.form['language']
        
        supported_languages = ["english", "hindi", "kannada", "malayalam", "french", "german", "chinese", "korean"]
        
        if language in supported_languages:
            try:
                summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext, language)
                rouge_score = calculate_rouge(summary, rawtext)
                return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary, rouge_score=rouge_score)
            except ValueError as e:
                return render_template('error.html', message=str(e))
        else:
            return render_template('error.html', message="Unsupported language.")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
