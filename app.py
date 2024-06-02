from flask import Flask, render_template, request, redirect, url_for,flash
from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import assemblyai as aai
from pydub.utils import mediainfo
import datetime
from transformers import BartForConditionalGeneration, BartTokenizer
from rake_nltk import Rake
import warnings
warnings.filterwarnings("ignore")
from summarizer import Summarizer

from gtts import gTTS



from nltk.sentiment.vader import SentimentIntensityAnalyzer



aai.settings.api_key = f"3da1144af9fa4c809e459899cccff6c5"



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Set the database connection URL
app.secret_key = 'yjcnsjnf' 
app.config['UPLOAD_FOLDER']="static"
db = SQLAlchemy(app)

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

bert_model = Summarizer()

transcript_text = ""

def summarize_article(article_text):
    input_ids = tokenizer.encode(article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=300, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

def bert_summary(text1):
    summary = bert_model(text1)
    return summary


def generate_summaries_with_keywords(text, max_keywords=10):
    # Extract keywords using RAKE
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = [phrase for phrase in r.get_ranked_phrases() if len(phrase.split()) >= 2 and len(phrase.split()) <= 4][:max_keywords]
    
    summaries = {}
    # Generate summaries for each keyword
    for keyword in keywords:
        # Tokenize and encode the keyword
        input_ids = tokenizer.encode(keyword, return_tensors="pt", max_length=1024, truncation=True)
        # Generate summary
        summary_ids = model.generate(input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)
        # Decode and store the summary
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries[keyword] = summary_text
    
    return summaries

def get_audio_info(file_path):
    audio_info = mediainfo(file_path)
    duration = int(float(audio_info["duration"]))
    start_time = "00:00:00"
    end_time = str(datetime.timedelta(seconds=duration))
    return duration, start_time, end_time

def sentiment_analysis(transcript_text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(transcript_text)
    return score


uploaded_podcast=None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        # print(user)
        if user:
            if user.password==password:
                # print("hii")
                if user.role == 'user':
                    return redirect(url_for('user_dashboard'))
                elif user.role == 'admin':
                    return redirect(url_for('admin_dashboard'))

        error = 'Invalid credentials. Please try again.'
        return render_template('login.html', error=error)

    return render_template('login.html', error=None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')

        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            flash('Username already exists. Please choose another.', 'error')
        else:
            user = User(username=username, password=password, role=role)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/user_dashboard', methods=['GET', 'POST'])
def user_dashboard():
    uploaded_podcast="Summerized text will here !!!"
    keyword_summaries = {}
    filename=None
    duration=None
    start_time=None
    global transcript_text
    end_time=None
    bert_summ=None
    bert_audio_file=None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            filename1=file.filename
            if file.filename != '':
                filename = secure_filename(file.filename)
                if filename != os.path.join(app.config['UPLOAD_FOLDER'], filename):
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                
                bert_audio_file = os.path.join(app.config['UPLOAD_FOLDER'], 'bert_summary.mp3')
                # podcast_audio_file = os.path.join(app.config['UPLOAD_FOLDER'], 'podcast_summary.mp3')

                uploaded_podcast = summarize_article(transcript.text)
                bert_summ=bert_summary(transcript.text)
                # Generate audio files for BERT summary and uploaded podcast summary
                combined_text = uploaded_podcast + " " + bert_summ
                if bert_summ:
                    tts = gTTS(text=combined_text, lang='en')
                    tts.save(bert_audio_file)


                keyword_summaries = generate_summaries_with_keywords(transcript.text)  # Generate keyword summaries
                duration, start_time, end_time = get_audio_info(os.path.join("static", filename))
                print(duration,start_time,end_time)
                transcript_text = transcript.text


    return render_template('user_dashboard.html', uploaded_podcast=uploaded_podcast,keyword_summaries=keyword_summaries,filename=filename,duration=duration,start_time=start_time,end_time=end_time,transcript_text=transcript_text,bert_summ=bert_summ,bert_audio_file=bert_audio_file)

@app.route('/admin_dashboard')
def admin_dashboard():
    user_count=User.query.filter_by(role="user").count()
    return render_template('admin_dashboard.html',user_count=user_count)
@app.route('/logout')
def logout():
    uploaded_podcast="Summerized text will here !!!"
    keyword_summaries = {}
    filename=None
    duration=None
    start_time=None
    transcript_text=None
    end_time=None
    bert_summ=None
    bert_audio_file=None
    return redirect(url_for("home"))

@app.route('/transcript')
def transcript_page():
    global transcript_text  # Replace this with the actual transcript text
    return render_template('transcript.html', transcript_text=transcript_text)

@app.route('/sentiment_details')
def sentiment_details():
    sentiment_score = sentiment_analysis(transcript_text)
    return render_template('sentiment_details.html', sentiment_score=sentiment_score)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
