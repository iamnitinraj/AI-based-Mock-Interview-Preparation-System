from flask import Blueprint, render_template, redirect, url_for, request, flash, abort
from models import User
from flask_login import current_user, login_required
import pandas as pd
import speech_recognition as sr
import pandas as pd
import pyttsx3
import os
import statistics
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import myspsolution as mysp
from decimal import Decimal
from pydub import AudioSegment as am
from app import db

# import whisper 

model = SentenceTransformer('all-distilroberta-v1')
# model_stt = whisper.load_model("base.en")

summarizer = pipeline("summarization", model = 'sshleifer/distilbart-cnn-12-6')


user_response = ""
text = []                   #It will contain ['Ques1','ans1','Ques2','ans2' ....]
ques_index = [0]
answise_cosine_similarity = []
words_per_minute_per_answer = []
balance_per_answer = []                #Speech duration / total duration
overall_score = 0

class _TTS:

    engine = None
    rate = None
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.setProperty('rate', 135)


    def start(self,text_):
        self.engine.say(text_)
        self.engine.runAndWait()





#Data Set used
hr = pd.read_csv('static/hr.csv')
se = pd.read_csv('static/software_engineer.csv')
ds = pd.read_csv('static/data_science.csv')
data = None

main = Blueprint('main', __name__)




@main.route('/')
def index():
    return render_template('index.html')



@login_required
@main.route('/launch')
def launch():
    user = User.query.filter_by(email=current_user.email).first_or_404()
    return render_template('launch.html', user = user)


@main.route('/contact')
def contact():
   
    return render_template('contactus.html')




@login_required
@main.route('/interview' ,methods = ['GET', 'POST'])
def interview():
    global data
    user = User.query.filter_by(email=current_user.email).first_or_404()
    if request.form['dropdown'] == 'hr':
        data = hr
    elif request.form['dropdown'] == 'se':
        data = se
    elif request.form['dropdown'] == 'ds':
        data = ds

    return render_template('interview.html' , user = user)


@login_required
@main.route("/speak",methods = ['POST'])
def speech():
    if request.method == 'POST':
        try:
            tts = _TTS()
            tts.start(text[-1])
            del(tts)
        except:
            try:
                tts = _TTS()
                tts.start(text[-1])
                del(tts)
            except:
                pass
    return 'done'



@login_required
@main.route("/interview/<int:qno>")
def interviewwork(qno):
    if qno == 1:
        user = User.query.filter_by(email=current_user.email).first_or_404()
        if len(text) == 0:
            text.append(data['Ques'][0])
        return render_template('interview2.html' , user = user, text = text, i = qno + 1)

    if qno == 11:
        return redirect(url_for('main.result'))

    user = User.query.filter_by(email=current_user.email).first_or_404()
    if len(text) >2 and user_response == text[-2]:
        return render_template('interview2.html' , user = user, text = text, i = qno)

    text.append(user_response)


    
    #Generating Next Question According to user's previous ans
    ans=user_response
    ques=''
    max_score=-float('inf')
    idx = None
    for i in range(1,data.shape[0]):
        nques=data['Ques'][i]
        if data['Ques'][i] not in text:
            embeddings1 = model.encode(ans, convert_to_tensor=True)
            embeddings2 = model.encode(nques, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            if max_score < (cosine_scores[0][0]):
                max_score=(cosine_scores[0][0])
                ques=nques
                idx = i

    ques_index.append(idx)
    text.append(ques)


    return render_template('interview2.html' , user = user, text = text, i = qno + 1)





@main.route('/audio', methods=['POST'])
def audio():
    global user_response
    global words_per_minute_per_answer
    user_response = ""


    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename

    # if file.filename == '':
    #     flash('No selected file')
    #     return redirect(request.url)


    file_name = 'audio' +".mp3"
    full_file_name = os.path.join('upload', file_name)
    file.save(full_file_name)


    r = sr.Recognizer()


    sound = am.from_file('upload/audio.mp3')
    sound = sound.set_frame_rate(44000)
    sound = sound.set_channels(1)
    sound.export('upload/audio.wav', format='wav')

    with sr.AudioFile('upload/audio.wav') as source:
        audio_data = r.record(source)
        response = r.recognize_google(audio_data, language='en-IN', show_all=True)
        
    try:
 
        user_response = response['alternative'][0]['transcript']
        overview = mysp.mysptotal('audio',os.getcwd() + r"\upload")[0]
        
        try:
            words_per_minute = (Decimal(overview['number_of_syllables'])/Decimal(overview['original_duration']))*60/Decimal(1.66)
            words_per_minute_per_answer.append(words_per_minute)
        except:
            words_per_minute_per_answer.append(0)

        try:
            balance = overview['balance']
            balance_per_answer.append(balance)
        except:
            balance_per_answer.append(0)

    except:
        tts = _TTS()
        tts.start("Speak Louder Couldn't hear you properly")
        del(tts)
        user_response = ""
    
    return '<h1>Success</h1>'



 
@login_required
@main.route('/result')
def result():
    metrics = {}
    global text
    user = User.query.filter_by(email=current_user.email).first_or_404()
 


    #Question Cosine Simlarity

    if len(text)%2 != 0:
        text.append("")
    
    ans_idx = 1
    q_idx = 0
    while ans_idx < len(text):
        curr_user_ans = text[ans_idx]
        if curr_user_ans != "":

            if len(curr_user_ans) > 500:
                summary = summarizer(curr_user_ans, max_length = 100, min_length = 80, do_sample = False)
                curr_user_ans = summary[0]['summary_text']
            
            data_ans = data['Ans'][ques_index[q_idx]]
            data_ans = data_ans.split('!@#@!')

            for k in range(len(data_ans)):
                if len(data_ans[k]) > 500: 
                    summary = summarizer(data_ans[k], max_length = 100, min_length = 80, do_sample = False)
                    data_ans[k] = summary[0]['summary_text']
                        
            embeddings1 = model.encode(curr_user_ans, convert_to_tensor=True)
            temp = []
            for x in data_ans:
                embeddings2 = model.encode(x, convert_to_tensor=True)
                cosine_scores = util.cos_sim(embeddings1, embeddings2)
                temp.append(cosine_scores)

            answise_cosine_similarity.append(max(temp))

        else:
            answise_cosine_similarity.append(0)

        ans_idx += 2
        q_idx += 1
    performance_score = statistics.median(answise_cosine_similarity)
    if type(performance_score) != float and type(performance_score) != int:
        performance_score = performance_score.item()
    
    performance_score = round(performance_score * 100,2)
    metrics['performance_score'] = performance_score



    #Verbal Fluency (WPM)
    
    if len(words_per_minute_per_answer) == 0:
        verbal_fluency = 0
    else:
        verbal_fluency = statistics.median(words_per_minute_per_answer)

    vf = 0
    if verbal_fluency >= 120 and verbal_fluency <=150:
        metrics['verbal_fluency'] = 'Excellent'
        vf = 3
    elif (verbal_fluency > 150 and verbal_fluency < 160) or (verbal_fluency > 100 and verbal_fluency < 120):
        metrics['verbal_fluency'] = 'Good'
        vf = 2
    elif (verbal_fluency > 160) or (verbal_fluency < 100):
        metrics['verbal_fluency'] = 'Bad'
        vf = 1

    
    #Confidence (balance)

    if len(balance_per_answer) == 0:
        confidence = 0
    else:
        confidence = statistics.median(balance_per_answer)

    confidence = float(confidence)
   

    if (confidence >= 0.7):
        metrics['confidence'] = 'Excellent'
    elif (confidence >= 0.5 and confidence < 0.7):
        metrics['confidence'] = 'Good, but can be better'
    else:
        metrics['confidence'] = 'Need improvement'
    metrics['confidence_value'] = confidence*100

    text = []
    overall_score = 0.5*metrics['performance_score'] + 10*vf + 20* confidence
    metrics['overall_score'] = overall_score
    user.best_score = max(user.best_score,overall_score)
    db.session.commit()
    
    return render_template('result.html' , user = user, metrics = metrics)
