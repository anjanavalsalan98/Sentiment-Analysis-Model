from flask import Blueprint, render_template, request
import requests
import json
 
url = 'https://sa-model.herokuapp.com/emotion'

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    data = request.form.get('msg','')
    if data == '':
        emotion = ''
    else:
        emotion = model(data)
    prediction = [{'msg':data,'emotion':emotion}]
    return render_template('home.html',prediction=prediction)

def model(data):
    msg = {'msg':data}
    response = requests.post(url,data=msg)
    emotion = json.loads(response.text)['emotion']
    return emotion