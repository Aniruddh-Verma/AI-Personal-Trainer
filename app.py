import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import engine
from sqlalchemy.orm import sessionmaker
from project_orm import User
from flask import Flask,session,flash,redirect,render_template,url_for

app = Flask(__name__)
engine = create_engine('sqlite:///database.sqlite')
session = sessionmaker(bind=engine)
sess = session()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('base.html',title='login')

@app.route('/signin',methods=['GET','POST'])
def signup():
    return render_template('signup.html',title='register')

@app.route('/forgot',methods=['GET','POST'])
def forgot():
    return render_template('forgot.html',title='forgot password')

@app.route('/Home',methods=['GET','POST'])
def home():
    return render_template('home.html',title='home')

@app.route('/About',methods=['GET','POST'])
def about_us():
    return render_template('About.html',title='About Us')

@app.route('/Logout',methods=['GET','POST'])
def logout():
    return redirect('home.html',title='Logout')


if __name__ == '__main__':
    app.run(debug=True)