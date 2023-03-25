import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import engine
from sqlalchemy.orm import sessionmaker
from project_orm import User
from flask import Flask,session,flash,redirect,render_template,url_for
from flask.globals import request
from utils import *

app = Flask(__name__)
app.secret_key = 'AI personal Trainer'
engine = create_engine('sqlite:///database.sqlite')
session = sessionmaker(bind=engine)
sess = session()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('base.html',title='login')

@app.route('/signin',methods=['GET','POST'])
def signup():
      if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email') 
            password = request.form.get('password')
            cpassword = request.form.get('cpassword')
            if name and len(name) >= 3:
                  if email and validate_email(email) == True: 
                        if password and len(password)>= 6:
                              if cpassword and cpassword == password:
                                    try: 
                                          new_user = User(name=name,email=email,password=password)
                                          sess.add(new_user)
                                          sess.commit()
                                          flash('Account created successfully','success')
                                          return redirect("home.html")
                                    except:
                                         flash('email already exists','danger')
                              else:
                                   flash('confirm password is not matching','danger')
                        else:
                             flash('password is of 6 characters or more','danger')
                  else:
                       flash('email is not valid','danger')
            else:
                 flash('name is not valid must contain 3 or more letters','danger')
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