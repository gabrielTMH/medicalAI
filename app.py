from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


# configure Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///model.db'
db = SQLAlchemy(app)


@app.before_first_request
def create_tables():
    db.create_all()


class UserRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem = db.Column(db.String(200), nullable=False)
    error_code1 = db.Column(db.String(200), nullable=False)
    error_code2 = db.Column(db.String(200), nullable=False)
    error_code3 = db.Column(db.String(200), nullable=False)

    interlock1 = db.Column(db.String(200), nullable=False)
    interlock2 = db.Column(db.String(200), nullable=False)
    interlock3 = db.Column(db.String(200), nullable=False)

    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<UserRequest %r>' % self.id


# index
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        problem = request.form['problem']
        error_code1 = request.form['error_code1']
        #add all codes here


        new_request = UserRequest(problem=problem, error_code1=error_code1)

        try:
            db.session.add(new_request)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your request'

    else:
        tickers = UserRequest.query.order_by(UserRequest.date_created).all()
        return render_template('index.html', tickers=tickers)

    # debugger mode
    if __name__ == "__main__":
        app.run(debug=True)

