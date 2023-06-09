from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
from sqlalchemy import desc

# configure Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///model.db'
db = SQLAlchemy(app)


@app.before_first_request
def create_tables():
    db.create_all()


with app.app_context():
    # create the tables
    db.create_all()


class UserRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subsystem = db.Column(db.String(200), nullable=False)
    problem = db.Column(db.String(200), nullable=False)
    error_code1 = db.Column(db.String(200), nullable=False)
    error_code2 = db.Column(db.String(200), nullable=False)
    error_code3 = db.Column(db.String(200), nullable=False)

    interlock1 = db.Column(db.String(200), nullable=False)
    interlock2 = db.Column(db.String(200), nullable=False)
    interlock3 = db.Column(db.String(200), nullable=False)
    res1 = db.Column(db.String(200), nullable=True)
    res2 = db.Column(db.String(200), nullable=True)
    res3 = db.Column(db.String(200), nullable=True)
    other_codes = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(200), nullable=False)


    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<UserRequest %r>' % self.id


# index
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        subsystem = request.form['subsystem']
        problem = request.form['problem']
        error_code1 = request.form['error_code1']
        error_code2 = request.form['error_code2']
        error_code3 = request.form['error_code3']

        interlock1 = request.form['interlock1']
        interlock2 = request.form['interlock2']
        interlock3 = request.form['interlock3']

        other_codes = request.form['other_codes']

        # create "issue" for model
        complete_issue_list = [subsystem, problem, error_code1, error_code2, error_code3, interlock1, interlock2,
                               interlock3, other_codes]
        complete_issue = f'{subsystem}{problem}{error_code1}{error_code2}{error_code3}{interlock1}{interlock2}' \
                         f'{interlock3}{other_codes}'

        model, vectorizer = unpickle_and_split_pipeline()
        result = input_to_result(complete_issue_list, model, vectorizer)
        res1=result[0]
        res2=result[1]
        res3=result[2]
        result = str(result)


        new_request = UserRequest(subsystem=subsystem, problem=problem, error_code1=error_code1,
                                  error_code2=error_code2, error_code3=error_code3,
                                  interlock1=interlock1, interlock2=interlock2,
                                  interlock3=interlock3, other_codes=other_codes, result=result,res1=res1,res2=res2,res3=res3
                                  )

        try:
            db.session.add(new_request)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your request'

    else:
        requests = UserRequest.query.order_by(desc(UserRequest.date_created)).limit(10).all()
        return render_template('index.html', requests=requests)


# delete
@app.route('/delete/<int:id>')
def delete(id):
    request_to_delete = UserRequest.query.get_or_404(id)

    try:
        db.session.delete(request_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that request'


# update
@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    user_request = UserRequest.query.get_or_404(id)

    if request.method == 'POST':
        user_request.subsystem = request.form['subsystem']
        user_request.problem = request.form['problem']
        user_request.error_code1 = request.form['error_code1']
        user_request.error_code2 = request.form['error_code2']
        user_request.error_code3 = request.form['error_code3']

        user_request.interlock1 = request.form['interlock1']
        user_request.interlock2 = request.form['interlock2']
        user_request.interlock3 = request.form['interlock3']

        user_request.other_codes = request.form['other_codes']

        # create "issue" for model
        complete_issue_list = [user_request.subsystem, user_request.problem, user_request.error_code1,
                               user_request.error_code2, user_request.error_code3, user_request.interlock1,
                               user_request.interlock2, user_request.interlock3, user_request.other_codes]

        model, vectorizer = unpickle_and_split_pipeline()
        result = input_to_result(complete_issue_list, model, vectorizer)
        result = str(result)

        user_request.result = result

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your request'


    else:
        return render_template('update.html', user_request=user_request)


def top_predictions(model, input, num_of_resolutions):
    predictions = model.predict_proba(input)[0]
    top_indices = predictions.argsort()[::-1][:num_of_resolutions]
    answers = []
    c=1
    for i in top_indices:
        answers.append(str(c) + ". " + model.classes_[i] + ": " + str(int(float(predictions[i])*100)) + "%")
        c+=1
    return answers


def combine_raw_text(text):
    result = ''
    for i in range(len(text)):
        if text[i] in (None, ""):
            text[i] = '-'
        elif text[i - 1] == '-':
            result += ' ' + text[i]
        result += text[i]
    return result


def encode_user_input(vectorizer, raw_text):
    combined_text = combine_raw_text(raw_text)
    return vectorizer.transform(combined_text)


def unpickle_and_split_pipeline(picklepath='pipeline.pkl'):
    pipe = pickle.load(open(picklepath, 'rb'))
    classifier = pipe['classifier']
    vectorizer = pipe['vectorizer']
    return classifier, vectorizer


def input_to_result(list, classifier, vectorizer):
    text_to_vectorize = combine_raw_text(list)
    vectorized_text = vectorizer.transform([text_to_vectorize])
    predictions = top_predictions(classifier, vectorized_text, 3)
    return predictions


# debugger mode
if __name__ == "__main__":
    app.run(debug=True)
