from app import *
import csv



def accuracy_top(num):
    model, vectorizer = unpickle_and_split_pipeline()
    results=[0]*num
    total=0
    with open('reorganized.csv') as inputData:
        inputReader = csv.reader(inputData)
        next(inputReader, None)
        for row in inputReader:
            total+=1
            input = vectorizer.transform([row[0]])
            predictions = top_predictions(model, input, num)
            for i in range(len(predictions)):
                if row[1] in predictions[i]:
                    results[i]+=1
        for j in range(len(results)):
            results[j]=results[j]/total
        print(results)

accuracy_top(5)
