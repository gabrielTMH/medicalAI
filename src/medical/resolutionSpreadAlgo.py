import pandas as pd
import csv
df = pd.read_excel('TestData.xlsx', na_filter=False)

output = []

Rules= {'afc':'AFC',
        'filament':'filament ps',
        'cold deck':'cold deck',
        'bmag':'bmag power supply',
        'transducer':'flow switch',
        'hot deck': 'hot deck',
        'ion chamber': 'ion chamber',
        'pulse': 'gun pulse driver',
        'dqing': 'thyratron grid control pcb',
        'flow switch': 'flow switch'}
def combine_features_from_data(inputFile='example1.csv',outputFile='exmple2.csv',):
  with open(outputFile, 'w') as outputData:
    with open(inputFile) as inputData:
        outputData.write('issue' + ',')
        outputData.write('label' + '\n')
        inputReader = csv.reader(inputData)
        next(inputReader, None)
        for row in inputReader:
          for i in range(2,11):
              if  row[i] in (None, ""):
                print(row[i])
                row[i]='-'
              outputData.write(row[i])
          outputData.write(',')
          outputData.write(row[12]+'\n')

def modifyResolution(string):
    string=string.lower()
    if string[0] =='w' and string[1].isdigit():
        return 'Cable'
    if 'leaf' in string and not 'mlc' in string:
        return 'Leaf Drive Train'
    if 'mlc' in string and not 'leaf' in string:
        return 'MLC Motor'
    for rule in Rules.keys():
        if rule in string:
            string = Rules[rule]
    return string

def swap_subsystem_with_problem(dataset):
    dataset['Sub System'],dataset['Problem'] = dataset['Problem'],dataset['Sub System']
    dataset = dataset.rename(columns={'Problem': 'Sub System', 'Sub System': 'Problem'})
    return dataset


def resolutionSpreadAlgo():
    for index, row in df.iterrows():
        otherInfo = row[:"Error Code 3"].to_dict()
        otherInfo["Resolution 1"] = ""
        rowValue = row
        k = 1
        while True:
            rowLine = "Resolution " + str(k)
            if (rowLine in row) and (row[rowLine] != ""):
                finalRowInfo = dict(otherInfo)
                resolution = row[rowLine]
                modified_resolution = modifyResolution(resolution)
                finalRowInfo['Resolution 1'] = resolution
                finalRowInfo['Modified_Resolution'] = modified_resolution
                output.append(finalRowInfo)
                k += 1
            else:
                break
    finalOutput = pd.DataFrame(output)
    finalOutput = finalOutput[finalOutput.duplicated(subset='Modified_Resolution', keep=False)]
    finalOutput = swap_subsystem_with_problem(finalOutput)
    finalOutput.to_csv('TestDataUpdated.csv')