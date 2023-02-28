import pandas as pd

df = pd.read_excel('TestData.xlsx', na_filter=False)

output = []

for index, row in df.iterrows():
    otherInfo = row[:"Error Code 3"].to_dict()
    otherInfo["Resolution 1"] = ""
    rowValue = row
    k = 1
    while True:
        rowLine = "Resolution " + str(k)
        if (rowLine in row) and (row[rowLine] != ""):
            finalRowInfo = dict(otherInfo)
            finalRowInfo['Resolution 1'] = row[rowLine]
            output.append(finalRowInfo)
            k += 1
        else:
            break

finalOutput = pd.DataFrame(output)
finalOutput.to_csv('TestDataUpdated.csv')
print(finalOutput)
