import pandas as pd

df = pd.read_excel('TestData.xlsx', na_filter=False)

finalOutput = []

for index, row in df.iterrows():
    rowValue = row
    k = 1
    while True:
        rowLine = "Resolution " + str(k)
        # print(rowLine)
        if (rowLine in row) and (row[rowLine] != ""):
            finalOutput.append(index)
            k += 1
        else:
            break

print(finalOutput)
