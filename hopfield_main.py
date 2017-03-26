import neurolab as nl
import numpy as np
import pylab as pl
import csv
import math
from sklearn.metrics import classification_report
import sys
adata = []

with open('dataset_full.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        adata.append(row)

xdata_zeros = []
ydata_zeros = []

xdata_ones = []
ydata_ones = []

for i in range(0, len(adata)):
    temp = []
    yval = int(adata[i][len(adata[i]) - 1])
    if yval ==1:
        for j in range(0, len(adata[i]) - 1):
            temp.append(int(adata[i][j]))
        xdata_ones.append(temp)
        ydata_ones.append([yval])
    else:
        for j in range(0, len(adata[i]) - 1):
            temp.append(int(adata[i][j]))
        xdata_zeros.append(temp)
        ydata_zeros.append([yval])

nzeros = len(xdata_zeros)
nones = len(xdata_ones)

# print (nzeros)
# print (nones)

zeros_div = int(math.floor(nzeros * 0.75 - 1))
ones_div = int(math.floor(nones * 0.75 - 1))

train_inputs_m = np.array(xdata_zeros[0:zeros_div] + xdata_ones[0:ones_div])
train_outputs = np.array(ydata_zeros[0:zeros_div] + ydata_ones[0:ones_div])
test_inputs_m = np.array(xdata_zeros[zeros_div:nzeros]+xdata_ones[ones_div:nones])
test_outputs = ydata_zeros[zeros_div:nzeros]+ydata_ones[ones_div:nones]

train_inputs = train_inputs_m[:, [5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [5, 6, 7, 8, 9, 10]]

print train_inputs.shape
print train_outputs.shape
print test_inputs.shape

target = np.asfarray(train_inputs)

# Create and train network
net = nl.net.newhop(target)

thinks_pattern = net.sim(test_inputs)
# print("Test on train samples:")
# for i in range(len(target)):
#     print(train_outputs[i], (output[i] == target[i]).all())

# print thinks_pattern
notfound = False
predicted_outputs = []
for i in range(0,len(thinks_pattern)):
    row_thinks = thinks_pattern[i]
    for j in range(0,len(target)):
        row_actual = target[j]
        # print row_thinks,
        # print " | ",
        # print row_actual
        if(np.all(row_thinks == row_actual)):
            print "Pattern Found .. | Count = " + str(len(predicted_outputs))
            out_thinks = train_outputs[j][0]
            predicted_outputs.append(out_thinks)
            notfound = False
            break
        else:
            notfound = True
    if(notfound):
        print "No Pattern Found For : " + str(row_thinks)
        predicted_outputs.append(2)

print predicted_outputs

atest_outputs = []

for i in test_outputs:
    atest_outputs.append(i[0])

target_names = ['0 - Non Defaulter', '1 - Defaulter', '2 - Unpredictable']

print "Classification Report For Hopfield Networks"
cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
report_file = open("Classification_Report.txt", 'a')
report_file.write("Classification Report For Hopfield Networks")
report_file.write(cfreport)
report_file.write("\n")
report_file.close()




