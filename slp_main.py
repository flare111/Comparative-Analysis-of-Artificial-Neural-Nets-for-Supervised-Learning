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

np.random.seed(0)
indices = np.arange(train_inputs.shape[0])
np.random.shuffle(indices)

print train_inputs.shape


net = nl.net.newp([[-7, 7]]*6, 1)
print train_outputs.shape
print net.co
print train_outputs


# train with delta rule
# see net.trainf
error = net.train(train_inputs[indices], train_outputs[indices], epochs=100, show=10, lr=0.1)

pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
# pl.show()
pl.savefig("slp.png")

out = net.sim(test_inputs)

# print out
predicted_outputs = []
for i in out:
    predicted_outputs.append(int(i[0]))

#
print "Training Complete, Test Results Generated for Single Layer Perceptron"
# print predicted_outputs
#
atest_outputs = []
#
for i in test_outputs:
    atest_outputs.append(i[0])

target_names = ['0 - Non Defaulter','1 - Defaulter']

print "Classification Report for Single Layer Perceptron"
cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
report_file = open("Classification_Report_slp.txt", 'a')
report_file.write("Classification Report For Single Layer Perceptron\n")
report_file.write(cfreport)
report_file.write("\n\n\n")
report_file.close()
