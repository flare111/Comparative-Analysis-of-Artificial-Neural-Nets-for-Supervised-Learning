import neurolab as nl
import numpy as np
import numpy as np
import pylab as pl
import csv
import math
from sklearn.metrics import classification_report
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

print train_inputs_m.shape
print train_outputs.shape
print test_inputs_m.shape


#TODO BACKWARD FEATURE SELECTION


#TODO STAGE 1
features_file_name = "features_012345678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 1 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[9999, 1000001]] + [[0,3]] + [[-1, 7]] + [[-1,4]] + [[20, 80]] + [[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()

pl.figure(1)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")





#TODO STAGE 2
features_file_name = "features_12345678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 2 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[0,3]] + [[-1, 7]] + [[-1,4]] + [[20, 80]] + [[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()


pl.figure(2)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")





#TODO STAGE 3
features_file_name = "features_2345678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 3 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [2, 3, 4, 5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[-1, 7]] + [[-1,4]] + [[20, 80]] + [[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()


pl.figure(3)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")






#TODO STAGE 4
features_file_name = "features_345678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 4 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [3, 4, 5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [3, 4, 5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[-1,4]] + [[20, 80]] + [[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()


pl.figure(4)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")






#TODO STAGE 5
features_file_name = "features_45678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 5 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [4, 5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [4, 5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[20, 80]] + [[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()


pl.figure(5)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")








#TODO STAGE 6
features_file_name = "features_5678910"
report_file = open("Classification_Report.txt", 'a')
report_file.write("\n ----------- STAGE 6 ------------- \n")
report_file.write(features_file_name)
report_file.write("\n")
report_file.close()
train_inputs = train_inputs_m[:, [5, 6, 7, 8, 9, 10]]
test_inputs = test_inputs_m[:, [5, 6, 7, 8, 9, 10]]

error = []

for hidden_nodes in xrange(2, 4):
    net = nl.net.newff([[-7, 7]] * 6, [hidden_nodes, 1], transf=[nl.trans.LogSig()] * 2)
    net.trainf = nl.train.train_rprop

    error.append(net.train(train_inputs, train_outputs, show=0, epochs=30))
    out = net.sim(test_inputs)

    predicted_outputs = []
    for i in out:
        if i<=0.5:
            predicted_outputs.append(0)
        else:
            predicted_outputs.append(1)

    print "Training Complete, Test Results Generated for " + str(hidden_nodes) + " Hidden Nodes"
    # print predicted_outputs

    atest_outputs = []

    for i in test_outputs:
        atest_outputs.append(i[0])
    # print "test_outputs"
    # print atest_outputs
    # print "predicted_outputs"
    # print predicted_outputs

    target_names = ['0 - Non Defaulter','1 - Defaulter']

    print "Classification Report For " + str(hidden_nodes) + " Hidden Nodes"
    cfreport = classification_report(atest_outputs, predicted_outputs, target_names=target_names)
    report_file = open("Classification_Report.txt", 'a')
    report_file.write("Classification Report For " + str(hidden_nodes) + " Hidden Nodes\n")
    report_file.write(cfreport)
    report_file.write("\n\n\n")
    report_file.close()

pl.figure(6)
for i in range(len(error)):
    labelname = str(i+2) + " Hidden Layer Nodes"
    pl.plot(range(len(error[i])), error[i], label=labelname)
pl.legend()
pl.xlabel("Epochs")
pl.ylabel("SS-Error")
pl.title("Plot")
# pl.show()
pl.savefig(features_file_name+".png")


