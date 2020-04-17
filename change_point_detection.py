import numpy as np
import csv
import matplotlib as pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

array1=[]
array2=[]
array3=[]
array4=[]
with open('final_aruba.txt','r') as csv_file:
    #csv.register_dialect('strip', skipinitialspace=True)
    csv_reader = csv.DictReader(csv_file,delimiter=' ')

    line_counter=0
    for line in csv_reader:
        if line['state']=='begin' or line['state']=='end':
            array4.append(1)
        else:
            array4.append(0)
        array1.append(float(line['sensor']))
        array2.append(float(line['value']))
        array3.append(float(line['label']))

print(len(array1))
print(len(array2))
print(len(array3))
print(len(array4))
w, h = 4, 1719551
X = [[1 for a in range(w)] for b in range(h)]
for i in range (h):
    X[i][0]=array1[i]
    X[i][1]=array2[i]
    X[i][2]=array3[i]
    X[i][3]=array4[i]

##############################################################################################
# X is the required data matrix
result=[]
f=open("dotproduct_index_value31908-68.txt","w+")
f1=open("cpd_dotproduct_index_value31908-68.txt","w+")
f2=open("noncpd_dotproduct_index_value31908-68.txt","w+")
model = Sequential()
from keras.models import load_model
model =  load_model('/content/aruba_50_bidirectional_lstm_model.h5')
for z in range(31908,31968):
    wlength=50
    w, h = 3, wlength;
    seq1= [[1 for a in range(w)] for b in range(h)]
    seq2= [[1 for a in range(w)] for b in range(h)]
    for i in range(wlength):
        seq1[i][0]=X[z-i][0]
        seq1[i][1]=X[z-i][1]
        seq1[i][2]=X[z-i][2]
        seq2[i][0]=X[z+i][0]
        seq2[i][1]=X[z+i][1]
        seq2[i][2]=X[z+i][2]
    #print(np.shape(seq1))
    #print(np.shape(seq2))
    #seq1= np.array(seq1)
    #seq2= np.array(seq2)

    data = [[[0 for e in range (3)] for a in range(wlength)] for b in range(2)]
    data [0]= seq1
    data [1]= seq2

    data = np.asarray(data)
    yhat = model.predict(data, verbose=0)
    #print(np.shape(yhat))
    res=0
    for i in range(12):
        res=res+yhat[0][i]*yhat[1][i]
    #print(res)
    f.write(str(res))
    f.write("\n")
    #result.append(res)
    if X[z][3]==1:
        f1.write(str(res))
	  f1.write("\n")
    else:
        f2.write(str(res))
        f2.write("\n")
#print(result)
f.close()
f1.close()
f2.close()


