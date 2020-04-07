import gzip
import numpy as np
import matplotlib.pyplot as plt
import math

def draw(arr):
    img = np.asarray(arr.reshape(28, 28)).squeeze()
    plt.imshow(img)
    plt.show()

fimg = gzip.open('train-images-idx3-ubyte.gz')
#Skip 4 bytes each of magic, nimg, nrow, ncol
fimg.read(16)

count = int(input("Input number of images to read(< 60000): "))
buf = fimg.read(28*28*count)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(count, 28, 28)
fimg.close()

flabel = gzip.open('train-labels-idx1-ubyte.gz')
#skip magic, nlabels
flabel.read(8)

labels = []
for i in range(count):
	buf = flabel.read(1)
	labels.append( np.frombuffer(buf, dtype=np.uint8).astype(np.int64) )
flabel.close()

mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
d = {}
for i in range(10):
    d[i] = []

#Storing sum in mean
for i in range(count):
	n = int(labels[i])
	d[n].append(data[i].reshape(784,1))
	mean[n] += data[i].reshape(784, 1)

#Calculating mean and storing in mean
for i in range(10):
	mean[i] = mean[i]/len(d[i])

#Showing mean images of each digit :)
for i in range(10):
	draw(mean[i])

#Show covariance matrix, relation b/w each of the 28*28 cells

eval_array = []
num = 0
while num<10:#Keep array a in loop later..
    a = []
    evals, evecs = [[], []]
    
    #d[num] has images of num in 784x1 format
    A = np.array([list(d[num][i].squeeze().T) for i in range(len(d[num]))])
    a = np.cov(A, rowvar = False)
    evals, evecs = np.linalg.eigh(a)
    L1 = max(evals)
    v1 = []
    for i in range(784):
        if evals[i] == L1:
            v1 = evecs[:,i]
            break
    v1 = v1.reshape(784, 1)
    '''
    draw(mean[num] - math.sqrt(L1)*v1)
    draw(mean[num])
    draw(mean[num] + math.sqrt(L1)*v1)'''
    a1 = (mean[num] - math.sqrt(L1)*v1).reshape(28,28)
    a2 = mean[num].reshape(28,28)
    a3 = (mean[num] + math.sqrt(L1)*v1).reshape(28,28)
    f,axarr = plt.subplots(nrows=1, ncols=3, figsize=(28,28))
    axarr[0].set_title('Mean - sl*ev')
    axarr[1].set_title('Mean')
    axarr[2].set_title('Mean + sl*ev')
    axarr[0].imshow(np.asarray(a1).squeeze())
    axarr[1].imshow(np.asarray(a2).squeeze())
    axarr[2].imshow(np.asarray(a3).squeeze())
    plt.savefig(str(num)+'.png', bbox_inches='tight')
    
    eval_array.append(evals)
    num+=1
    
f, axarr = plt.subplots(nrows = 2, ncols = 3, constrained_layout=True)
for i in range(5):
    x = i//3
    y = i%3
    axarr[x,y].set_title(str(i))
    axarr[x,y].plot([j for j in range(1, 785)], eval_array[i], 'o')
    if i == 4:
        plt.savefig('Graphs1.png')

f, axarr = plt.subplots(nrows = 2, ncols = 3, constrained_layout=True)
for i in range(5,10):
    x = (i-5)//3
    y = (i-5)%3
    axarr[x,y].set_title(str(i))
    axarr[x,y].plot([j for j in range(1, 785)], eval_array[i], 'o')
    if i == 9:
        plt.savefig('Graphs2.png')

