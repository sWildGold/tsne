import numpy as np
import matplotlib.pyplot as plt
import time

def neg_squared_ecu_dists(X):
	sumX2=np.sum(X*X,1)
	dist= np.add(np.add(-2*np.dot(X,X.T),sumX2).T,sumX2)
	return -dist

def cal_Prob_P(dist,sigmas,option=None):

	if option is not None:
		dist=dist[option]/(2*sigmas*sigmas)
	else:
		dist=dist/((2*sigmas*sigmas).reshape([-1, 1]))
	E = np.exp(dist)
	if option is not None:
		E[option]=0.
		sumE=np.sum(E)
	else:
		np.fill_diagonal(E,0.)
		sumE = np.sum(E, 1).reshape([-1, 1])
	P = E / (sumE+1e-20)
	return P

def cal_Prob_P_joint(X,sigmas):
	dist=neg_squared_ecu_dists(X)
	P=cal_Prob_P(dist,sigmas)
	P_joint=(P+P.T)/(2*P.shape[0])
	return np.maximum(P_joint,2.5e-13)

def cal_Prob_Q(Y):
	dist=neg_squared_ecu_dists(Y)
	a=1./(1-dist)
	np.fill_diagonal(a,0.)
	Q=a/np.sum(a)
	return a,np.maximum(Q,1e-12)

def cal_perplexity(dist,sigmas,i):
	P=cal_Prob_P(dist,sigmas,i)
	P=P[np.concatenate((np.r_[0:i],np.r_[i+1:P.size]))]
	H=-np.sum(P*np.log(P))
	Perplexity=np.exp(H)
	return Perplexity

def cal_optim_sigmas(X,perplexity,tol=1e-5):
	sigmas=np.zeros(X.shape[0])
	lower = 1e-5
	upper = 1000
	dist = neg_squared_ecu_dists(X)
	for i in range(X.shape[0]):
		lower_i=lower
		upper_i=upper
		mid=1
		cal_Perp=cal_perplexity(dist,mid,i)
		times=1

		while np.abs(cal_Perp-perplexity) > tol:
			if cal_Perp>perplexity:
				upper_i=mid
				mid=(lower_i+upper_i)/2
			else:
				lower_i = mid
				if upper_i==upper:
					mid*=2
				else:
					mid=(lower_i+upper_i)/2
			cal_Perp=cal_perplexity(dist,mid,i)
			times+=1
		sigmas[i]=mid
	return sigmas

def pca(X, no_dims=50):

	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:, 0:no_dims])
	return Y.real

perplexity=20
eta=500
initial_momentum=0.5
final_momentum=0.8
min_gain = 0.01
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
X=pca(X)
Y=np.random.randn(X.shape[0],2)
start=time.time()
sigmas=cal_optim_sigmas(X,perplexity)
stop=time.time()-start
print(stop)

P=cal_Prob_P_joint(X,sigmas)*4
iY=np.zeros([X.shape[0],2])
dY = np.zeros((X.shape[0],2))
gains=np.ones([X.shape[0],2])
for t in range(1000):
	num,Q = cal_Prob_Q(Y)
	if(t==100):
		P/=4
	PQ = P - Q
	for i in range(X.shape[0]):
		dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (2, 1)).T * (Y[i, :] - Y), 0)

	if t%10==0:
		cost=np.sum(P*np.log(P/(Q+1e-20)))
		print("%d:"%t,cost)

	if t<20:
		momentum=initial_momentum
	else:
		momentum=final_momentum
	iY=momentum*iY-eta*dY
	Y+=iY
	Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))
	a=1

Y_5=Y[labels[:]==5]
Y_1=Y[labels[:]==1]
Y_11=Y_1[:,0]
Y_12=Y_1[:,1]
Y_51=Y_5[:,0]
Y_52=Y_5[:,1]
plt.figure()
plt.scatter(Y_51,Y_52,marker="^")
plt.scatter(Y_11,Y_12,marker="v")
plt.show()


