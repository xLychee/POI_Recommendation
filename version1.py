import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
df = pd.read_csv('./data/checkins.txt', names=['user_id','tweet_id','lat','lon','date','place_id','tags'],sep='\t');
users = df.user_id.unique();
pois = df.place_id.unique();

print 'Number of sample users = ' + str(users.shape[0]) + ' | Number of sample_pois = ' + str(pois.shape[0]);


from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df,test_size=0.25)

train_data = pd.DataFrame(train_data);
test_data = pd.DataFrame(test_data);

# Create training and test matrix
userids = train_data['user_id']
placeids = train_data['place_id']
data_1 = np.ones(len(train_data))
R = sps.coo_matrix((data_1,(userids,placeids)))
#print userids;

# Predict the unknown places through the dot product of the latent features for users and pois

def prediction(P,Q):
    return np.dot(P,Q);

# # Calculate hit percentage
# def calHit(P,Q,I,R):
#     preR = np.dot(P.T,Q);
#     preR[ I * preR >= 0.5] = 1;
#     preR[ I * preR < 0.5] = 0;
#     return np.sum((I * (R - preR)) ** 2) / len(I[I > 0]);

lmbda = 0.1 # Regularisation weight
k = 15 # Dimension of the latent feature space
m, n = R.shape # Number of users and pois
n_epochs = 100 # Number of epochs
gamma = 0.01 # Learning rate

P = np.random.rand(m,k); # Latent user feature matrix
Q = np.random.rand(k,n); # Latent poi feature matrix

def cal(i,j,k,P,Q):
    PT_ij = 0;
    for c in range(k):
        PT_ij += P[i,c] * Q[c,j];
    return PT_ij;


# Calculate the RMSE
def rmse(R,P,Q):
    rmse = 0;
    # find nonzero index
    row_array, col_array, value = sps.find(R);
    for i in range(len(row_array)):
        pre = cal(row_array[i],col_array[i],k,P,Q);
        rmse += np.square((value[i] - pre));
    rmse = np.sqrt(rmse);
    return rmse;

train_errors = []
# test_errors = []

# only consider non-zro matrix
row_array,col_array,value_array = sps.find(R);
validData = np.array([row_array,col_array,value_array])
validData = validData.T

for epoch in xrange(n_epochs):
    validData=np.random.permutation(validData)
    row_array=validData[:,0]
    col_array=validData[:,1]
    value_array=validData[:,2]

    rmse_ep = 0;
    for c in range(len(row_array)):
        # Extract none-zero value
        i = row_array[c];
        j = col_array[c];
        value = value_array[c];
        e = value - np.dot(P[i,:],Q[:,j]);  # Calculate error for gradient
        new_P = np.zeros((1,k));
        new_Q = np.zeros((k, 1));
        for d in range(k):
            new_P[0, d] = P[i,d] - gamma * e * (-Q[d, j])
            new_Q[d, 0] = Q[d,j] - gamma * e * (-P[i, d])
        P[i,:] = new_P;
        Q[:,j] = new_Q.T;

    # Update the RMSE
    rmse_ep = rmse(R,P,Q);
    train_errors.append(rmse_ep);


# Check performance by plotting train and test errors
print train_errors;
plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
# plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
plt.title('SGD-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()


# Calculate prediction matrix R_hat (low-rank approximation for R)
# R = pd.DataFrame(R)
# R_hat=pd.DataFrame(prediction(P,Q))
#
# # Compare true ratings of user 17 with predictions
# ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
# ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
# ratings.columns = ['Actual Rating', 'Predicted Rating']
# ratings
#
#
# def calHit(P,Q,I,T):
#     preR = np.dot(P.T,Q);
#     preR[ preR >= 0.5] = 1;
#     preR[ preR < 0.5] = 0;
#     return 1 - np.sum((I * (T - preR)) ** 2) / len(I[I > 0]);
#
#
# print calHit(P,Q,I2,T);
#


