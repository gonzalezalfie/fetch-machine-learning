import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data.

data = pd.read_csv("data_daily.csv")
data = data.rename(columns = {"# Date":"date", 
                              "Receipt_Count":"receipt_count"})

# Get monthly data.

def get_montly_data(data):
    n = data.date.shape[1-1]
    data = data.assign(
        month = [data.date[i][1-1:7] for i in range(n)]
    )
    
    data = data.groupby(["month"]). \
        agg(receipt_count = ("receipt_count", "sum")). \
        reset_index()
    
    return(data)
    
monthly_data = get_montly_data(data)

# Model funtion: y = beta0+beta1*t+beta2*x_(t-1)

def f(beta, x):
    return(beta[0]+beta[1]*x.t+beta[2]*x.x)

# These functions normalize and denormalize data.

def normalize(u, train):
    return((u-train.mean())/train.std())

def denormalize(u, mean, std):
    return(mean+std*u)

# This function updates beta in the gradient descent process.

def update_beta(x, y, f, beta_0, eta):
    n = len(y)
    e = y-f(beta_0, x)

    dbeta0 = -1/n*np.sum(e)
    dbeta1 = -1/n*np.dot(x.t, e)
    dbeta2 = -1/n*np.dot(x.x, e)
    
    beta_1 = np.array([beta_0[0]-eta*dbeta0, 
                       beta_0[1]-eta*dbeta1, 
                       beta_0[2]-eta*dbeta2])
    
    return(beta_1)

# Gradient descent algorithm

def gradient_descent(x, y, f, eta, n_epochs):
    n = len(y)
    
    #beta_1 = np.random.rand(2)
    beta_1 = np.array([0, 0, 0])
    
    X = pd.DataFrame({
        "t" : normalize(x.t, x.t), 
        "x" : normalize(x.x, x.x), 
        })
    
    y_norm = normalize(y, y)

    loss = [0]*n_epochs

    for i in range(1, n_epochs+1):
        
        beta_1 = update_beta(X, y_norm, f, beta_1, eta)
        
        e = y_norm-f(beta_1, X)
        
        loss[i-1] = 1/(2*n)*np.dot(e, e)
    
    res = {"beta":beta_1, "loss":loss}
    
    return(res)

# Set t, x and y.

t = np.arange(2, 12+1)
x = monthly_data.receipt_count[:-1].values
y = monthly_data.receipt_count[2-1:].values

# Create data frame to train the model.

X = pd.DataFrame({
    "t" : t, 
    "x" : x, 
    })

# Calculate beta using gradient descent.

eta = 0.1
n_epochs = 1000

model = gradient_descent(X, y, f, 
                        eta = eta, n_epochs = n_epochs)

beta = model["beta"]

# Plot loss convergence.

loss = model["loss"]

plt.figure(figsize = (16, 9))

g = sns.lineplot(x = np.arange(1, n_epochs + 1), y = loss, 
                 linewidth = 2)
g.axhline(loss[-1], color = "red", linestyle = "--")

g.set_xlabel("epoch")
g.set_ylabel("lmse")
g.set_title("Loss convergence")

g.get_figure().savefig("loss_convergence.png", bbox_inches = "tight")

# This function predicts h steps ahead.

def predict(beta, x, y, h):
    
    X = pd.DataFrame({
        "x" : normalize(x.x, x.x)
        })
    
    pred = [0]*h
    
    for i in range(h):
        if i == 1-1:
            t = normalize(x.t.iloc[-1]+i+1, x.t)
            pred[i] = beta[0]+beta[1]*t+beta[2]*X.x.iloc[-1]
        else:
            t = normalize(x.t.iloc[-1]+i+1, x.t)
            u = normalize(denormalize(pred[i-1], y.mean(), y.std()), x.x)
            pred[i] = beta[0]+beta[1]*t+beta[2]*u
    
    pred = np.array(pred)
    
    res = denormalize(pred, y.mean(), y.std())
    
    return(np.rint(res).astype("int"))

# Get predictions for 2022.

pred = predict(beta, X, y, h = 12)

# Gather monthly data of 2021 and predictions for 2022.

months = list(monthly_data.month)+ \
    list([month.replace("2021", "2022") for month in monthly_data.month])

df = pd.DataFrame({
        "month":months, 
        "receipt_count":np.hstack((monthly_data.receipt_count, pred)), 
        "year": ["2021" if i <= 12 else "2022" for i in range(1, 24+1)]
    })
df

df.to_csv("forecast.csv", index = False)

# Plot data and store it into a jpg file.

plt.figure(figsize = (16, 9))

g = sns.lineplot(df, x = "month", y = "receipt_count", hue = "year")
plt.setp(g.get_xticklabels(), rotation = 90);

g.set_title("Monthly scanned receipts")

g.get_figure().savefig("forecast.png", bbox_inches = "tight")

