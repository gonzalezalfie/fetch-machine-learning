# fetch-machine-learning
This repository has the necessary files for the Fetch Rewards Take-home Exercise.

We have a time series of the daily number of scanned receipts for the year of 2021. Our goal is to predict the approximate number of the scanned receipts for each month of 2022.

# Our approach

To predict the monthly number of scanned receipts, we will do the following:

1. Aggregate the data into monthly data.
2. Define a linear regression model to make the predictions.

# The model

Let $Y_t$ be the monthly number of scanned receipts. Let $f_\beta(t, \ X_t) = \beta_0+\beta_1t+\beta_2X_{t-1}$.

Our model will be the following:

$$Y_t = f_\beta(t, \ X_t)+e_t$$

where $t = 1, \ 2, \ \dots, \ N$.

We want to minimize the following loss function:

$$L(\beta) = \frac{1}{2N}\sum_{t=1}^N e_t^2 = \frac{1}{2N}\sum_{t = 1}^N (Y_t - f_\beta(t, \ X_t))^2$$

We will use gradient descent to get the values of $\beta_0$, $\beta_1$ and $\beta_2$ that minimize the function.

According to the algorithm, we update each $\beta_j$ a number of epochs $n$ with a learning rate $\eta$ as follows:

$$\beta_j^{(i+1)} = \beta_j^{(i)}-\eta\frac{\partial L(\beta^{(i)})}{\partial\beta_j^{(i)}}$$

The partial derivatives of each $\beta_j$ are the following:

$$\frac{\partial L(\beta)}{\partial\beta_0} = -\frac{1}{N}\sum_{t = 1}^N e_t$$

$$\frac{\partial L(\beta)}{\partial\beta_1} = -\frac{1}{N}\sum_{t = 1}^N te_t$$

$$\frac{\partial L(\beta)}{\partial\beta_2} = -\frac{1}{N}\sum_{t = 1}^N X_te_t$$

We will run the algorithm using a docker image that will execute the `model.py` file included in this repostitory. If you want to see in detail how gradient descent estimates the values of $\beta_j$ and how it predicts the future values of the series, you can read the comments in the file

Our algorithm will use a value of $\eta$ equal to 0.1 and a number of epochs $n$ equal to 1000.

# Run the solution

I'm assuming that the user can use a Linux terminal (preferably Ubuntu) and has a relatively new version of docker installed.

# Install Git (if you don't have it)

Run this commands:

`sudo apt-get update`

`sudo apt-get install git`

Type "yes" or "y" if needed.

# Clone the repository

Preferably, but not necessesarily, go to the `home` directory:

`cd`

Clone the repository:

`git clone https://github.com/gonzalezalfie/fetch-machine-learning.git`

# Build the docker image

Got to the repository directory:

`cd fetch-machine-learning`

Build the docker image:

`sudo docker build -t fetch-machine-learning .`

# Run the container

To run the container, run this command:

`sudo docker run --rm -v $(pwd):/home/ubuntu fetch-machine-learning`

Once the container finishes its execution, run 

`ls`

and you should be able to see three files: `loss_convergence.png`, `forecast.csv` and `forecast.png`.

`loss_convergence.png` shows how the loss function changes as the number of epochs increases. We would expect the loss function to converge to some value. That's an indicator that the algorithm reached a minimum.

`forecast.csv` has the historic monthly values of 2021 as wells as the predicted values for 2022.

`forecast.png` shows the plot of the historic and predicted values.

# Check the results

To see the values of the predicted values for 2022, run this command:

`cat forecast.csv`

If you have GUI or if you have a share folder with your Linux system, have a look at the `loss_convergence.png` and `forecast.png` to check the convergence of the algorithm and how the predicted values look compared to the historic values.

And that's it!
