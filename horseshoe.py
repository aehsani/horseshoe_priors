import scipy.io
from scipy.special import expit
import numpy as np
from pystan import StanModel
from pystan.constants import MAX_UINT
import pickle

seed = 431994

# load data
np.random.seed(seed)
data = scipy.io.loadmat('colon.mat')
X, y = np.float64(data['X']), np.int64((data['Y'].ravel() + 1) / 2)
N = len(y)

# shuffle data
p = np.random.permutation(N)
X, y = X[p], y[p]

# train test split
Xtrain, ytrain = X[:N // 2], y[:N // 2]
Xtest, ytest = X[N // 2:], y[N // 2:]

# initialize data dict
data = dict(N=Xtrain.shape[0], M=Xtrain.shape[1], X=Xtrain, y=ytrain)

# Build STAN model

model_code = '''
data {
  int<lower=1> N; // Number of data
  int<lower=1> M; // Number of covariates
  matrix[N, M] X;
  int<lower=0,upper=1> y[N];
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> tau_tilde;
  real<lower=0> csquared;
  real alpha;
}

model {
  vector[M] lambda_tilde = sqrt(csquared * lambda .* lambda ./ (csquared + tau_tilde * tau_tilde * lambda .* lambda));
  vector[M] beta = beta_tilde .* lambda_tilde * tau_tilde;

  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 0.001);
  csquared ~ inv_gamma(2, 8);
  alpha ~ normal(0, 10);

  y ~ bernoulli_logit(X * beta + alpha);
}
'''

def examine(beta, threshold=0.1):
    N, M = beta.shape
    beta_mask = beta > threshold
    results = []
    for i in range(N):
        results.append(np.sum(beta_mask[i, :]))
    results = np.array(results)
    return results

def calc_mlpd(beta, alpha, X, y):
    p = expit(X @ beta + alpha)
    log_probs = np.log(p) * y + np.log(1. - p) * (1 - y)
    lp = np.mean(log_probs) 
    return lp


# Compile model from scratch
sm = StanModel(model_code=model_code)
with open('sm3.pkl', 'wb') as f:
    pickle.dump(sm, f)

# Read model from from file
# with open('sm3.pkl', 'rb') as f:
#     sm = pickle.load(f)

fit = sm.sampling(data=data, seed=seed, chains=6, algorithm='NUTS')
samples = fit.extract(permuted=True)
beta_tilde, lamb, tau_tilde, csquared, alphas= samples['beta_tilde'], samples['lambda'], samples['tau_tilde'], samples['csquared'], samples['alpha']
numer = (csquared * lamb.T * lamb.T)
denom = (csquared + tau_tilde * tau_tilde * lamb.T * lamb.T)
lambda_tilde = np.sqrt((numer / denom).T)
betas = (tau_tilde * (beta_tilde * lambda_tilde).T).T

best_mlpd = -1000000
best_beta = None
best_alpha = None

for beta, alpha in zip(betas, alphas):
    mlpd = calc_mlpd(beta, alpha, Xtrain, ytrain)
    if mlpd > best_mlpd:
        best_beta = beta
        best_alpha = alpha
        best_mlpd = mlpd

test_mlpd = calc_mlpd(best_beta, best_alpha, Xtest, ytest)

print('Train mlpd:', best_mlpd)
print('Test mlpd:',  test_mlpd)

examinations = examine(betas)

import ipdb
ipdb.set_trace()
