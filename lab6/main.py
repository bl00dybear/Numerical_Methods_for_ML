import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

def normal_distribution(mu, sigma):
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_vals = stats.norm.pdf(x_vals, mu, sigma)
    
    plt.plot(x_vals, y_vals)
    plt.title(f'Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

def comp_form(x_val, mu, sigma):
    prob_formula = (1 / (math.sqrt(2*math.pi*sigma**2))) * math.exp(-(x_val-mu)**2 / (2*sigma**2))
    prob_scipy = stats.norm.pdf(x_val, mu, sigma)
    
    print(f"\n{f'Formula (mu={mu}, sigma={sigma})':<50}  {prob_formula:.4e}")
    print(f"{f'Scipy (mu={mu}, sigma={sigma})':<50}  {prob_scipy:.4e}")
    

def likelihood(data, mu, sigma):
    likelihoods = stats.norm.pdf(data, mu, sigma)
    total_likelihood = np.prod(likelihoods)

    print(f"\n{f'Total likelihood (mu={mu}, sigma={sigma})':<50}  {total_likelihood:.4e}")

    return total_likelihood

def priori_mean_deviation(mu, sigma):
    prior_mu = stats.norm.pdf(mu, 100, 50)
    prior_sigma = 1/(70-1) * (1<=sigma<=70)

    print(f"\n{f'Media apriori (mu={mu})':<50}  {prior_mu:.4e}")
    print(f"{f'Deviatia standard apriori (sigma={sigma})':<50}  {prior_sigma:.4e}")
    
    return prior_mu*prior_sigma

def posterior_prob(data, mu, sigma):
    likelihood_ = likelihood(data, mu, sigma)
    prior = priori_mean_deviation(mu, sigma)
    posterior = likelihood_*prior

    print(f"\n{f'Prob posterior (mu={mu}, sigma={sigma})':<50}  {posterior:.4e}")
    
    return posterior

def grid_srch(data):
    print("\n Grid search:\n")
    mus = [70, 75, 80, 85, 90, 95, 100]
    sigmas = [5, 10, 15, 20]
    
    best_model = None
    max_posterior = -1
    
    for mu in mus:
        for sigma in sigmas:
            posterior = posterior_prob(data, mu, sigma)
            
            if posterior > max_posterior:
                max_posterior = posterior
                best_model = (mu, sigma)
                
    print(f"\n{f'Best model (mu={best_model[0]}, sigma={best_model[1]})':<50}  {max_posterior:.4e}")


def main():
    data = [82, 106, 120, 68, 83, 89, 130, 92, 99, 89]
    mu_init = 90
    sigma_init = 10
    
    normal_distribution(mu_init, sigma_init)
    
    comp_form(82, mu_init, sigma_init)

    posterior_prob(data, mu_init, sigma_init)
    
    grid_srch(data)

if __name__ == "__main__":
    main()