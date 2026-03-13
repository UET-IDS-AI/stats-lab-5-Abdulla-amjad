import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.

    P(a<X<b) = F(b) - F(a)
             = (1 - e^(-lam*b)) - (1 - e^(-lam*a))
             = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    pA = 0.3
    pB = 0.7

    likelihood_A = gaussian_pdf(time, 40, 4)
    likelihood_B = gaussian_pdf(time, 45, 4)

    numerator = likelihood_B * pB
    denominator = likelihood_A * pA + likelihood_B * pB

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    pA = 0.3
    pB = 0.7

    # sample classes
    classes = np.random.choice(['A','B'], size=n, p=[pA,pB])

    samples = np.zeros(n)

    # generate values
    samples[classes=='A'] = np.random.normal(40,4,np.sum(classes=='A'))
    samples[classes=='B'] = np.random.normal(45,4,np.sum(classes=='B'))

    # approximate likelihood near the time value
    eps = 0.5
    mask = (samples > time-eps) & (samples < time+eps)

    if np.sum(mask) == 0:
        return 0

    return np.sum(classes[mask]=='B') / np.sum(mask)
