function [mu, sigma, pi, ll, labels] = GMM(X, K, imax)
% Input:
% X is the input data matrix with N rows (data ponts) and D columns (dimensions)
% K is the number of culsters of gaussian mixture models
% imax is the maximum number of iterations

% Main algorithm:
% 1. Initialize pi_k, mu_k, sigma_k
% 2. Evaluate the initial value of the log likelihood
% 3. Expectaion step: Calculate the responsibilities r_nk 
%    (p of k mixture generated n data points)
% 4. Maximization step: Re-estimate and update the parameters: mu_new,
%    sigma_new and pi_new.
% 5. Evaluate the log likelihood, check if convergence criterion is
%    satisfied, if not, go back to step 3.

% Initialize the GMM parameters
[N, D] = size(X); 
mu = datasample(X, K, 'Replace', true);
sigma = repmat(eye(D), [1, 1, K]);
pi = ones(1, K) / K;

% Compute the initial log likelihood
K = size(mu, 1);
ll = 0;
for k = 1:K
    ll = ll + pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
end
ll = sum(log(ll));

% Iterate until convergence
tolerance = 1e-6;

for iter = 1:imax
    % Expectation step: compute the responsibilities of each component for each data point
    K = size(mu, 1);
    r_nk = zeros(size(X, 1), K);
    for k = 1:K
        r_nk(:, k) = pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
    end
    r_nk = r_nk ./ sum(r_nk, 2);
    
    % Maximization step: update the parameters of each component using the responsibilities
    K = size(r_nk, 2);
    nk = sum(r_nk, 1);
    pi = nk / N;
    mu = r_nk' * X ./ nk';
    sigma = zeros(D, D, K);
    for k = 1:K
        X_c = X - mu(k, :);
        sigma(:, :, k) = (X_c' * (X_c .* r_nk(:, k))) / nk(k);
    end
    
    % Compute the log-likelihood of the updated model
    K = size(mu, 1);
    r_nk_new = 0;
    for k = 1:K
        r_nk_new = r_nk_new + pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
    end
    ll_new = sum(log(r_nk_new));
    
    % Check for convergence, if satisfied, stop.
    if abs(ll_new - ll) < tolerance || iter >= imax
        break;
    end
    
    ll = ll_new;
end
[~,labels] = max(r_nk,[],2);
end