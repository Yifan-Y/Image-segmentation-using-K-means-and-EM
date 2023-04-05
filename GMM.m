function [mu, sigma, pi, ll] = GMM(X, K, imax)
% X is the input data matrix with N rows (samples) and D columns (features)
% K is the desired number of components (clusters) in the GMM
[N, D] = size(X);

% Initialize the GMM parameters
mu = randn(K, D);
sigma = repmat(eye(D), [1, 1, K]);
pi = ones(1, K) / K;

% Compute the initial log-likelihood
K = size(mu, 1);
ll = 0;
for k = 1:K
    ll = ll + pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
end
ll = -sum(log(ll));

% Iterate until convergence
tolerance = 1e-6;
for iter = 1:imax
    % Expectation step: compute the responsibilities of each component for each data point
    K = size(mu, 1);
    resp = zeros(size(X, 1), K);
    for k = 1:K
        resp(:, k) = pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
    end
    resp = resp ./ sum(resp, 2);
    
    % Maximization step: update the parameters of each component using the responsibilities
    [N, D] = size(X);
    K = size(resp, 2);

    nk = sum(resp, 1);
    pi = nk / N;
    mu = resp' * X ./ nk';
    sigma = zeros(D, D, K);
    for k = 1:K
        X_c = X - mu(k, :);
        sigma(:, :, k) = (X_c' * (X_c .* resp(:, k))) / nk(k);
    end
    
    % Compute the log-likelihood of the updated model
    K = size(mu, 1);
    resp_new = 0;
    for k = 1:K
        resp_new = resp_new + pi(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
    end
    ll_new = -sum(log(resp_new));
    
    % Check for convergence, if satisfied, stop.
    if abs(ll_new - ll) < tolerance
        break;
    end
    
    ll = ll_new;
end
end