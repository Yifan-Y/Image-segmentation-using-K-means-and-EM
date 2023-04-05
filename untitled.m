%%
function [mu, sigma, w, loglikelihood] = gaussian_mixture_model(data, num_components, max_iter)
% Gaussian mixture model algorithm
%
% data: NxD data matrix (N data points, D dimensions)
% num_components: number of Gaussian components to fit
% max_iter: maximum number of iterations
%
% mu: KxD matrix of means (K components, D dimensions)
% sigma: DxDxK matrix of covariance matrices (K components)
% w: Kx1 vector of component weights (K components)
% loglikelihood: vector of log-likelihoods for each iteration

% Initialize the model parameters
[N, D] = size(data);
mu = randn(num_components, D);
sigma = repmat(eye(D), [1,1,num_components]);
w = ones(num_components,1)/num_components;

% Run the EM algorithm
loglikelihood = zeros(max_iter,1);
for iter = 1:max_iter
    % E-step
    for k = 1:num_components
        pdf(:,k) = mvnpdf(data, mu(k,:), squeeze(sigma(:,:,k)));
    end
    pdf_w = pdf .* w';
    pdf_w_sum = sum(pdf_w, 2);
    pdf_w_norm = pdf_w ./ repmat(pdf_w_sum, [1,num_components]);
    loglikelihood(iter) = sum(log(sum(pdf_w_norm, 2)));
    
    % M-step
    for k = 1:num_components
        w(k) = mean(pdf_w_norm(:,k));
        mu(k,:) = sum(repmat(pdf_w_norm(:,k), [1,D]) .* data) / sum(pdf_w_norm(:,k));
        sigma(:,:,k) = (data-repmat(mu(k,:), [N,1]))' * ...
            (repmat(pdf_w_norm(:,k), [1,D]) .* (data-repmat(mu(k,:), [N,1]))) ...
            / sum(pdf_w_norm(:,k));
    end
end

end

%%
function [mu, Sigma, w] = gaussian_mixture_model(X, K, max_iter)
% X: n-by-d matrix of data points
% K: number of mixture components
% max_iter: maximum number of iterations

% Initialize the parameters
[n, d] = size(X);
mu = X(randperm(n, K), :);  % randomly choose K data points as initial means
Sigma = repmat(eye(d), [1, 1, K]);  % initialize covariance matrices to identity matrices
w = ones(1, K) / K;  % initialize mixing coefficients to uniform distribution

% Run the EM algorithm
for iter = 1:max_iter
    % E-step
    p = zeros(n, K);
    for k = 1:K
        p(:, k) = w(k) * mvnpdf(X, mu(k, :), Sigma(:, :, k));
    end
    p_sum = sum(p, 2);
    p_norm = bsxfun(@rdivide, p, p_sum);
    
    % M-step
    w = mean(p_norm);
    for k = 1:K
        mu(k, :) = sum(bsxfun(@times, p_norm(:, k), X)) / sum(p_norm(:, k));
        Sigma(:, :, k) = bsxfun(@times, p_norm(:, k), bsxfun(@minus, X, mu(k, :)))' * bsxfun(@minus, X, mu(k, :)) / sum(p_norm(:, k));
    end
end


%%
function [mu, Sigma, pi, ll] = gmm(X, K)
% X is the input data matrix with N rows (samples) and D columns (features)
% K is the desired number of components (clusters) in the GMM

[N, D] = size(X);

% Initialize the GMM parameters
mu = randn(K, D);
Sigma = repmat(eye(D), [1, 1, K]);
pi = ones(1, K) / K;

% Compute the initial log-likelihood
ll = compute_log_likelihood(X, mu, Sigma, pi);

% Iterate until convergence
max_iter = 100;
tolerance = 1e-6;
for iter = 1:max_iter
    % Expectation step: compute the responsibilities of each component for each data point
    resp = compute_responsibilities(X, mu, Sigma, pi);
    
    % Maximization step: update the parameters of each component using the responsibilities
    [mu, Sigma, pi] = update_parameters(X, resp);
    
    % Compute the log-likelihood of the updated model
    ll_new = compute_log_likelihood(X, mu, Sigma, pi);
    
    % Check for convergence
    if abs(ll_new - ll) < tolerance
        break;
    end
    
    ll = ll_new;
end

function resp = compute_responsibilities(X, mu, Sigma, pi)
K = size(mu, 1);
resp = zeros(size(X, 1), K);
for k = 1:K
    resp(:, k) = pi(k) * mvnpdf(X, mu(k, :), Sigma(:, :, k));
end
resp = resp ./ sum(resp, 2);
end

function [mu, Sigma, pi] = update_parameters(X, resp)
[N, D] = size(X);
K = size(resp, 2);

nk = sum(resp, 1);
pi = nk / N;
mu = resp' * X ./ nk';
Sigma = zeros(D, D, K);
for k = 1:K
    X_c = X - mu(k, :);
    Sigma(:, :, k) = (X_c' * (X_c .* resp(:, k))) / nk(k);
end
end

function ll = compute_log_likelihood(X, mu, Sigma, pi)
K = size(mu, 1);
ll = 0;
for k = 1:K
    ll = ll + pi(k) * mvnpdf(X, mu(k, :), Sigma(:, :, k));
end
ll = sum(log(ll));
end
%%
function [means, covariances, priors, likelihood] = gaussian_mixture_model(data, num_clusters)

% Input:
% data: n x d matrix where each row is a data point of d dimensions
% num_clusters: number of Gaussian clusters

% Output:
% means: num_clusters x d matrix where each row is the mean of a Gaussian cluster
% covariances: d x d x num_clusters array where the ith 2D matrix is the covariance matrix of the ith Gaussian cluster
% priors: 1 x num_clusters matrix where each element is the prior probability of the corresponding Gaussian cluster
% likelihood: the log-likelihood of the fitted model

% Set the initial parameters of the model
[num_points, num_dimensions] = size(data);
means = datasample(data, num_clusters, 'Replace', false);
covariances = repmat(eye(num_dimensions), [1, 1, num_clusters]);
priors = repmat(1/num_clusters, [1, num_clusters]);

% Initialize some variables for the EM algorithm
likelihood_old = -Inf;
tolerance = 1e-6;
max_iterations = 100;
iteration = 0;
while true
    iteration = iteration + 1;

    % Expectation step
    likelihood = 0;
    responsibility = zeros(num_points, num_clusters);
    for i = 1:num_clusters
        covariance = squeeze(covariances(:, :, i));
        covariance_det = det(covariance);
        covariance_inv = inv(covariance);
        for j = 1:num_points
            point = data(j, :);
            likelihood_point = mvnpdf(point, means(i, :), covariance);
            responsibility(j, i) = priors(i) * likelihood_point / covariance_det^(0.5);
        end
    end
    likelihood = sum(log(sum(responsibility, 2)));
    responsibility = responsibility ./ sum(responsibility, 2);
    
    % Maximization step
    Nk = sum(responsibility, 1);
    for i = 1:num_clusters
        means(i, :) = sum(repmat(responsibility(:, i), [1, num_dimensions]) .* data, 1) / Nk(i);
        difference = data - repmat(means(i, :), [num_points, 1]);
        covariance = (difference' * (repmat(responsibility(:, i), [1, num_dimensions]) .* difference)) / Nk(i);
        covariances(:, :, i) = covariance;
        priors(i) = Nk(i) / num_points;
    end
    
    % Check for convergence
    if abs((likelihood - likelihood_old) / likelihood_old) < tolerance || iteration >= max_iterations
        break;
    end
    likelihood_old = likelihood;
end

end

%%
function [mu, sigma, alpha, ll] = nD_GMM(X, K, max_iter, tol)
% Fits an n-dimensional Gaussian mixture model to data X using EM algorithm.
% 
% INPUTS:
%   X: N-by-D matrix of N data points with D dimensions.
%   K: number of Gaussian components.
%   max_iter: maximum number of iterations for EM algorithm.
%   tol: tolerance for stopping criterion based on change in log likelihood.
% 
% OUTPUTS:
%   mu: K-by-D matrix of means for each Gaussian component.
%   sigma: D-by-D-by-K matrix of covariance matrices for each Gaussian component.
%   alpha: K-dimensional vector of mixing coefficients for each Gaussian component.
%   ll: log likelihood of data under the fitted Gaussian mixture model.

[N, D] = size(X);

% Initialize parameters randomly
mu = rand(K, D) * range(X) + min(X);
sigma = repmat(eye(D), [1, 1, K]);
alpha = ones(K, 1) / K;

ll = -Inf;
ll_prev = -Inf;

for iter = 1:max_iter
    % E-step: compute posterior probabilities of each point belonging to each component
    P = zeros(N, K);
    for k = 1:K
        P(:, k) = alpha(k) * mvnpdf(X, mu(k, :), sigma(:, :, k));
    end
    P_sum = sum(P, 2);
    P_norm = P ./ P_sum;
    
    % M-step: update parameters
    Nk = sum(P_norm, 1)';
    alpha = Nk / N;
    for k = 1:K
        mu(k, :) = sum(X .* P_norm(:, k), 1) / Nk(k);
        X_centered = X - mu(k, :);
        sigma(:, :, k) = (X_centered .* P_norm(:, k))' * X_centered / Nk(k) + 1e-6 * eye(D);
    end
    
    % Compute log likelihood of data
    ll = sum(log(P_sum));
    
    % Check for convergence
    if abs(ll - ll_prev) < tol
        break
    end
    ll_prev = ll;
end

end

%%
function [means, covs, weights, likelihoods] = gaussian_mixture_model(X, K, max_iters, tol)
% X - data matrix (N x D) where N is the number of observations and D is the number of dimensions
% K - number of components
% max_iters - maximum number of iterations
% tol - convergence tolerance

[N, D] = size(X);

% initialize means, covariances, and weights
means = zeros(K, D);
covs = zeros(D, D, K);
weights = ones(1, K) / K;

% randomly initialize means and covariances
for k = 1:K
    means(k, :) = X(randi(N), :);
    covs(:, :, k) = cov(X) + eye(D) * 1e-6; % add small value to diagonal for numerical stability
end

likelihoods = zeros(1, max_iters);

for iter = 1:max_iters
    % E-step: calculate responsibilities
    resp = zeros(N, K);
    for k = 1:K
        resp(:, k) = weights(k) * mvnpdf(X, means(k, :), covs(:, :, k));
    end
    resp = resp ./ sum(resp, 2);

    % M-step: update means, covariances, and weights
    Nk = sum(resp, 1);
    for k = 1:K
        means(k, :) = sum(resp(:, k) .* X) / Nk(k);
        covs(:, :, k) = (X - means(k, :))' * diag(resp(:, k)) * (X - means(k, :)) / Nk(k);
        weights(k) = Nk(k) / N;
    end

    % calculate log-likelihood
    likelihoods(iter) = sum(log(sum(resp, 2)));

    % check for convergence
    if iter > 1 && abs(likelihoods(iter) - likelihoods(iter-1)) < tol
        break;
    end
end

end