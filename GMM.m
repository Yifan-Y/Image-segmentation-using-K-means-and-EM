function [mu, Sigma, w] = GMM(X, K, max_iters, tol)

% X: NxD matrix of input data
% K: number of mixture components
% max_iters: maximum number of iterations to run the algorithm
% tol: tolerance level for convergence

[N, D] = size(X);

% Random initialization of the parameters
mu = randn(K, D);
Sigma = repmat(eye(D), [1, 1, K]);
w = ones(K, 1) / K;

% Initialize the log-likelihood
prev_ll = -Inf;

for iter = 1:max_iters
    
    % E-step: compute responsibilities
    r = zeros(N, K);
    for k = 1:K
        r(:, k) = w(k) * mvnpdf(X, mu(k, :), squeeze(Sigma(:, :, k)));
    end
    r = r ./ sum(r, 2);
    
    % M-step: update parameters
    Nk = sum(r, 1);
    for k = 1:K
        mu(k, :) = sum(repmat(r(:, k), 1, D) .* X, 1) / Nk(k);
        Sigma(:, :, k) = ((X - mu(k, :))' * (repmat(r(:, k), 1, D) .* (X - mu(k, :)))) / Nk(k);
        w(k) = Nk(k) / N;
    end
    
    % Compute the log-likelihood and check for convergence
    ll = sum(log(sum(bsxfun(@times, r, w'), 2)));
    if ll - prev_ll < tol
        break;
    end
    prev_ll = ll;
end

end

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
function [p, mu, sigma, log_likelihood] = gaussian_mix_model(data, num_components, max_iter)
% Inputs:
%   data: the data matrix, where each row is a data point
%   num_components: the number of Gaussian components in the mixture model
%   max_iter: the maximum number of iterations for the EM algorithm
% Outputs:
%   p: the mixture coefficients for the Gaussian components
%   mu: the mean vectors for the Gaussian components
%   sigma: the covariance matrices for the Gaussian components
%   log_likelihood: the log-likelihood of the data under the fitted model

% Initialize the parameters
[num_samples, num_features] = size(data);
p = ones(num_components, 1) / num_components;
mu = randn(num_components, num_features);
sigma = repmat(eye(num_features), [1, 1, num_components]);

% Run the EM algorithm
log_likelihood = -inf;
for iter = 1:max_iter
    % E-step: calculate the posterior probability of each component given each data point
    pdf = zeros(num_samples, num_components);
    for k = 1:num_components
        pdf(:, k) = mvnpdf(data, mu(k, :), squeeze(sigma(:, :, k)));
    end
    likelihood = pdf * p;
    post_prob = bsxfun(@rdivide, bsxfun(@times, pdf, p'), likelihood');
    
    % M-step: update the parameters
    Nk = sum(post_prob, 1);
    p = Nk / num_samples;
    for k = 1:num_components
        mu(k, :) = sum(bsxfun(@times, data, post_prob(:, k)), 1) / Nk(k);
        sigma(:, :, k) = bsxfun(@times, bsxfun(@minus, data, mu(k, :))', post_prob(:, k)) * bsxfun(@minus, data, mu(k, :)) / Nk(k);
    end
    
    % Calculate the log-likelihood
    log_likelihood_new = sum(log(likelihood));
    if abs(log_likelihood_new - log_likelihood) < 1e-6
        break;
    else
        log_likelihood = log_likelihood_new;
    end
end
