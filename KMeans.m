function [centroids, labels, distortion] = KMeans(data, k, max_iter)
% Inputs:
%   data: the data matrix, where each row is a data point
%   k: the number of clusters
%   max_iter: the maximum number of iterations for the algorithm
% Outputs:
%   centroids: the cluster centroids
%   labels: the cluster labels for each data point
%   distortion: the sum of squared distances of each data point to its nearest centroid

% Initialize the centroids
[~, num_features] = size(data);
centroids = randn(k, num_features);

% Run the algorithm
for iter = 1:max_iter
    % Assign each data point to its nearest centroid
    dist = pdist2(data, centroids);
    [~, labels] = min(dist, [], 2);
    
    % Update the centroids
    for i = 1:k
        centroids(i, :) = mean(data(labels == i, :), 1);
    end
    
    % Calculate the distortion
    distortion = sum(min(dist, [], 2).^2);
end


%%
function [centers, assignments, objective] = k_means(data, k, max_iter)
% Inputs:
%   data: the data matrix, where each row is a data point
%   k: the number of clusters
%   max_iter: the maximum number of iterations for the algorithm
% Outputs:
%   centers: the centroids of the clusters
%   assignments: the cluster assignments for each data point
%   objective: the value of the objective function (sum of squared distances)

% Initialize the centroids randomly
[num_samples, num_features] = size(data);
perm = randperm(num_samples);
centers = data(perm(1:k), :);

% Run the k-means algorithm
assignments = zeros(num_samples, 1);
objective = inf;
for iter = 1:max_iter
    % Assign each data point to the closest centroid
    dists = pdist2(data, centers);
    [~, assignments_new] = min(dists, [], 2);
    
    % Update the centroids
    for j = 1:k
        centers(j, :) = mean(data(assignments_new == j, :), 1);
    end
    
    % Calculate the objective function
    objective_new = sum(sum((data - centers(assignments_new, :)).^2));
    if abs(objective_new - objective) < 1e-6
        break;
    else
        objective = objective_new;
        assignments = assignments_new;
    end
end
