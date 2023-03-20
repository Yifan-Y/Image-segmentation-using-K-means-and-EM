% Generate some data
data = [randn(100,2)+5; randn(100,2)-5];

% Set the number of components
num_components = 2;

% Initialize the model parameters
mu = randn(num_components, size(data,2));
sigma = repmat(eye(size(data,2)), [1,1,num_components]);
w = ones(num_components,1)/num_components;

% Run the EM algorithm
for iter = 1:50
    % E-step
    for k = 1:num_components
        pdf(:,k) = mvnpdf(data, mu(k,:), squeeze(sigma(:,:,k)));
    end
    pdf_w = pdf .* w';
    pdf_w_sum = sum(pdf_w, 2);
    pdf_w_norm = pdf_w ./ repmat(pdf_w_sum, [1,num_components]);
    
    % M-step
    for k = 1:num_components
        w(k) = mean(pdf_w_norm(:,k));
        mu(k,:) = sum(repmat(pdf_w_norm(:,k), [1,size(data,2)]) .* data) / sum(pdf_w_norm(:,k));
        sigma(:,:,k) = (data-repmat(mu(k,:), [size(data,1),1]))' * ...
            (repmat(pdf_w_norm(:,k), [1,size(data,2)]) .* (data-repmat(mu(k,:), [size(data,1),1]))) ...
            / sum(pdf_w_norm(:,k));
    end
end

% Plot the results
figure; hold on;
scatter(data(:,1), data(:,2), '.');
for k = 1:num_components
    plot_gaussian_ellipsoid(mu(k,:), squeeze(sigma(:,:,k)), 1, 30);
end