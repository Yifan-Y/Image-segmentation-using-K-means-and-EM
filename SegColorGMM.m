% Load the color image
im = imread('road.png');

% Convert the image to RGB color space
rgb_im = im2double(im); % rgb_im: 240*320*3 matrix

% Reshape the image into a 2D array
rgb_im_2d = reshape(rgb_im, [], 3); % rgm_im_2d: 76800*3 matix

% Fit a Gaussian mixture model with 6 components, iteration set as 100
K = 6;
iter = 100;
[mu, sigma, pi, ll, labels] = GMM(rgb_im_2d, K, iter);

% Reshape the labels back into the original image dimensions with color
segmented_im = zeros(size(im));

for i = 1:3
    % Extract the i-th channel (R, G, or B)
    channel = rgb_im(:,:,i);
    % Reshape the channel into a 1D array
    channel_1d = reshape(channel, [], 1);
    % Assign the segmented labels to the channel values
    channel_1d(labels==1) = mean(channel_1d(labels==1));
    channel_1d(labels==2) = mean(channel_1d(labels==2));
    channel_1d(labels==3) = mean(channel_1d(labels==3));
    channel_1d(labels==4) = mean(channel_1d(labels==4));
    channel_1d(labels==5) = mean(channel_1d(labels==5));

    % Reshape the channel back into the original image dimensions
    segmented_im(:,:,i) = reshape(channel_1d, size(im,1), size(im,2));
end

% Display the segmented image
imshow(segmented_im);
% Convert the segmented image back to RGB color space
segmented_im = im2uint8(segmented_im);
