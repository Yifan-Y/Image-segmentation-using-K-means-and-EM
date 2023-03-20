% EM algorithm for road picture - using greyscale only
% Prepare the ground exactly as for the histogram-based K-means algorithm

% obtain initial approximation to the solution 
% do {
%   apply M-step to estimate responsibilities
%   apply E-step to update Gaussian parameters and mixture coefficients 
% } until log likelihood shows sufficient convergence

% initializing parameters
frame=imread('road.png');
red=frame(:,:,1); green=frame(:,:,2); blue=frame(:,:,3);
seqr=reshape(red,1,320*240);
seqg=reshape(green,1,320*240);
seqb=reshape(blue,1,320*240);
intensr=uint8(0); intensg=uint8(0); intensb=uint8(0);
seq2r=zeros(1,320*240); seq2r=uint8(seq2r);
seq2g=zeros(1,320*240); seq2g=uint8(seq2g);
seq2b=zeros(1,320*240); seq2b=uint8(seq2b);