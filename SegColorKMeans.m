% K-means for road picture – colour
% Generalize to RGB for color images
frame=imread('road.png'); % map the image into a 240*320*3 array
red=frame(:,:,1); green=frame(:,:,2); blue=frame(:,:,3);
seqr=reshape(red,1,320*240); % the value of r for each pixel, save as a 1 * 320*240 matrix
seqg=reshape(green,1,320*240); % the value of g for each pixel, save as a 1 * 320*240 matrix
seqb=reshape(blue,1,320*240); % the value of b for each pixel, save as a 1 * 320*240 matrix
intensr=uint8(0); intensg=uint8(0); intensb=uint8(0);

% empty containers for the results of segmentation
seq2r=zeros(1,320*240); seq2r=uint8(seq2r);
seq2g=zeros(1,320*240); seq2g=uint8(seq2g);
seq2b=zeros(1,320*240); seq2b=uint8(seq2b);

% segmentation of a color image using the K-means algorithm
K = 8; % Suppose K == 8, target number of clusters
mmr=zeros(1,K); mmg=zeros(1,K); mmb=zeros(1,K);
% image sampling points (u,v)
u=[30 30 30 160 160 160 230 230];
v=[30 130 230 60 130 230 30 130];
for k=1:K
    mmr(k)=frame(u(k),v(k),1); % starting values, initial centroids
    mmg(k)=frame(u(k),v(k),2); 
    mmb(k)=frame(u(k),v(k),3);
end 
sq=zeros(1,K);
summr=zeros(1,K); summg=zeros(1,K); summb=zeros(1,K); num=zeros(1,K);

% pass 1: find the closest centroid and recalculate the centroid
for i=1:320*240 % for each pixel
    intensr = seqr(i); 
    intensg = seqg(i); 
    intensb = seqb(i);
    % find the euclidean distance to each centroid
    sq(:) = (double(intensr)-mmr(:)).^2 + ...
            (double(intensg)-mmg(:)).^2 + ...
            (double(intensb)-mmb(:)).^2; 
    [mink,kk] = min(sq); % find index of minimum value (closed centroid) -> kk
    summr(kk) = summr(kk) + double(intensr); 
    summg(kk) = summg(kk) + double(intensg);
    summb(kk) = summb(kk) + double(intensb);
    num(kk) = num(kk) + 1;
end % pass 1
% calculate the means and update the new centroids
mmr = summr ./ num; 
mmg = summg ./ num; 
mmb = summb ./ num;

% pass 2 : assign each data point to the closet of the new cluster center positions
for i=1:320*240 % for each pixel
    intensr = seqr(i); intensg = seqg(i); intensb = seqb(i);
    %find the euclidean distance to each new centroid
    sq(:) = (double(intensr)-mmr(:)).^2 + ...
            (double(intensg)-mmg(:)).^2 + ...
            (double(intensb)-mmb(:)).^2;
    [mink,kk] = min(sq); % find index of minimum value
    % assign the index of closest centroid to the containers of each pixel
    seq2r(i) = mmr(kk); 
    seq2g(i) = mmg(kk); 
    seq2b(i) = mmb(kk);
end % pass 2

% generate the new image
KMout(:,:,1)=reshape(seq2r,240,320); % red;
KMout(:,:,2)=reshape(seq2g,240,320); % green;
KMout(:,:,3)=reshape(seq2b,240,320); % blue;
figure;
imshow(KMout);
imwrite(KMout,'roadclasses.png');