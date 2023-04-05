% K-means for road picture - greyscale
% Segmention without histogram, not efficient though

frame=imread('road.png'); % import target photo
grey=rgb2gray(frame); % convert from color to greyscale
imwrite(grey,'roadgrey.png'); % save the grescale version image
seq=reshape(grey,1,320*240); % sequence each pixel to a 1 * 320*240 matrix
intens=uint8(0);
seq2=zeros(1,320*240); seq2=uint8(seq2);

% segmentation of a greyscale image using the K-means algorithm
K = 6; % suppose k == 6, target number of clusters
mm = [10 20 60 160 220 250]; % starting values, initial centroids
imax = 10; % 10 iterations are set to meet accuracy
sq=zeros(1,K);
summ=zeros(1,K); num=zeros(1,K);

% pass 1: find the closest centroid and recalculate the centroid
for iter=1:imax
    summ(:)=0; num(:)=0;
    for i=1:320*240
        intens = seq(i);
        sq(:) = (double(intens)-mm(:)).^2; % find the euclidean distance to each centroid
        [mink,kk] = min(sq); % find minimum value of the distance and index
        summ(kk) = summ(kk) + double(intens);
        num(kk) = num(kk) + 1;
    end 
    mm = summ ./ num; % calculate and update the new centroids
end % pass 1

% pass 2 : assign each data point to the closet of the new cluster center positions
for i=1:320*240 
    intens = seq(i);
    sq(:) = (double(intens)-mm(:)).^2; %find the euclidean distance to each new centroid
    [mink,kk] = min(sq); % find index of minimum value (closed centroid)
    seq2(i) = mm(kk); % assign the index of closet centroid to each pixel
end % pass 2
KMout=reshape(seq2,240,320);
figure;
imshow(KMout);
imwrite(KMout,'roadclasses.bmp');