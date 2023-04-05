frame = imread('road.png');
grey = rgb2gray(frame); % convert from color to greyscale
imwrite(grey,'roadgrey.png'); % save greyscale version of input image
seq = reshape(grey,1,320*240); % greyscale image is sequenced in a 1-D array (1*76800)
hh = zeros(1,256); % initiate a 1 by 256 matrix of zero (1*256)
hh = uint16(hh); % convert each element in the matrix to unsigned 16 bit integer
intens = uint8(0);
for i = 1:320*240
    intens = seq(i) + 1; % histogram index needs to be 1-256, not 0-255, so coverting by adding 1
    hh(intens)=hh(intens)+1;
end 

% segmentation of a greyscale image using the K-means algorithm
dhh = double(hh); % convert the intensity histogram matrix to double precision
K = 6; % Suppose K == 6, target number of clusters
mm = [10 20 60 160 220 250]; % starting values, initial centroids
imax = 10; % 10 iterations are set to meet accuracy
sq=zeros(1,K);
summ=zeros(1,K); num=zeros(1,K);
MM=zeros(1,imax);

% pass 1: find the closest centroid and recalculate the centroid
for iter=1:imax  
    summ(:)=0; num(:)=0; sumsq=0;
    for j=1:256
        sq(:) = (j-mm(:)).^2; % find the euclidean distance to each centroid
        [mink,kk] = min(sq); % find minimum value of the distance and index
        summ(kk) = summ(kk)+dhh(j)*j; % sum of intensity * nums
        num(kk) = num(kk)+dhh(j);
        sumsq = sumsq+dhh(j)*mink;
    end 
    mm = summ ./ num; % calculate and update the new centroids
    MM(iter) = sqrt(sumsq); % save error information
end % pass 1

% assignment of classes to histogram bins
classk=zeros(1,256);
% pass 2 : assign each data point to the closet of the new cluster center positions
for j=1:256 
    sq(:) = (j-mm(:)).^2; %find the euclidean distance to each new centroid
    [mink,kk] = min(sq); % find index of minimum value
    classk(j)=kk; % assign class kk to histogram bin j
end % pass 2

seq2=zeros(1,320*240);
seq2=uint8(seq2);

for i=1:320*240 % final assignment of intensities to image points
    j=seq(i) + 1;
    kk=classk(j);
    intens=mm(kk) - 1;
    seq2(i)=intens;
end 
KMout=reshape(seq2,240,320); % re-form output image

figure; % show and save output image
imshow(KMout);
imwrite(KMout,'roadclasses.png');

% plot histogram
figure;
set(gca,'fontsize',11);
box on; hold on;
for i=1:K
    line([mm(i)-1,mm(i)-1],[0,0.025],'color','c','linewidth',1.0)
end 

j=1:256; % use j as a plot variable
plot(j-1,dhh/(320*240),'-r');
pbaspect([2 1 1]);
axis([0 255 0 0.025]);
saveas(gcf,'roadhist.tif')
saveas(gcf,'roadhist','epsc') % save eps version of color plot

% plot convergence
figure;
set(gca,'fontsize',11);
set(gca,'XTick',([ 1 2 3 4 5 6 7 8 9 10 ]))
set(gca,'YTick',([ 0 1000 2000 3000 4000 5000 6000 7000 ]))
grid on; box on; hold on
i=1:imax;
plot(i,MM,'r');
pbaspect([2 1 1]);
axis([1 10 0 7000]);
saveas(gcf,'roadconv.tif')
saveas(gcf,'roadconv','epsc') % save eps version of color plot