% K-means for road picture - greyscale
% Segmention without histogram, not efficient though
frame=imread('road.png');
grey=rgb2gray(frame);
imwrite(grey,'roadgrey.png');
seq=reshape(grey,1,320*240);
intens=uint8(0);
seq2=zeros(1,320*240); seq2=uint8(seq2);
K = 6;
mm = [10 20 60 160 220 250]; % starting values
imax = 10;
sq=zeros(1,K);
summ=zeros(1,K); num=zeros(1,K);
for iter=1:imax % pass 1
    summ(:)=0; num(:)=0;
    for i=1:320*240
        intens = seq(i);
        sq(:) = (double(intens)-mm(:)).^2;
        [mink,kk] = min(sq); % find index of minimum value
        summ(kk) = summ(kk) + double(intens);
        num(kk) = num(kk) + 1;
    end 
    mm = summ ./ num;
end % pass 1
for i=1:320*240 % pass 2
    intens = seq(i);
    sq(:) = (double(intens)-mm(:)).^2;
    [mink,kk] = min(sq); % find index of minimum value
    seq2(i) = mm(kk);
end % pass 2
KMout=reshape(seq2,240,320);
figure;
imshow(KMout);
imwrite(KMout,'roadclasses.bmp');