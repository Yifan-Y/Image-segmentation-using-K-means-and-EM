% K-means for road picture – colour
% Genetalize to RGB for color images
frame=imread('road.png');
red=frame(:,:,1); green=frame(:,:,2); blue=frame(:,:,3);
seqr=reshape(red,1,320*240);
seqg=reshape(green,1,320*240);
seqb=reshape(blue,1,320*240);
intensr=uint8(0); intensg=uint8(0); intensb=uint8(0);
seq2r=zeros(1,320*240); seq2r=uint8(seq2r);
seq2g=zeros(1,320*240); seq2g=uint8(seq2g);
seq2b=zeros(1,320*240); seq2b=uint8(seq2b);
K = 8;
mmr=zeros(1,K); mmg=zeros(1,K); mmb=zeros(1,K);
% image sampling points (u,v)
u=[30 30 30 160 160 160 230 230];
v=[30 130 230 60 130 230 30 130];
for k=1:K
    mmr(k)=frame(u(k),v(k),1); % starting values
    mmg(k)=frame(u(k),v(k),2); 
    mmb(k)=frame(u(k),v(k),3);
end 
sq=zeros(1,K);
summr=zeros(1,K); summg=zeros(1,K); summb=zeros(1,K); num=zeros(1,K);
for i=1:320*240 % pass 1
    intensr = seqr(i); intensg = seqg(i); intensb = seqb(i);
    sq(:) = (double(intensr)-mmr(:)).^2 + ...
        (double(intensg)-mmg(:)).^2 + (double(intensb)-mmb(:)).^2;
    [mink,kk] = min(sq); % find index of minimum value
    summr(kk) = summr(kk) + double(intensr);
    summg(kk) = summg(kk) + double(intensg);
    summb(kk) = summb(kk) + double(intensb);
    num(kk) = num(kk) + 1;
end % pass 1
mmr = summr ./ num; mmg = summg ./ num; mmb = summb ./ num;
for i=1:320*240 % pass 2
    intensr = seqr(i); intensg = seqg(i); intensb = seqb(i);
    sq(:) = (double(intensr)-mmr(:)).^2 + ...
        (double(intensg)-mmg(:)).^2 + (double(intensb)-mmb(:)).^2;
    [mink,kk] = min(sq); % find index of minimum value
    seq2r(i) = mmr(kk); seq2g(i) = mmg(kk); seq2b(i) = mmb(kk);
end % pass 2
KMout(:,:,1)=reshape(seq2r,240,320); % red;
KMout(:,:,2)=reshape(seq2g,240,320); % green;
KMout(:,:,3)=reshape(seq2b,240,320); % blue;
figure;
imshow(KMout);
imwrite(KMout,'roadclasses.png');