% EM algorithm for road picture - using greyscale only
frame=imread('road.png');
grey=rgb2gray(frame);
imwrite(grey,'roadgrey.png');
seq=reshape(grey,1,320*240);
hh=zeros(1,256);
hh=uint16(hh);
intens=uint8(0);
for i=1:320*240
    intens=seq(i) + 1;
    hh(intens)=hh(intens)+1;
end 
x=1:256;
dhh_org=double(hh);
Pmin=1;
while hh(Pmin)==0, Pmin=Pmin+1; end
found = find(hh==0);
hh(found) = [];
x(found) = []; % x now indicates non-zero weight in hh
P=length(hh); % use instead of 256
dhh=double(hh); % to revert to original intensities, add Pmin-1
x=x';
K = 6;
L=-300; old_L=L; dL=-1.0;
imax=100; % iterations needed to plot convergence
LL=zeros(1,imax); DD=zeros(1,imax);
% initialize Gaussian and mixture parameters
mixture = ones(1,K)/K;
mu=[30 60 90 120 160 220]; % starting values
mu=mu';
vari_k=ones(1,K)*10;
sigma_k=sqrt(vari_k);


% EM algorithm - main loop
iter=1;
while iter<=imax
    % E-step
    w = zeros(P,K);
    r = zeros(P,K); rx = zeros(P,K); rdd = zeros(P,K);
    for k = 1:K
    w(:,k) = mixture(k) * normpdf(x(:),mu(k),sigma_k(k)); 
    w(:,k) = w(:,k) .* dhh(:);
    end 
% find responsibilities
k_sum_w = sum(w,2);
for i = 1:P
    r(i,:) = w(i,:) ./ k_sum_w(i);
end 
% M-step
for i = 1:P
    r(i,:) = r(i,:) .* dhh(i);
end 
for i = 1:P
    rx(i,:) = r(i,:) .* x(i);
end 
    i_sum_r = sum(r,1);
    ik_sum_r = sum(sum(r,1),2);
    i_sum_rx = sum(rx,1);
    for k = 1:K
        % find new mixtures
        mixture(k) = i_sum_r(k) / ik_sum_r;
        % find new means
        mu(k) = i_sum_rx(k) ./ i_sum_r(k);
        % find new sigmas
        for i = 1:P
            dev = x(i) - mu(k);
            devsq = dev^2;
            rdd(i,:) = r(i,:) * devsq;
        end 
       i_sum_rdd = sum(rdd,1);
    vari_k(k) = i_sum_rdd(k) ./ i_sum_r(k); 
    sigma_k(k) = sqrt(vari_k(k));
    end 
% find log likelihood
for k = 1:K
    w(:,k) = mixture(k) * normpdf(x(:),mu(k),sigma_k(k));
end 
    k_sum_w = sum(w,2); % sum over k
    W(:) = dhh(:) .* log(k_sum_w(:));
    L = sum(W); % sum over i=1:P
    LL(iter) = L - old_L; % change in L
    old_L = L;
    % find sum of absolute errors
    diff=k_sum_w'-dhh/(320*240);
    DD(iter)=sum(abs(diff));
    iter=iter+1;
end % iteration loop

% find borderlines between Gaussians
border=zeros(1,K-1);
x=0;
for k=1:K-1
    y1=0; y2=-1;
    while y2<y1
        y1=mixture(k) * normpdf(x,mu(k)-1,sigma_k(k));
        kk=k+1;
        y2=mixture(kk) * normpdf(x,mu(kk)-1,sigma_k(kk));
        x=x+1;
    end
    border(k)=x-1;
end 
% plot histogram
figure;
set(gca,'fontsize',11);
box on; hold on
j=1:256;
plot(j-1,dhh_org/(320*240),'g');
pbaspect([2 1 1]);
for k=1:K-1
    line([border(k)-1,border(k)-1],[0,0.025],'color','c',...
    'linewidth',0.5)
end
for k = 1:K
    y = mixture(k) * normpdf(j,mu(k)-1,sigma_k(k));
    plot(j-1,y,'r');
end 
y = 0;
for k = 1:K
    y = y + mixture(k) * normpdf(j,mu(k)-1,sigma_k(k));
end 
plot(j-1,y,'b');
axis([0 255 0 0.025]);
saveas(gcf,'roadhist.tif')
saveas(gcf,'roadhist','epsc')
% plot convergence
figure;
set(gca,'fontsize',11);
grid on; box on; hold on
ax=zeros(1,imax);
i=1:imax;
plot(i,ax,'b');
plot(i,LL,'r');
plot(i,(DD-DD(imax))*1800,'m'); % adjusted to be seen on the same scale
pbaspect([5 3 1]);
axis([4 imax -4 12]);
saveas(gcf,'roadconv.tif')
saveas(gcf,'roadconv','epsc')

% show classified intensities in final image
seq2=zeros(1,320*240);
seq2=uint8(seq2);
for i=1:320*240
    intens=seq(i);
    if intens<border(1), intens=mu(1);
    elseif intens<border(2), intens=mu(2);
    elseif intens<border(3), intens=mu(3);
    elseif intens<border(4), intens=mu(4);
    elseif intens<border(5), intens=mu(5);
    else, intens=mu(6);
    end
    seq2(i)=intens - 1;
end 
greyout=reshape(seq2,240,320);
imwrite(greyout,'roadclasses.png');