% Apply EM algorithm to scatter plots representing sets of 2-D Gaussians
figure;
axis([-6,6,-5,5]);
imax = 20;
LL=zeros(imax);
for i=1:imax
    iter(i)=i*2;
    % generate 2-D dataset
    mu1=[1 -2.5]; sigma1=[2 0; 0 .4];
    mu2=[2 1];    sigma2=[0.5 0; 0 1.5];
    mu3=[-2 1];   sigma3=[1 -0.5; -0.5 1];
    mu4=[-3 0];   sigma4=[0.09 0; 0 0.09];
    rng('default'); % ensures same random seed every time
    x = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,200); ...
        mvnrnd(mu3,sigma3,200); mvnrnd(mu4,sigma4,80)];
    [mu, sigma, pi, ll] = GMM(x, 4, iter(i));
    L = ll;
    LL(i)=L;
    % present the mixture of Gaussians using meshgrid
    [UU,VV] = meshgrid(-10:0.01:10,-10:0.01:10);
    u = UU(:); v = VV(:); s = size(UU,1);
    pdf = mvnpdf([u v],mu(1,:),sigma(:,:,1));
    gauss1 = reshape(pdf,s,s);
    pdf = mvnpdf([u v],mu(2,:),sigma(:,:,2));
    gauss2 = reshape(pdf,s,s);
    pdf = mvnpdf([u v],mu(3,:),sigma(:,:,3));
    gauss3 = reshape(pdf,s,s);
    pdf = mvnpdf([u v],mu(4,:),sigma(:,:,4));
    gauss4 = reshape(pdf,s,s);
    mix = pi(1)*gauss1 + pi(2)*gauss2 + pi(3)*gauss3 ...
        + pi(4)*gauss4;
    % produce the contour plot
    if rem(iter(i),6)==0
        if     iter(i)==6, rr=1; pp= 0.1; qq=0;
        elseif iter(i)==12, rr=2; pp=-0.1; qq=0;
        elseif iter(i)==18, rr=3; pp= 0.1; qq=0.06;
        elseif iter(i)==24, rr=4; pp=-0.1; qq=0.06;
        elseif iter(i)==30, rr=5; pp= 0.1; qq=0.12;
        elseif iter(i)==36, rr=6; pp=-0.1; qq=0.12;
        end
        gca=subplot(3,2,rr);
        p=get(gca,'position');
        p(1)=p(1)+pp; p(2)=p(2)+qq;
        set(gca,'position',p)
        scatter(gca,x(:,1),x(:,2),3,'s','filled','b'); hold on
        contour(gca,UU,VV,mix,[0.005 0.01 0.02 0.03 0.04 ...
            0.05 0.07 0.1 0.15 0.25 0.40],'r');
        axis([-6,6,-5,5]); pbaspect([4,3,1]); axis on
    end % if
end % for
saveas(gcf,'EMquadruple.tif');
% saveas(gcf,'EMquadruple','epsc'); % gives too low a resolution
print('EMquadruple','-depsc2','-r300') % -r is followed by no. of dpi
% print -depsc2 -tiff -r300 -painters 'EMquadruple.eps' % alternative
% drawing a convergence plot for the EM algorithm
figure;
i=1:imax;
plot(iter(i),LL(i),'b'); grid off
saveas(gcf,'EMquadrupleconv.tif')
saveas(gcf,'EMquadrupleconv','epsc')