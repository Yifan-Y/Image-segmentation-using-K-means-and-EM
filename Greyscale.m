frame = imread('road.png');
grey = rgb2gray(frame);
imwrite(grey,'roadgrey.png'); % save greyscale version of input image
seq = reshape(grey,1,320*240);
hh = zeros(1,256);
hh = uint16(hh);
intens = uint8(0);
for i = 1:320*240
    intens = seq(i) + 1; % histogram index needs to be 1-256, not 0-255
    hh(intens)=hh(intens)+1;
end 
...