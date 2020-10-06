clc;
close all;
clear all;
%% Import
% Make shure that the files are in the same folder as the script or add the
% full path to them
addpath('bm3d_matlab_package');
im = imread('The_Subject.jpeg');
img = rgb2gray(im);


%% Noise Introducction to the images
gaussed = imnoise(img, 'gaussian');
gaussed_2 = imnoise(img, 'gaussian', 0, 0.1);
gaussed_3 = imnoise(img, 'gaussian', 0, 0.8);

poissoned = imnoise(img, 'poisson'); %basic poisson noise
poissoned_2 = imnoise(img, 'poisson');
poissoned_3 = imnoise(img, 'poisson');

% %for a mid range poisson noise the image will go through 21 times
% for i = 1:20
%     poissoned_2 = imnoise(poissoned_2, 'poisson');
%     poissoned_3 = imnoise(poissoned_3, 'poisson');  
% end 
% 
% %for a high level poisson noise the image will go through 151 times (20 + 130)
% for i = 1:80
%     poissoned_3 = imnoise(poissoned_3, 'poisson'); 
% end
% 
% save("pois2.mat",'poissoned_2')
% save("pois3.mat",'poissoned_3')

load('pois2.mat');
load('pois3.mat');

impulsed = imnoise(img, 'salt & pepper');
impulsed_2 = imnoise(img, 'salt & pepper', 0.3);
impulsed_3 = imnoise(img, 'salt & pepper', 0.8);


%% Denoising (imgaussfilt, medfilt2, imfilter)
P = 1.8;
Pp1 = 1.7;
Pp = 1.7;%
Pm = 1.27;
Pm2 =0.9;
% gaussian filtering
gaussian_impulse_noise1 = imgaussfilt(impulsed, Pp1);
gaussian_impulse_noise2 = imgaussfilt(impulsed_2, P);
gaussian_impulse_noise3 = imgaussfilt(impulsed_3, P);

gaussian_gaussian_noise1 = imgaussfilt(gaussed, Pm);
gaussian_gaussian_noise2 = imgaussfilt(gaussed_2, P);
gaussian_gaussian_noise3 = imgaussfilt(gaussed_3, P);

gaussian_poisson_noise1 = imgaussfilt(poissoned, Pm2);
gaussian_poisson_noise2 = imgaussfilt(poissoned_2, Pp);
gaussian_poisson_noise3 = imgaussfilt(poissoned_3, P);

% Median filtering

median_impulse_noise1 = medfilt2(impulsed);
median_impulse_noise2 = medfilt2(impulsed_2);
median_impulse_noise3 = medfilt2(impulsed_3);

median_gaussian_noise1 = medfilt2(gaussed);
median_gaussian_noise2 = medfilt2(gaussed_2);
median_gaussian_noise3 = medfilt2(gaussed_3);

median_poisson_noise1 = medfilt2(poissoned);
median_poisson_noise2 = medfilt2(poissoned_2);
median_poisson_noise3 = medfilt2(poissoned_3);

% Average filtering
windowWidth = 5;
kernel = ones(windowWidth) / windowWidth .^ 2;
windowWidth = 8;
kernel8 = ones(windowWidth) / windowWidth .^ 2;
windowWidth = 11;
kernel11 = ones(windowWidth) / windowWidth .^ 2;
windowWidth = 3;
kernel3 = ones(windowWidth) / windowWidth .^ 2;
windowWidth = 7;
kernel7 = ones(windowWidth) / windowWidth .^ 2;

average_impulse_noise1 = imfilter(impulsed, kernel, 'replicate');
average_impulse_noise2 = imfilter(impulsed_2, kernel8, 'replicate');
average_impulse_noise3 = imfilter(impulsed_3, kernel, 'replicate');

average_gaussian_noise1 = imfilter(gaussed, kernel, 'replicate');
average_gaussian_noise2 = imfilter(gaussed_2, kernel, 'replicate');
average_gaussian_noise3 = imfilter(gaussed_3, kernel, 'replicate');

average_poisson_noise1 = imfilter(poissoned, kernel3, 'replicate');
average_poisson_noise2 = imfilter(poissoned_2, kernel11, 'replicate');
average_poisson_noise3 = imfilter(poissoned_3, kernel7, 'replicate');


%% BM3D Denoising
y = im2double(img);


%gauss
z = im2double(gaussed);
y_gauss = BM3D(z, sqrt(0.05));

z2 = im2double(gaussed_2);
y_gauss2 = BM3D(z2, sqrt(0.5));%0.5

z3 = im2double(gaussed_3);
y_gauss3 = BM3D(z3, sqrt(0.8));%0.3


%poisson

z = im2double(poissoned);
y_poi = BM3D(z, sqrt(0.16));%0.16

z2 = im2double(poissoned_2);
y_poi2 = BM3D(z2, sqrt(0.5));%0.5

z3 = im2double(poissoned_3);
y_poi3 = BM3D(z3, sqrt(0.9));%0.9


%salt & peper
z = im2double(impulsed);
y_est = BM3D(z, sqrt(0.16));%0.16

z2 = im2double(impulsed_2);
y_est2 = BM3D(z2, sqrt(0.5));%0.5

z3 = im2double(impulsed_3);
y_est3 = BM3D(z3, sqrt(0.9));%0.9



%% PSNR BM3D

%gauss
 ngauss_1 = psnr( gaussed, img);
 ngauss_2 = psnr( gaussed_2, img);
 ngauss_3 = psnr( gaussed_3, img);
 
 dgauss_1 = psnr( y_gauss, y);
 dgauss_2 = psnr( y_gauss2, y);
 dgauss_3 = psnr( y_gauss3, y);
 
 
%poisson
 npoi_1 = psnr( poissoned, img);
 npoi_2 = psnr( poissoned_2, img);
 npoi_3 = psnr( poissoned_3, img);
 
 dpoi_1 = psnr( y_poi, y);
 dpoi_2 = psnr( y_poi2, y);
 dpoi_3 = psnr( y_poi3, y);
 
%salt
 nsalt_1 = psnr( impulsed, img);
 nsalt_2 = psnr( impulsed_2, img);
 nsalt_3 = psnr( impulsed_3, img);
 
 dsalt_1 = psnr( y_est, y);
 dsalt_2 = psnr( y_est2, y);
 dsalt_3 = psnr( y_est3, y);
 
 %% PSNR Linear
peaksnr_impulse_noise1 = psnr(impulsed, img);
peaksnr_impulse_noise2 = psnr(impulsed_2, img);
peaksnr_impulse_noise3 = psnr(impulsed_3, img);

peaksnr_gaussian_noise1 = psnr(gaussed, img);
peaksnr_gaussian_noise2 = psnr(gaussed_2, img);
peaksnr_gaussian_noise3 = psnr(gaussed_3, img);

peaksnr_poisson_noise1 = psnr(poissoned, img);
peaksnr_poisson_noise2 = psnr(poissoned_2, img);
peaksnr_poisson_noise3 = psnr(poissoned_3, img);

% Gaussian
peaksnr_gaussian_impulse_noise1 = psnr(gaussian_impulse_noise1, img);
peaksnr_gaussian_impulse_noise2 = psnr(gaussian_impulse_noise2, img);
peaksnr_gaussian_impulse_noise3 = psnr(gaussian_impulse_noise3, img);

peaksnr_gaussian_gaussian_noise1 = psnr(gaussian_gaussian_noise1, img);
peaksnr_gaussian_gaussian_noise2 = psnr(gaussian_gaussian_noise2, img);
peaksnr_gaussian_gaussian_noise3 = psnr(gaussian_gaussian_noise3, img);

peaksnr_gaussian_poisson_noise1 = psnr(gaussian_poisson_noise1, img);
peaksnr_gaussian_poisson_noise2 = psnr(gaussian_poisson_noise2, img);
peaksnr_gaussian_poisson_noise3 = psnr(gaussian_poisson_noise3, img);

% Median

peaksnr_median_impulse_noise1 = psnr(median_impulse_noise1, img);
peaksnr_median_impulse_noise2 = psnr(median_impulse_noise2, img);
peaksnr_median_impulse_noise3 = psnr(median_impulse_noise3, img);

peaksnr_median_gaussian_noise1 = psnr(median_gaussian_noise1, img);
peaksnr_median_gaussian_noise2 = psnr(median_gaussian_noise2, img);
peaksnr_median_gaussian_noise3 = psnr(median_gaussian_noise3, img);

peaksnr_median_poisson_noise1 = psnr(median_poisson_noise1, img);
peaksnr_median_poisson_noise2 = psnr(median_poisson_noise2, img);
peaksnr_median_poisson_noise3 = psnr(median_poisson_noise3, img);

% Average

peaksnr_average_impulse_noise1 = psnr(average_impulse_noise1, img);
peaksnr_average_impulse_noise2 = psnr(average_impulse_noise2, img);
peaksnr_average_impulse_noise3 = psnr(average_impulse_noise3, img);

peaksnr_average_gaussian_noise1 = psnr(average_gaussian_noise1, img);
peaksnr_average_gaussian_noise2 = psnr(average_gaussian_noise2, img);
peaksnr_average_gaussian_noise3 = psnr(average_gaussian_noise3, img);

peaksnr_average_poisson_noise1 = psnr(median_poisson_noise1, img);
peaksnr_average_poisson_noise2 = psnr(median_poisson_noise2, img);
peaksnr_average_poisson_noise3 = psnr(median_poisson_noise3, img);

denoisepsnr = [ dgauss_1, dgauss_2, dgauss_3, dpoi_1, dpoi_2, dpoi_3, dsalt_1, dsalt_2, dsalt_3 ];


%% BM3D Plotting
figure
imshow(img)
title('Original')

%gauss
figure
subplot(1,3,1)
imshow(y_gauss)
title({['Original PSNR: ' num2str(ngauss_1)],[ 'Denoised PSNR: ' num2str(dgauss_1(1))]})
subplot(1,3,2)
imshow(y_gauss2)
title({['Original PSNR: ' num2str(ngauss_2)],[ 'Denoised PSNR: ' num2str(dgauss_2(1))]})
subplot(1,3,3)
imshow(y_gauss3)
title({['Original PSNR: ' num2str(ngauss_3)],[ 'Denoised PSNR: ' num2str(dgauss_3(1))]})
sgtitle('BM3D filtering for Gaussian noise')

%possion

figure
subplot(1,3,1)
imshow(y_poi)
title({['Original PSNR: ' num2str(npoi_1)],[ 'Denoised PSNR: ' num2str( dpoi_1(1))]})
subplot(1,3,2)
imshow(y_poi2)
title({['Original PSNR: ' num2str(npoi_2)],[ 'Denoised PSNR: ' num2str( dpoi_2(1))]})
subplot(1,3,3)
imshow(y_poi3)
title({['Original PSNR: ' num2str(npoi_3)],[ 'Denoised PSNR: ' num2str( dpoi_3(1))]})
sgtitle('BM3D filtering for Poisson noise')

%impulse
figure
subplot(1,3,1)
imshow(y_est)
title({['Original PSNR: ' num2str(nsalt_1)],[ 'Denoised PSNR: ' num2str( dsalt_1(1))]})
subplot(1,3,2)
imshow(y_est2)
title({['Original PSNR: ' num2str(nsalt_2)],[ 'Denoised PSNR: ' num2str( dsalt_2(1))]})
subplot(1,3,3)
imshow(y_est3)
title({['Original PSNR: ' num2str(nsalt_3)],[ 'Denoised PSNR: ' num2str( dsalt_3(1))]})
sgtitle('BM3D filtering for Impulse noise')


%% Linear Filter Plotting
denoisepsnr = [...
    peaksnr_gaussian_impulse_noise1, peaksnr_gaussian_impulse_noise2, peaksnr_gaussian_impulse_noise3, ...
    peaksnr_gaussian_gaussian_noise1, peaksnr_gaussian_gaussian_noise2, peaksnr_gaussian_gaussian_noise3, ...
    peaksnr_gaussian_poisson_noise1, peaksnr_gaussian_poisson_noise2, peaksnr_gaussian_poisson_noise3,...
%     peaksnr_median_impulse_noise1, peaksnr_median_impulse_noise2, peaksnr_median_impulse_noise3, ...
%     peaksnr_median_gaussian_noise1, peaksnr_median_gaussian_noise2, peaksnr_median_gaussian_noise3, ...
%     peaksnr_median_poisson_noise1, peaksnr_median_poisson_noise2, peaksnr_median_poisson_noise3, ...
    peaksnr_average_impulse_noise1, peaksnr_average_impulse_noise2, peaksnr_average_impulse_noise1,...
    peaksnr_average_gaussian_noise1, peaksnr_average_gaussian_noise2, peaksnr_average_gaussian_noise3,...
    peaksnr_average_poisson_noise1, peaksnr_average_poisson_noise2, peaksnr_average_poisson_noise3 ...
    ];


% Gaussian filtering
figure
subplot(1,3,1)
imshow(gaussian_impulse_noise1)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_impulse_noise1(1))]})
subplot(1,3,2)
imshow(gaussian_impulse_noise2)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_impulse_noise2(1))]})
subplot(1,3,3)
imshow(gaussian_impulse_noise3)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_impulse_noise3(1))]})
sgtitle('Gaussian filtering for impulse noise')

figure
subplot(1,3,1)
imshow(gaussian_gaussian_noise1)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_gaussian_noise1(1))]})
subplot(1,3,2)
imshow(gaussian_gaussian_noise2)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_gaussian_noise2(1))]})
subplot(1,3,3)
imshow(gaussian_gaussian_noise3)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_gaussian_noise3(1))]})
sgtitle('Gaussian filtering for gaussian noise')

figure
subplot(1,3,1)
imshow(gaussian_poisson_noise1)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_poisson_noise1(1))]})
subplot(1,3,2)
imshow(gaussian_poisson_noise2)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_poisson_noise2(1))]})
subplot(1,3,3)
imshow(gaussian_poisson_noise3)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_gaussian_poisson_noise3(1))]})
sgtitle('Gaussian filtering for poisson noise')

%Median filtering

figure
subplot(1,3,1)
imshow(median_impulse_noise1)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_median_impulse_noise1(1))]})
subplot(1,3,2)
imshow(median_impulse_noise2)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_median_impulse_noise2(1))]})
subplot(1,3,3)
imshow(median_impulse_noise3)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_median_impulse_noise3(1))]})
sgtitle('Median filtering for impulse noise')

figure
subplot(1,3,1)
imshow(median_gaussian_noise1)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_median_gaussian_noise1(1))]})
subplot(1,3,2)
imshow(median_gaussian_noise2)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_median_gaussian_noise2(1))]})
subplot(1,3,3)
imshow(median_gaussian_noise3)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_median_gaussian_noise3(1))]})
sgtitle('Median filtering for gaussian noise')

figure
subplot(1,3,1)
imshow(median_poisson_noise1)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_median_poisson_noise1(1))]})
subplot(1,3,2)
imshow(median_poisson_noise2)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_median_poisson_noise2(1))]})
subplot(1,3,3)
imshow(median_poisson_noise3)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_median_poisson_noise3(1))]})
sgtitle('Median filtering for poisson noise')

%Average filter

figure
subplot(1,3,1)
imshow(average_impulse_noise1)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_average_impulse_noise1(1))]})
subplot(1,3,2)
imshow(average_impulse_noise2)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_average_impulse_noise2(1))]})
subplot(1,3,3)
imshow(average_impulse_noise3)
title({['Original PSNR: ' num2str(peaksnr_impulse_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_average_impulse_noise3(1))]})
sgtitle('Average filtering for impulse noise')

figure
subplot(1,3,1)
imshow(average_gaussian_noise1)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_average_gaussian_noise1(1))]})
subplot(1,3,2)
imshow(average_gaussian_noise2)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_average_gaussian_noise2(1))]})
subplot(1,3,3)
imshow(average_gaussian_noise3)
title({['Original PSNR: ' num2str(peaksnr_gaussian_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_average_gaussian_noise3(1))]})
sgtitle('Average filtering for gaussian noise')

figure
subplot(1,3,1)
imshow(average_poisson_noise1)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise1)],[ 'Denoised PSNR: ' num2str(peaksnr_average_poisson_noise1(1))]})
subplot(1,3,2)
imshow(average_poisson_noise2)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise2)],[ 'Denoised PSNR: ' num2str(peaksnr_average_poisson_noise2(1))]})
subplot(1,3,3)
imshow(average_poisson_noise3)
title({['Original PSNR: ' num2str(peaksnr_poisson_noise3)],[ 'Denoised PSNR: ' num2str(peaksnr_average_poisson_noise3(1))]})
sgtitle('Average filtering for poisson noise')