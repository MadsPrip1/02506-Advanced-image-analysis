clear, close all
addpath GraphCut functions
I = imread('../../../../Data/week7/peaks_image.png');
figure, 
subplot(1,2,1)
imagesc(I), colormap gray, axis image

% COST FUNCTION
% The region in the middle is bright compared to two darker regions. 
P = permute(I, [2,3,1]); % making sure that up is the third dimension
region_cost = cat(4, P, (255-P), P);

% GEOMETRIC CONSTRAINS
delta_xy = 1; % smoothness very constrained, try also 3 to see less smoothness
wrap_xy = 0; % 0 for terren-like surfaces
delta_ul = [1 size(I,1)]; % can be very close, but may not overlap

% CUT
s = grid_cut([],region_cost,delta_xy,wrap_xy,delta_ul);

% VISUALIZATION
subplot(1,2,2)
imagesc(I), axis image ij, colormap gray, hold on
plot(1:size(I,2),permute(s,[1,3,2]),'r', 'LineWidth',2)

