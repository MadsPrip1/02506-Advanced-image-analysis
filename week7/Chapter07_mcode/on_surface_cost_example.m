clear, close all
addpath GraphCut functions

lw = 2;

I = imread('../../../../Data/week7/layers_A.png');
surface_cost = permute(I,[2,3,1]);
dim = size(I);

%% input
figure
subplot(1,4,1)
imagesc(I), axis image, colormap gray
title('input image'), drawnow

%% one line
delta = 3;
wrap =  false;
s = grid_cut(surface_cost, [], delta, wrap);
subplot(1,4,2)
imagesc(I), axis image, colormap gray, hold on
plot(1:dim(2), s, 'r', 'LineWidth', lw)
title(['delta = ',num2str(delta)]), drawnow

%% a smoother line
delta = 1;
wrap =  false;
s = grid_cut(surface_cost, [], delta, wrap);
subplot(1,4,3)
imagesc(I), axis image, colormap gray, hold on
plot(1:dim(2), s, 'r', 'LineWidth', lw)
title(['delta = ',num2str(delta)]), drawnow

%% two lines
costs = cat(4,surface_cost,surface_cost);
delta = 3;
wrap = false;
overlap = [15, size(I,1)];

s = grid_cut(costs, [], delta, wrap, overlap);
subplot(1,4,4)
imagesc(I), axis image, colormap gray, hold on
plot(1:dim(2),permute(s,[1,3,2]),'r', 'LineWidth',lw)
title('two dark lines'), drawnow