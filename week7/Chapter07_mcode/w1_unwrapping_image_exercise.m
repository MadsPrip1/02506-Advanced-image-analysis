% optional exercise 1.1.5 
I = imread('../../../../Data/week1/dental/slice100.png');

a = 180; % number of angles for unfolding
angles = (0 : (a-1)) *2*pi/a; % angular coordinate

center = (1+size(I))/2;
r = 1:min(size(I)/2); % radial coordinate for unwrapping

X = center(1) + r'*cos(angles);
Y = center(2) + r'*sin(angles);

F = griddedInterpolant(double(I));
U = F(X,Y);

figure
subplot(121), imagesc(I), axis image, colormap gray
subplot(122), imagesc(U), axis image, colormap gray