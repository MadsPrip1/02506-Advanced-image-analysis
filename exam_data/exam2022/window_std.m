function S = window_std(I)
    window = ones(5)/25;
    K_I2 = imfilter(I.^2, window, 'symmetric');
    KI_2 = imfilter(I, window, 'symmetric').^2;
    S = sqrt(K_I2 - KI_2); 
end