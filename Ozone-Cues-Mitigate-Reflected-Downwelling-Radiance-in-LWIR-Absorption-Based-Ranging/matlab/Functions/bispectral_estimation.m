function d_hat = bispectral_estimation(lambda, measurements, attenuation, index_1, index_2, T_air)
transmittance_hat = (measurements(:,:,index_1) - blackbody(lambda(index_1),T_air))./(measurements(:,:,index_2) - blackbody(lambda(index_2),T_air));
d_hat = -10*log10(transmittance_hat)./(attenuation(index_1) - attenuation(index_2));
idx = find(imag(d_hat)~=0);
d_hat(idx) = NaN;
idx = find(d_hat<0);
d_hat(idx) = NaN;
end