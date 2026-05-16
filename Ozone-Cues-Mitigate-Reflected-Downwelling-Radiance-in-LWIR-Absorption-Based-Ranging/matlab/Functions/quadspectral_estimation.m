function d_hat = quadspectral_estimation(lambda, measurements, attenuation, index_1, index_2, index_3, index_4, T_air, cor_coeff)
%lambda: Wavelength vector in um
%measurements: Measurements vector in uflicks 
%index_1: Water vapor absorption line
%index_2: Clear band
%index_3: Ozone absorption
%index_4: Clear band
%Note: - Choose index_2, index_3, and index_4 with similar attenuation
%      - Choose differences between index_1 and index_2, index_3 and
%        index_4 small.

bb_air = blackbody(lambda, T_air);
L_1_centralized = measurements(:,:,index_1) - bb_air(index_1);
L_2_centralized = measurements(:,:,index_2) - bb_air(index_2);
L_3_centralized = measurements(:,:,index_3);
L_4_centralized = measurements(:,:,index_4);

%cor_coeff = cor_coeff*0.95; %slight adjusment

downwelling_correction = cor_coeff*(L_4_centralized - L_3_centralized);
transmittance_hat = (L_2_centralized - downwelling_correction)./(L_1_centralized);
d_hat = -10*log10(transmittance_hat)./(attenuation(index_2) - attenuation(index_1));

idx = find(imag(d_hat)~=0);
d_hat(idx) = NaN;
idx = find(d_hat<0);
d_hat(idx) = NaN;
end