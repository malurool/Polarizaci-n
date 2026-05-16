function [microflicks] = blackbody(lambda, T)
lambda = lambda*1e-6;
h = 6.63e-34;
c = 3e8;
k_B = 1.38e-23;
microflicks = 1e-4*(2*h*c^2./lambda.^5)./(exp(h*c./(lambda.*k_B.*T)) - 1);
end