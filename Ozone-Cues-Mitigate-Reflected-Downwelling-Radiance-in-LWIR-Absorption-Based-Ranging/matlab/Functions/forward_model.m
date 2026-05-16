function radiance_sensor = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air)
obj_emission = blackbody(lambda, T).*emissivity;
obj_reflection = (1-emissivity) .* sum(V.*reflected, 4);

tau = 10.^(-attenuation.*d./10);

radiance_sensor = tau.*(obj_emission + obj_reflection) + (1 - tau).*blackbody(lambda, T_air);
end