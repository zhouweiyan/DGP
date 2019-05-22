function err = rms(x)

x = x(find(isnan(x) ~= 1));
err = sqrt(sum(x.^2) / numel(x));