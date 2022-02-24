T = 10000;
d = 4.5;
x = trnd(d,T,1)*sqrt((d-2)/d);  % generated T obs of standardized t(d)
x = sort(x,'descend');          % sort in descending order (largest first)

params = mle(x,'distribution','tLocationScale');    % ML estimation
d_ML = params(3);                                   % MLE of d

Tu = 500;               % take Tu as 0.05*T, so use 5% largest values of x
u = x(Tu+1);            % threshold for EVT
p = ([1:1:T]-0.5)/T;    % vector of p_i values