%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classification by Gaussian Class Conditionals
%
% This function uses a trained GCC model to classify an input dataset
%
% Inputs:
%
% input_data   -  The training data set  (one input sample per row)
% centers      -  Centers of the Gaussians in the GCC model
% covs         -  Covariances for the Gaussians in the GCC model
% ais          -  Mixture proportions
%
% Returns:
%
% labels       - Output labels for input data samples. Each entry is a
%                value in [1,K] for a GCC model with K components.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [labels]=GCC_Classify(input_data,centers,covs,ais);

k = size(centers,1);
tcenters = reshape(centers',[1, size(input_data,2), k]);
tdata = repmat(input_data, [1,1,k]);
differ = tdata - tcenters;

for row=1:k
  %calculate inverses of the covariances
  cov_inverses(:,:,row) = pinv(covs(:,:,row));
  % get the determinants for the covariance matrices
  covariances_det(row) = sqrt(abs(det(covs(:,:,row))));
  % column wise mult of (u-y) and Sigma AKA covariances
  u_y_SigmaInv(:,:,row) = (-0.5*differ(:,:,row))*cov_inverses(:,:,row); %TRY TO TAKE THIS OUT
end;
% MULTIPLY ROW OF 1 MATRIx TO RESPECTIVE COLUMN OF OTHER MATRIX
% (element-wise multiply then sum over row)
u_y_SigmaInv_u_y = sum(u_y_SigmaInv .* differ,2);

%concat vectors to make a matrix
u_y_SigmaInv_u_y = reshape(u_y_SigmaInv_u_y,[size(input_data,1),k]);
u_y_SigmaInv_u_y = e.^(u_y_SigmaInv_u_y);

constant = 1/sqrt((2*pi)^(size(input_data,2))) .* 1./covariances_det;
u_y_SigmaInv_u_y = reshape(constant,[1,k]) .* u_y_SigmaInv_u_y;

Gamma_Num = ais' .* u_y_SigmaInv_u_y;
Gamma_Denom = sum(Gamma_Num,2);
Gamma = Gamma_Num ./ Gamma_Denom;

[m, ind] = max(Gamma');
labels = ind'-1;
