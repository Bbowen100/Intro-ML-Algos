%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimensionality reduction by PCA
%
% In this function you will apply PCA to an input dataset to
% return a low-dimensional representation for the input data.
%
% Inputs:
%
% input_data    -   Input dataset, one sample PER ROW
% k             -   Number of dimensions for the low-dimensional data
%
% Return values:
%
% LodWim_data   -   The low-dimensional representation of the dataset
%                   one sample per row
% V             -   The matrix with the PCA directions (one per column)
% mu            -   The mean of the input data (needed for reconstruction)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [LowDim_data,V,mu]=PCA(input_data, k);

if(k > size(input_data,2) || k < 1)
  printf('not a legitimate value for k');
  return;
end;
% get the mean of the data
mu = sum(input_data) ./ repmat(size(input_data,1), [1, size(input_data,2)]);
% data - mean
differ = input_data - mu;
% covariance matrix
cov_mat = differ' * differ;
% normalized covariance matrix;
cov_mat = cov_mat ./ repmat(size(input_data,1), [1, size(input_data,2)]);
% eigen values and vectors
[V, eigen_vals] = eigs(cov_mat, k, 'la');
eigen_vals = sum(eigen_vals,2);

%get low dimensional data;
LowDim_data = (V' * differ')';
