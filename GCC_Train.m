%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classification by Gaussian Class Conditionals
%
% In this function, you will implement the code that trains the GCC model
% for the input dataset.
%
% Inputs:
%
% train_data   -  The training data set  (one input sample per row)
% train_labels -  Labels for the training data set
% K            -  Number of components in the GCC model
%
% Returns:
%
% centers      - Each row is the center of a Gaussian in the GCC model
% covs         - A NxNxK matrix with K, NxN covariance matrices
% ais          - Mixture proportions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [centers,covs,ais]=GCC_Train(train_data,train_labels,K);

covs=zeros([size(train_data,2) size(train_data,2) K]); % array of covariance matrices
for j=1:K
 covs(:,:,j)=eye(size(train_data,2),size(train_data,2));
end;

for i=1:K
  label_size(i,:) = size(train_labels(train_labels == i-1,:),1);
  class_data{i} = train_data(train_labels == i-1,:);
  centers(i,:) = sum(train_data(train_labels == i-1,:)) ./ size(train_data(train_labels == (i-1),:),1);
  differ{i} = class_data{i} - centers(i,:);
  covs(:,:,i) = differ{i}' * differ{i};
end;
covs = covs ./ repmat(size(train_data,1), [1, size(train_data,2)]);
ais = label_size ./ sum(label_size);
