%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Naive Bayes classifier training
%
% This function computes the parameters of a Naive Bayes classifier
% for the given training dataset
%
% Inputs:
%
% training_data    -  The training data set, one sample PER ROW
% training_labels  -  Labels for the training dataset
% K                -  Number of classes
%
% Return values:
%
% NB_probs         -  A kxD array whose rows are the probabilities
%                     for input features being = 1 for each class
% NB_ais           -  A kx1 array with p(L=i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NB_probs, NB_ais]=NB_Train(training_data, training_labels, K)

%change training data to binary;
training_data = training_data > 0;

for i=1:K
  label_size(i,:) = size(training_labels(training_labels == i-1,:),1);
  class_data{i} = training_data(training_labels == i-1,:);
end;
NB_ais = label_size ./ sum(label_size);

for j=1:K
  NB_probs(j,:) = sum(class_data{j}) ./ size(class_data{j},1); %
end;

% set 0s in prob to be 1e-9
NB_probs = (NB_probs == 0)*1e-9 + NB_probs;
