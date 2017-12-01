%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Naive Bayes classification
%
% This function classifies the input dataset using a trained NB
% classifier
%
% Inputs:
%
% input_data       -  Input data set, one sample PER ROW
% NB_probs         -  A kxD array whose rows are the probabilities
%                     for input features being = 1 for each class
% NB_ais           -  A kx1 array with p(L=i)
%
% Return values:
%
% labels           - Labels for input samples in [1,K]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [labels]=NB_Classify(input_data, NB_probs, NB_ais)

% get probs into a 3d array ::
NB_probs = reshape(NB_probs',[1,size(NB_probs,2),size(NB_probs,1)]);
% ensure that data is binaryfied
input_data = input_data > 0;
% multiply binary data by the probabilities to see which feature
% probabilities will be present in the calculation
data_probs = input_data .* NB_probs;
% make all 0s 1s so that the multiplication / log works
data_probs = data_probs + (data_probs == 0);
% log(P(F_j_k|C_i)) + log(a_i)
dataProbs_plus_NBais = zeros(size(input_data,1), size(input_data,2), size(NB_ais,1));
for it=1:size(NB_ais,1)
  dataProbs_plus_NBais(:,:,it) = log(NB_ais(it,:)) + log(data_probs(:,:,it));
end;
center_probs = sum(dataProbs_plus_NBais,2);
% concat the columns of center_probs
for i=1:size(NB_ais,1)
  ndata_probs(:,i) = center_probs(:,:,i);
end;
[m , ind] = max(ndata_probs');
labels = ind'-1;
