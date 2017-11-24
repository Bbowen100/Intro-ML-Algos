%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CSC C11 - Assignment 2 - Gaussian Mixture Models clustering
%
% In this function, you will implement Gaussian Mixture Models.
%  the mixture models can be initialized randomly, or with the
%  given initial centers.
%  Covariance matrices are initialized to identity.
%
% function [centers,covariances,labels]=GMMs(data,cent_init,k)
%
% Inputs: data - an array of input data points size n x d, with n
%                input points (one per row), each of length d.
%         k - number of clusters
%         cent_init - either an empty array '[]', or an array of
%                     size k x d, with k initial cluster centers
%
% Outputs: centers - Final cluster centers
%          covariances - Covariance matrices for the GMM components,
%                        this will be an array of size
%                        d x d x k, with each 'layer' of size
%                        d x d corresponding to the covariance
%                        matrix of one of the Gaussians in the model.
%          mps - An array of size k x 1, with mixture proportions
%                giving the percent of data assigned to each
%                cluster. The sum of these has to be 1.
%          labels - An array of size n x 1, with labels indicating
%                   which cluster each input point belongs to.
%                   e.g. if data point i belongs to cluster j,
%                   then labels(i)=j
%
% Starter code: F. Estrada, Sep. 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPLETE THIS TEXT BOX:
%
% 1) Student Name: Brandon Bowen
% 2) Student Name:
%
% 1) Student number: 1000459620
% 2) Student number:
%
% 1) UtorID: bowenbra
% 2) UtorID
%
% We hereby certify that the work contained here is our own
%
% _Brandon Bowen                   _____________________
% (sign with your name)            (sign with your name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [centers,covariances,mps,labels]=GMMs(data,cent_init,k)

% Initialize all the arrays we will need
centers=zeros(k,size(data,2)); % k center points of dimension col(data)
labels=zeros(size(data,1),1); % assignment labels for each data point
mps=ones(k,1)/k; % init dist is uniform
covariances=zeros([size(data,2) size(data,2) k]); % array of covariance matrices
for i=1:k
 covariances(:,:,i)=eye(size(data,2),size(data,2));
end;

if (isempty(cent_init))
  % Initial centers is an empty array, choose initial centers
  % randomly

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % TO DO: Complete this part so that your code chooses k initial
  %        centers randomly. This comes down to picking random
  %        entries in your data array.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % picks k random numbers from 1 to number of data points (for indices)
  entries_i=randperm(size(data,1),k);
  % takes row vectors from data at indices specified above
  centers = data(entries_i,:);

else
  if (size(cent_init,1)~=k | size(cent_init,2)~=size(data,2))
    fprintf(2,'Initial centers array has wrong dimensions.'\n');
    return;
  end;
  centers=cent_init;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TO DO: Complete the function by implementing the E-M algorithm for
%  estimating the parameters of a Gaussian Mixture Model
%
%        As a reminder, this is a loop that:
%          * Assigns data points to the cluster that gives it maximum
%            likelihood
%          * Updates the cluster centers, covariances, and mixture
%            proportions
%          * Update the labels array to contain the index of the
%            cluster center each point is assigned to
%        Loop ends when the labels do not change from one iteration
%         to the next.
%
%  DO NOT compute ownership probabilities for data points using
%   a for loop over data points. Doing so will cause you to wait
%   forever for this thing to converge. Your TA certainly won't
%   wait that long.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loop until convergence
while 1
  cov_inverses=zeros([size(data,2) size(data,2) k]);
  u_y_SigmaInv = zeros(size(data,1), size(data,2), k);
  u_y_SigmaInv_u_y = zeros(size(data,1), size(data,2), k);
  covariances_det = zeros(k,1);
  % creates a 3d array that contains u-y(mean - data) called differ

  tcenters = reshape(centers',[1, size(data,2), k]);
  tdata = repmat(data, [1,1,k]);
  differ = tdata - tcenters;
  for row=1:k
    %calculate inverses of the covariances
    cov_inverses(:,:,row) = pinv(covariances(:,:,row));
    % get the determinants for the covariance matrices
    covariances_det(row) = sqrt(abs(det(covariances(:,:,row))));
    % column wise mult of (u-y) and Sigma AKA covariances
    u_y_SigmaInv(:,:,row) = (-0.5*differ(:,:,row))*cov_inverses(:,:,row); %TRY TO TAKE THIS OUT
  end;
  % MULTIPLY ROW OF 1 MATRIx TO RESPECTIVE COLUMN OF OTHER MATRIX
  % (element-wise multiply then sum over row)
  u_y_SigmaInv_u_y = sum(u_y_SigmaInv .* differ,2);

  %concat vectors to make a matrix
  u_y_SigmaInv_u_y = reshape(u_y_SigmaInv_u_y,[size(data,1),k]);
  u_y_SigmaInv_u_y = e.^(u_y_SigmaInv_u_y);

  constant = 1/sqrt((2*pi)^(size(data,2))) .* 1./covariances_det;
  u_y_SigmaInv_u_y = reshape(constant,[1,k]) .* u_y_SigmaInv_u_y;

  Gamma_Num = mps' .* u_y_SigmaInv_u_y;
  Gamma_Denom = sum(Gamma_Num,2);
  Gamma = Gamma_Num ./ Gamma_Denom;

  %  2ND STEP OF EM (MAXIMIZATION)
  new_mps = sum(Gamma) ./ size(data,1);
  new_mps = new_mps'; % make it a column vector

  rGamma = reshape(Gamma,[size(data,1),1,k]);
  tnew_centers = sum(rGamma .* data) ./ sum(rGamma);
  differf = tdata - tnew_centers;
  for ind=1:k
   % get new centers
   new_centers(ind,:) = tnew_centers(:,:,ind);
   % get new cavariance matrices
   new_covariances(:,:,ind) = (Gamma(:,ind).*differf(:,:,ind))'*differf(:,:,ind);%TRY TO TAKE THIS OUT
  end;

   % get new covariance
  new_covariances = new_covariances ./ sum(rGamma);
  % get labels
  [m, ind] = max(Gamma');
  new_labels = ind';
  % the variables are the same then acheived convergence
  if(isequal(new_labels,labels))
    %set labels then break
    mps = new_mps;
    covariances = new_covariances;
    centers = new_centers;
    labels = new_labels;
    break;
  else
    mps = new_mps;
    covariances = new_covariances;
    centers = new_centers;
    labels = new_labels;
  end;
end;
