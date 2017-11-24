%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CSC C11 - Assignment 2 - K-means clustering
%
% This function implements K-means clustering for a set of input
%  vectors, and an *initial* set of cluster centers. If the initial
%  set is empty, it initializes the centers and proceeds to do
%  the clustering.
%
% function [centers,labels]=kmeans(data,cent_init,k)
%
% Example calls (assuming data contains vectors of length 3 in each row)
%
% [centers,labels]=kmeans(data,[],5);	% Choose initial centers
%                                       % for 5 clusters
%
% [centers,labels]=kmeans(data,[1 2 3; 4 5 6],2);  % use initial centers
%                                                  % [1 2 3] and [4 5 6]
%
%
% Inputs: data - an array of input data points size n x d, with n
%                input points (one per row), each of length d.
%         k - number of clusters
%         cent_init - either an empty array '[]', or an array of
%                     size k x d, with k initial cluster centers
%
% Outputs: centers - Final cluster centers
%          labels - An array of size n x 1, with labels indicating
%                   which cluster each input point belongs to.
%                   e.g. if data point i belongs to cluster j,
%                   then labels(i)=j
%
% Starter code: F. Estrada, Sep. 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [centers,labels]=kmeans(data,cent_init,k)

centers=zeros(k,size(data,2));
labels=zeros(size(data,1),1);

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
% TO DO: Complete the function by implementing the k-means algorithm
%        below.
%
%        As a reminder, this is a loop that:
%          * Assigns data points to the closest cluster center
%          * Re-computes cluster centers based on the data points
%            assigned to them.
%          * Update the labels array to contain the index of the
%            cluster center each point is assigned to
%        Loop ends when the labels do not change from one iteration
%         to the next.
%
%  DO NOT compute distances from data points to cluster centers
%   with a for loop over data points. Doing so will cause you to wait
%   forever for this thing to converge. Your TA certainly won't
%   wait that long.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% REMINDER variable labels is defined above, also CODE STARTS HERE
perm_labels= zeros(size(data,1),k); % hold the permanent label matrix
while 1 % until the labeling converges
  temp_labels = zeros(size(data,1),k); % hold the temporary label matrix
  n_centers = []; % hold the temporary centers
  for row1 = 1:size(data,1) % assign labels to data points
    x_i = data(row1,:);
    % find the center with the smallest distance
    d_i = sqrt(sum((x_i-centers).^2,2));
    [min_d, min_vec_idx] = min(d_i);
    % make val @ row, min_vec_idx in temp_labels = 1
    temp_labels(row1, min_vec_idx) = 1;
  end

  % check if the labeling has changed
  if (isequal(temp_labels,perm_labels))
    break;
  else
    perm_labels = temp_labels;
  end;

  % move the centers around
  for row2 = 1:size(centers,1) % 1 to k
    c_i = perm_labels(:,row2); % take column vector c_i from perm_labels
    % set all data points not labeled by c_i to 0 and sum all of the remaining
    S_i = sum(data.*c_i);
    D_i = sum(c_i); % get number of data points labeled by c_i
    avg_center = S_i/D_i; % get the average
    n_centers = [n_centers; avg_center]; % append the center to matrix
  end
  % set new centers to old centers
  centers = n_centers;
end
% compress perm_labels to labels
for row3 = 1:size(perm_labels,1); % each row in perm_labels
  [c, c_max_idx] = max(perm_labels(row3,:));
  labels(row3, :) = c_max_idx;
end
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
% Brandon Bowen             _____________________
% (sign with your name)            (sign with your name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
