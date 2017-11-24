
function [centers,labels]=kmeans(data,cent_init,k)

centers=zeros(k,size(data,2));
labels=zeros(size(data,1),1);

if (isempty(cent_init))
  % Initial centers is an empty array, choose initial centers
  % randomly

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
