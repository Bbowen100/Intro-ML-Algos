
function [centers,labels]=kmeans(data,cent_init,k)

centers=zeros(k,size(data,2));
labels=zeros(size(data,1),1);

if (isempty(cent_init))
  % Initial centers is an empty array, choose initial centers
  % randomly

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

% REMINDER variable labels is defined above, also CODE STARTS HERE
while 1 % until the labeling converges
  temp_labels = zeros(size(data,1),k); % hold the temporary label matrix
  n_centers = []; % hold the temporary centers
  rcenters = reshape(centers',[1, size(data,2), k]);
  rdata = repmat(data, [1,1,k]);
  dif = rdata - rcenters;
  dist = sqrt(sum(dif.^2,2));
  dist = reshape(dist, size(data,1),k);
  [m,i] = min(dist');
  temp_labels = i';

  % check if the labeling has changed
  if (isequal(temp_labels,labels))
    break;
  else
    labels = temp_labels;
  end;

  % move the centers around
  for center = 1:k % 1 to k
    rel_data = data(labels==center,:);
    rdSize = size(rel_data,1);

    avg_center = sum(rel_data) ./ rdSize; % get the average
    n_centers = [n_centers; avg_center]; % append the center to matrix
  end;
  % set new centers to old centers
  centers = n_centers;
end;
