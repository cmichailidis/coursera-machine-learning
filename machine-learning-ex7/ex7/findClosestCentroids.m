function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);

% loop through the entire dataset 

for i=1:m
  % for each training example find the vectors
  % which connect it with the centroids.
  % store those vectors in a new matrix called
  % dist_vectors. Every row of this matrix
  % contains a single vector. 
  dist_vectors = X(i,:)-centroids;
  
  % calculate the magnitute of each vector
  % by squaring every element of the previous
  % matrix and adding its new elements row by 
  % row. This will compute the squared distance
  % of the current training example from every
  % centroid. Those distances are stored in a
  % vector called dist_squared.
  dist_squared = sum(dist_vectors.^2, 2);
  
  % find the centroid which is closest to the current
  % training example. Store the index of this centroid
  % on the idx vector for future use.
  [dummy, idx(i)] = min(dist_squared); 
  
endfor

% =============================================================

end