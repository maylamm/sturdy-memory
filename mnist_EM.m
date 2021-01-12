%% GMM, Meihong Lamm
clear;
clc;
close all

data = load('data.mat').data; %1,990 images 
label = load('label.mat').trueLabel;

%% a) Explore dataset
[n m]=size(data);

% two = data(:,1);
% two = reshape(two, [28 28]);
% figure;
% imshow(two);
% 
% six = data(:,1990);
% six = reshape(six, [28 28]);
% figure;
% imshow(six);

%% c) Initialize EM
d=5;
K=2;

rng(1e5)

% initialize proportions
pi = rand(K,1);
pi = pi./sum(pi);

% initialize means
mu = randn(d, K);

% initialize covariance 
sigma = zeros(d, d, K); 
I = eye(5); 
for i = 1:K
    tmp = randn(d, d);
    sigma(:,:,i) = tmp * tmp' + I;
end

% initialize poster probability p(z|x)
tau = zeros(m, K); 

%% c) Prepare data for EM 

% 1) original data 
X = data; 

% 2) mean of original data 
X_bar = mean(data,2);

% 3) centered data point 
X_delta = X - X_bar;

% 4) C
C = (1/m)*X_delta*X_delta';

% 5) eigen decomposition on C
[eigenvectors eigenvalues] = eig(C); 

% 6) k largest eigenvectors 
U_hat = eigenvectors(:,1:d);
lambda_hat = eigenvalues(1:d,1:d);

% 7) reduced dimension 
X_hat = lambda_hat^(-0.5)*U_hat'*(X-X_bar); 


%% c) Implement EM 
X_hat = X_hat';

iterno = 100; 
loglikelihood = [];
for it = 1:iterno
    % fprintf(1, '--iteration %d of %d\n', it, iterno);
    % alternate between e and m step; 
    
    % E-step; 
    for i = 1:K
        tau(:,i) = pi(i) * mvnpdf(X_hat, mu(:,i)', sigma(:,:,i)); 
    end

    % keep track of log likelihood 
    stepone = sum(tau, 2); % add columns
    steptwo = log(stepone); % take log 
    stepthree = sum(steptwo, 1); % add rows 
    loglikelihood = [loglikelihood stepthree];
  
    
    sum_tau = sum(tau, 2); 
    
    % normalize
    tau = tau ./ repmat(sum_tau, 1, K);
    


    % M-step; 
    for i = 1:K
        % update mixing proportion; 
        pi(i) = sum(tau(:,i), 1) ./ m; 
        % update gaussian center; 
        mu(:, i) = X_hat' * tau(:,i) ./ sum(tau(:,i), 1); 
        % update gaussian covariance;
        tmpdata = X_hat - repmat(mu(:,i)', m, 1); 
        sigma(:,:,i) = tmpdata' * diag(tau(:,i)) * tmpdata ./ sum(tau(:,i), 1); 
    end
    

end
% plot(loglikelihood) 
%% d) Report fitted GMM model and visualize avg images 
recovered_mu = U_hat*lambda_hat^(0.5)*mu+X_bar;

% avg_two = reshape(recovered_mu(:,1), [28 28]);
% avg_six = reshape(recovered_mu(:,2), [28 28]);
% figure; 
% subplot(1,2,1);
% imshow(avg_two);
% title('reconstructed mu for two');
% subplot(1,2,2);
% imshow(avg_six);
% title('reconstructed mu for six');


recovered_sigma(:,:,1) = U_hat*lambda_hat^(0.5)*sigma(:,:,1)*lambda_hat^(0.5)'*U_hat';
recovered_sigma(:,:,2) = U_hat*lambda_hat^(0.5)*sigma(:,:,2)*lambda_hat^(0.5)'*U_hat';

% sigma_two = recovered_sigma(:,:,1);
% sigma_six = recovered_sigma(:,:,2);
% figure; 
% subplot(1,2,1);
% imshow(sigma_two); 
% title('reconstructed sigma for two');
% subplot(1,2,2);
% imshow(sigma_six); 
% title('reconstructed sigma for six');


%% e) Calculate misclassification rate 

% GMM label 
for i = 1:1990
    if tau(i,1) > tau(i,2)
        gmm_label(i) = 6;
    else 
        gmm_label(i) = 2;
    end 
end


% GMM misclassification rate 
gmm_miss_2 = 0;
gmm_miss_6 = 0;
for i = 1:1990
    if label(i) == 2
        if gmm_label(i)~=2
            gmm_miss_2 = gmm_miss_2 + 1; 
        end
    end
    if label(i) == 6
        if gmm_label(i) ~= 6
            gmm_miss_6 = gmm_miss_6 + 1; 
        end
    end
end
gmm_miss_rate_2 = gmm_miss_2 / m; 
gmm_miss_rate_6 = gmm_miss_6 / m; 

%% e) k-means
% kmeans(matrix, k) 
% dim(matrix) = (points, variables)
kmeans_label = kmeans(X_hat,2); 

% kmeans misclassification rate 
kmeans_miss_2 = 0;
kmeans_miss_6 = 0; 
kmeans_label = kmeans_label';
for i = 1:1990 
    if label(i) == 2
        if kmeans_label(i) ~= 1
            kmeans_miss_2 = kmeans_miss_2 + 1; 
        end
    end
    if label(i) == 6
        if kmeans_label(i) ~= 2
            kmeans_miss_6 = kmeans_miss_6 + 1; 
        end
    end

end
kmeans_miss_rate_2 = kmeans_miss_2 / m; 
kmeans_miss_rate_6 = kmeans_miss_6 / m; 
