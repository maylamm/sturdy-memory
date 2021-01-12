%% Meihong Lamm

clear all;
close all;
clc;

rng('default');

%% Import data
dat = importdata('mnist_data.mat');
dat = dat'; 
label = importdata('mnist_label.mat');
label = label';

for i=1:1990
    if label(i)==6
        label(i) = 1;
    end
end


%% Create training data and testing data 
nInst = size( dat, 1 );
% random permutation of 1:270
rp = randperm( nInst );
nTrain = 1592;
nTest = nInst - nTrain;

trainDat = dat( rp( 1 : nTrain ), : );
trainLabel = label( rp( 1 : nTrain ) );
testDat = dat( rp( nTrain + 1 : end ), : );
testLabel = label( rp( nTrain + 1 : end ) );

%% Add noise to training data 
noiseDat = trainDat + 0.01*randn(size(trainDat));
%% Logistic Regression 
disp("Logistic Regression")
B = mnrfit( trainDat, trainLabel );

% predict 
threshold = 0.5;

pred = mnrval( B, trainDat );
trainPred = ( pred( :, 2 ) > threshold ) + 1;   % have to add 1 to get class labels 1 and 2

pred = mnrval( B, testDat );
testPred = ( pred( :, 2 ) > threshold ) + 1;

% calculate accuracy 
nTrainCorrect = sum( trainLabel == trainPred );
nTestCorrect = sum( testLabel == testPred );

[ mat, order ] = confusionmat( trainLabel, trainPred );
disp( ' ' );
disp( order' );
disp( mat );

[ mat, order ] = confusionmat( testLabel, testPred);
disp( ' ' );
disp( order' );
disp( mat );

%% Naive Bayes
% Assumption: each feature makes an independent and equal contribution to
disp('Naive Bayes');
bayes_mdl = fitcnb(noiseDat, trainLabel);
train_pred = predict(bayes_mdl, trainDat); 
test_pred = predict(bayes_mdl, testDat); 
[ mat, order ] = confusionmat( trainLabel, train_pred );
disp( ' ' );
disp( order' );
disp( mat );
[ mat, order ] = confusionmat( testLabel, test_pred );
disp( ' ' );
disp( order' );
disp( mat );
%% KNN
disp("KNN")
knn_mdl = fitcknn(trainDat, trainLabel);
train_pred = predict(knn_mdl, trainDat); 
test_pred = predict(knn_mdl, testDat); 
[ mat, order ] = confusionmat( trainLabel, train_pred );
disp( ' ' );
disp( order' );
disp( mat );
[ mat, order ] = confusionmat( testLabel, test_pred );
disp( ' ' );
disp( order' );
disp( mat );



%% PCA
[coeff,score,latent] = pca(trainDat);
pca_train = score(:,1:2);
gscatter(pca_train(:,1), pca_train(:,2), trainLabel,'rb','v^',[],'off'); 
legend("6", "2"); 
title("PCA of Training Data"); 


%% Graph Naive Bayes and KNN
X = pca_train; 
y = categorical(trainLabel); 

classifier_name = {'Naive Bayes','Nearest Neighbor'};
classifier{1} = fitcnb(pca_train, trainLabel);
classifier{2} = fitcknn(pca_train, trainLabel); 


x1range = min(pca_train(:,1)):.01:max(pca_train(:,1));
x2range = min(pca_train(:,2)):.01:max(pca_train(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

figure; 
for i = 1:numel(classifier)
   predictedclass = predict(classifier{i},XGrid);
   subplot(2,2,i);
   gscatter(xx1(:), xx2(:), predictedclass,'kw');
   hold on; 
   gscatter(pca_train(:,1), pca_train(:,2), trainLabel);
   title(classifier_name{i})
   legend off, axis tight
end

%% Graph Logistic Regression 
B = mnrfit( pca_train, trainLabel );

pred = mnrval(B, XGrid); 

predictedclass = ( pred( :, 2 ) > threshold ) + 1;

subplot(2,2,3);
gscatter(xx1(:), xx2(:), predictedclass,'kw');
hold on; 
gscatter(pca_train(:,1), pca_train(:,2), trainLabel);
title("Logistic Regression");
legend off, axis tight

