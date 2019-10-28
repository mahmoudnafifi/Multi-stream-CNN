% Written by Mahmoud Afifi -- mafifi@eecs.yorku.ca | m.3afifi@gmail.com
% MIT License
% Requires Matlab 2019b or higher


clc
clear;
close all;

datasetDir = fullfile('..','dataset','stream1'); %initially, we will use stream1 as the directory for dataset

validation_ratio = 0.25; % use 25% for validation

fprintf('Training code\n');

imageSize = [128, 128, 3 * 2]; %load 1st and 2nd stream images

epochs = 100; %number of epochs

miniBatch = 32; % number of images per minibatch

lR = 10^-4; % learning rate

GPUDevice = 1; % which gpu device?

L2Reg = 10^-4; % L2 regularization factor

fprintf('Prepare training data ...\n');

imds = imageDatastore(datasetDir, 'LabelSource', 'foldernames', ...
    'IncludeSubfolders',true,'ReadFcn',@inRead);

[trainingSet, testSet] = splitEachLabel(imds, validation_ratio, 'randomize');

numClasses = length(unique(trainingSet.Labels));

fprintf('Setting training options ...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate',lR, ...
    'L2Regularization',L2Reg, ...
    'MaxEpochs',epochs, ...
    'MiniBatchSize',miniBatch, ...
    'ExecutionEnvironment','gpu',...
    'GradientDecayFactor',0.9,... %default is 0.9
    'SquaredGradientDecayFactor' ,0.999,...
    'ValidationData',testSet);


fprintf('Creating the model ...\n');
net = buildNet(imageSize, numClasses); % build the architecture

fprintf('Start training ...\n');

net = trainNetwork(trainingSet,net,options); % start training ...

fprintf('Saving the trained model ...\n');

if exist('models','dir') == 0
    mkdir('models');
end

save(fullfile('models','two-stream-model.mat'),'net','-v7.3'); % save the trained model
