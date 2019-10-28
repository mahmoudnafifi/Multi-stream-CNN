% Written by Mahmoud Afifi -- mafifi@eecs.yorku.ca | m.3afifi@gmail.com
% MIT License
% Requires Matlab 2019b or higher

function layers = buildNet(imageSize,numClasses)


input = imageInputLayer(imageSize,'Name','InputLayer',...
    'Normalization','zerocenter');

depths = [128, 128, 128, 256, 256]; %conv depth for each block

split_1st = splittingLayer('Splitting-1st','1st');
split_2nd = splittingLayer('Splitting-2nd','2nd');

%stream-1st
layers = [input
    split_1st];

for i = 1 : length(depths)
    if i == 1
        layers = [layers
            addBlock(depths(i)/2, i, sprintf('Block_%d_1st',i),3); % 3 color channels
            ];
    else
        layers = [layers
            addBlock(depths(i)/2, i, sprintf('Block_%d_1st',i));
            ];
    end
end

%stream-2nd
layers_2nd =[split_2nd];
%batch_LR];
for i = 1 : length(depths)
    if i == 1
        layers_2nd = [layers_2nd
            addBlock(depths(i)/2, i, sprintf('Block_%d_2nd',i),3); % 3 color channels
            ];
    else
        layers_2nd = [layers_2nd
            addBlock(depths(i)/2, i,sprintf('Block_%d_2nd',i));
            ];
    end
end


% cat convs
numInputs = 2;
cat_dim = 3; %third dimension
cat_Layer = concatenationLayer(cat_dim,numInputs,'Name','Cat-Layer');


fc1 = fullyConnectedLayer(1024,'Name', 'FC-1');
relu1 = reluLayer('Name','ReLu-FC-1');
dropout1 = dropoutLayer('Name','dropOut-FC-1');


fc = fullyConnectedLayer(numClasses,'Name', 'FC-out');
softmxLayer = softmaxLayer('Name','Softmaxx');
endLayer = classificationLayer('Name','outLayer');


layers = [layers
    cat_Layer
    fc1
    relu1
    dropout1
    fc
    softmxLayer
    endLayer];


layers= layerGraph(layers);
layers= addLayers(layers,layers_2nd);

layers = connectLayers(layers,sprintf('Block_%d_2nd_Pooling_%d',...
    length(depths),length(depths)),'Cat-Layer/in2');


layers = connectLayers(layers,'InputLayer','Splitting-2nd');


end



function block = addBlock(depth, number, prefix, channels)
if number > 1
    conv = convolution2dLayer(3,depth,'Stride',1,'Padding',1,'Name',...
        sprintf('%s_Conv_%d',prefix,number));
else
    conv = convolution2dLayer(3,depth,'Stride',1,'Padding',1,'Name',...
        sprintf('%s_Conv_%d',prefix,number),'NumChannels',channels);
end

relu = reluLayer('Name',sprintf('%s_ReLU_%d',prefix,number));

pool = maxPooling2dLayer(2,'Stride',2,'Padding',0, 'Name',...
    sprintf('%s_Pooling_%d',prefix,number));

block = [conv
    relu
    pool
    ];
end

