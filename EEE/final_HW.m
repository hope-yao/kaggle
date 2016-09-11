%% final project for EEE511 course.
%
% automatic clustering nueral spike signal of any amount dataset
% 
% input: need to put spike data into <test data> folder
% output: an excle would be generated  
%
% good result could be obtained for all spike datasets
% by default parameter settings.
%
% for further info, feel free to contact hope-yao@asu.edu
%

function final_HW()
clc; clear;
max_num = 4;%number of data set here
for dataset_num = 1:max_num
% for dataset_num = 1:2
    % parameter setup
    paras = hw6_setup(dataset_num,max_num);
    % preprocess
    hw6_prep(paras);
    % feature extraction
    hw6_fex(paras);
    % dimension reduction
    hw6_reduc(paras);
    % dimension reduction
    [X,idx,C] = hw6_clustering(paras);
    % post process
    hw6_postp(paras,X,idx,C);
    % save result for TA's test
    result(:,dataset_num) = idx;
end
filename = 'result.xls';
xlswrite(filename,result)
end

%% optimum parameter setup
function paras=hw6_setup(dataset_num,max_num)
paras = struct( ...
    'max_num', max_num,... %maxum number of spike datasets
    'spike_num', 1,... %switch between different spike dataset
    'include_frq', 2,... %0: only time info; 1: fft; 2: wavelet
    'fft_term', 32,... % number of expansion in fft
    'frq_weight',3,... % weight of frequency information. the bigger the more importance of freq info
    'trunc_beg', 10,... % number of terms truncated in the begining of time series
    'trunc_end', 20,... % number of terms truncated in the ending of time series
    'dim_reduce', 2, ... % 1: pca, 2: tsne
    'perplexity', 30,... % perplexity for tsne
    'dim_num', 3, ... % reduce to how many dimension
    'tsne_itr', 600, ... % number of iteration in tsne
    'cluster_method', 1, ... % 1: kmeans; 2: hierachical(problematic)
    'visualize', 2 ... % 0:no plot; 1: plot all; 2:  plot cluster
    );
paras.spike_num = dataset_num;
% if dataset_num<3
%     paras.include_frq = 0;
%     paras.trunc_beg = 1; %starts from 1
%     paras.trunc_end = 0;
% end
end

%% data preprocess
function hw6_prep(paras)
for i=1:paras.max_num
    filename = strcat('./test data/Sampledata_test_',num2str(i),'.mat');
    X = load(filename);
    spikes = X.spikes;
    filename_t = strcat('./test data/spikes_',num2str(paras.spike_num),'.txt');
    save(filename_t,'spikes','-ascii');
end
end

%% feature extraction
function hw6_fex(paras)

filename_t = strcat('./test data/spikes_',num2str(paras.spike_num),'.txt');
X1 = load(filename_t);  X1 = X1./(max(max(X1))); %normalize
X3 = X1(:,paras.trunc_beg:(size(X1,2)-paras.trunc_end));%truncation
if 0~=paras.include_frq
    if 1 == paras.include_frq
        disp('====================FFT=======================')
        hw6_fft(paras);
    elseif 2 == paras.include_frq
        disp('====================WAVELET=======================')
        hw6_wavelet(paras);
    end
    % combine time and frq info
    filename_f = strcat('./test data/spikes_',num2str(paras.spike_num),'_frq.txt');
    X2 = load(filename_f);
    X2 = X2./max(max(X2)) * paras.frq_weight;
    X = [X2,X3] ; %% consider both frequency and time domain
else
    X = X3;
end
% save final info
filename = strcat('./test data/spikes_',num2str(paras.spike_num),'_info.txt');
save(filename,'X','-ascii');
end

%% dimension reduction
function hw6_reduc(paras)
if 1 == paras.dim_reduce
    disp('====================PCA=======================')
    hw6_pca(paras);
elseif 2 == paras.dim_reduce
    disp('====================TSNE=======================')
    hw6_tsne(paras);
else
    %doing nothing
end
end

%% wavelet
function hw6_wavelet(paras)
filename_t = strcat('./test data/spikes_',num2str(paras.spike_num),'.txt');
X = load(filename_t);
for i=1:size(X,1)
    
    [c,l] = wavedec(X(i,:),3,'db1');
    D(i,:) = detcoef(c,l,1);
    
end

filename_f = strcat('./test data/spikes_',num2str(paras.spike_num),'_frq.txt');
save(filename_f,'D','-ascii')

end

%% FFT
function hw6_fft(paras)

filename_t = strcat('./test data/spikes_',num2str(paras.spike_num),'.txt');
filename_f = strcat('./test data/spikes_',num2str(paras.spike_num),'_frq.txt');

X = load(filename_t);
n = paras.fft_term;
Y = fft(X,n,2);
P2 = abs(Y/n);
P1 = P2(:,1:n/2+1);
P1(:,2:end-1) = 2*P1(:,2:end-1);
save(filename_f,'P1','-ascii')

end

%% clustering
function [X,idx,C]=hw6_clustering(paras)
disp('====================CLUSTERING=======================')
X = load(strcat('./test data/spikes_',num2str(paras.spike_num),'_mapped.txt'));
if 1 == paras.cluster_method
    opts = statset('Display','final');
    [idx,C] = kmeans(X,3,'Distance','cityblock',...
        'Replicates',5,'Options',opts);
else
    %     d = pdist(X);
    %     Z = linkage(d);
    %     idx = cluster(Z,'maxclust',3);
end
end

%% result visualization
function hw6_postp(paras,X,idx,C)
disp('====================POST PROCESSING=======================')
% visualize clusters
if 1 == paras.cluster_method && 0~=paras.visualize
    % KMEANS
    figure();
    if 2<paras.dim_num
        plot3(X(idx==1,1),X(idx==1,2),X(idx==1,3),'r.','MarkerSize',12)
        hold on
        plot3(X(idx==2,1),X(idx==2,2),X(idx==2,3),'b.','MarkerSize',12)
        plot3(X(idx==3,1),X(idx==3,2),X(idx==3,3),'k.','MarkerSize',12)
        % plot3(X(idx==4,1),X(idx==4,2),X(idx==4,3),'y.','MarkerSize',12)
        plot3(C(:,1),C(:,2),C(:,3),'kx',...
            'MarkerSize',15,'LineWidth',3)
    else
        plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
        hold on
        plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
        plot(X(idx==3,1),X(idx==3,2),'k.','MarkerSize',12)
        plot(C(:,1),C(:,2),'kx',...
            'MarkerSize',15,'LineWidth',3)
    end
    legend('Cluster 1','Cluster 2','Cluster 3')
    filename = strcat('Clusters for dataset ',num2str(paras.spike_num));
    title(filename);
    % savefig(filename);
    hold off
else
    % DBSCAN
    plot3(X(idx==1,1),X(idx==1,2),X(idx==1,3),'r.','MarkerSize',12)
    hold on
    plot3(X(idx==2,1),X(idx==2,2),X(idx==2,3),'b.','MarkerSize',12)
    plot3(X(idx==3,1),X(idx==3,2),X(idx==3,3),'k.','MarkerSize',12)
    % plot3(X(idx==4,1),X(idx==4,2),X(idx==4,3),'y.','MarkerSize',12)
    legend('Cluster 1','Cluster 2','Cluster 3',...
        'Location','NW')
    filename = strcat('Cluster spike_',num2str(paras.spike_num));
    title(filename);
    % savefig(filename);
    hold off
end

% visualize spikes
spike_idx = zeros(3,length(X));
cnt = zeros(3,1);
for ii=1:length(X)
    if idx(ii)==1
        cnt(1) = cnt(1) + 1;
        spike_idx(1,cnt(1)) = ii;
    elseif idx(ii)==2
        cnt(2) = cnt(2) + 1;
        spike_idx(2,cnt(2)) = ii;
    else
        cnt(3) = cnt(3) + 1;
        spike_idx(3,cnt(3)) = ii;
    end
end
if 1==paras.visualize
    spikes = load(strcat('./test data/spikes_',num2str(paras.spike_num),'.txt'));
    average_spikes = zeros(3,size(spikes,2));
    for ii=1:3
        figure;
        for j=1:cnt(ii)
            if 0 == spike_idx(ii,j)
                break;
            end
            plot(spikes(spike_idx(ii,j),:));hold on;
            average_spikes(ii,:) = average_spikes(ii,:) + spikes(spike_idx(ii,j),:);
        end
        average_spikes(ii,:) = average_spikes(ii,:) / cnt(ii);
        plot(average_spikes(ii,:),'k','LineWidth',3); grid on; axis([0 50 -2 2.5])
        filename = strcat('spike ',num2str(paras.spike_num),'class ',num2str(ii));
        title(filename);
        %     savefig(filename);
    end
    figure;
    for ii=1:3
        plot(average_spikes(ii,:),'LineWidth',3); grid on; axis([0 50 -2 2.5])
        hold on;
    end
    legend('spike class 1','spike class 2','spike class 3');
end
end

%% PCA
function hw6_pca(paras)

% loading data
filename_t = strcat('./test data/spikes_',num2str(paras.spike_num),'.txt');
filename_f = strcat('./test data/spikes_',num2str(paras.spike_num),'_frq.txt');
if (1==paras.include_frq)
    X1 = load(filename_t);
    X2 = load(filename_f);
    X1 = X1./(max(max(X1))); X2 = X2./max(max(X2)) * paras.frq_weight;%normalize
    X3 = X1(:,paras.trunc_beg:(size(X1,2)-paras.trunc_end));%truncation
    X = [X2,X3]; %% consider both frequency and time domain
else
    X = load(filename_t);
end

% running pca
dim = paras.dim_num;
spikes_data = X;
% Run pca
covx = cov(spikes_data);
avex = sum(spikes_data,1)/size(spikes_data,1);
[V,~] = eig(covx);

Z = zeros(size(spikes_data,1),dim);
for i = 1:size(spikes_data,1)
    t=V'*(spikes_data(i,:)'-avex');
    Z(i,:)=t(size(covx,1)-dim+1:size(covx,1));
end

% figure;hold on;
% for i=1:500
% plot(Z(i,:))
% end
mappedX = Z; % dimension reduced data
save(strcat('./test data/spikes_',num2str(paras.spike_num),'_mapped.txt'),'mappedX','-ascii');

% Plot results
if 1==paras.visualize
    figure;
    if 2<paras.dim_num
        plot3(mappedX(:,1), mappedX(:,2), mappedX(:,3),'b.');
    else
        plot(mappedX(:,1), mappedX(:,2),'b.');
    end
    filename = strcat( 'Data Set ', num2str(paras.spike_num), ' after PCA');
    title(filename);
    % savefig(filename);
end
end

%% t-SNE
function hw6_tsne(paras)

% loading data
filename = strcat('./test data/spikes_',num2str(paras.spike_num),'_info.txt');
X = load(filename);

% running tsne
%train_labels = train_labels(ind(1:5000));
% Set parameters
no_dims = paras.dim_num;
initial_dims = size(X,2);
perplexity = paras.perplexity;
% Run t?SNE
itr_num = paras.tsne_itr;
mappedX = tsne(X, [], no_dims, initial_dims, perplexity, itr_num);
save(strcat('./test data/spikes_',num2str(paras.spike_num),'_mapped.txt'),'mappedX','-ascii');
% Plot results
if 1==paras.visualize
    figure;
    if 2>paras.dim_num
        plot3(mappedX(:,1), mappedX(:,2), mappedX(:,3),'b.');
    else
        scatter(mappedX(:,1), mappedX(:,2));
    end
    filename = strcat( 'Data Set ', num2str(paras.spike_num), ' after T-SNE');
    title(filename);
    % savefig(filename);
end
end

function [P, beta] = x2p(X, u, tol)



if ~exist('u', 'var') || isempty(u)
    u = 15;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4;
end

% Initialize some variables
n = size(X, 1);                     % number of instances
P = zeros(n, n);                    % empty probability matrix
beta = ones(n, 1);                  % empty precision vector
logU = log(u);                      % log of perplexity (= entropy)

% Compute pairwise distances
disp('Computing pairwise distances...');
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * X * X'));

% Run over all datapoints
disp('Computing P-values...');
for i=1:n
    
    if ~rem(i, 500)
        disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
    end
    
    % Set minimum and maximum values for precision
    betamin = -Inf;
    betamax = Inf;
    
    % Compute the Gaussian kernel and entropy for the current precision
    Di = D(i, [1:i-1 i+1:end]);
    [H, thisP] = Hbeta(Di, beta(i));
    
    % Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU;
    tries = 0;
    while abs(Hdiff) > tol && tries < 50
        
        % If not, increase or decrease precision
        if Hdiff > 0
            betamin = beta(i);
            if isinf(betamax)
                beta(i) = beta(i) * 2;
            else
                beta(i) = (beta(i) + betamax) / 2;
            end
        else
            betamax = beta(i);
            if isinf(betamin)
                beta(i) = beta(i) / 2;
            else
                beta(i) = (beta(i) + betamin) / 2;
            end
        end
        
        % Recompute the values
        [H, thisP] = Hbeta(Di, beta(i));
        Hdiff = H - logU;
        tries = tries + 1;
    end
    
    % Set the final row of P
    P(i, [1:i - 1, i + 1:end]) = thisP;
end
disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end

% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H, P] = Hbeta(D, beta)
P = exp(-D * beta);
sumP = sum(P);
H = log(sumP) + beta * sum(D .* P) / sumP;
P = P / sumP;
end

function ydata = tsne_p(P, labels, no_dims,itr_num)


ii=1;
if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Initialize some variables
n = size(P, 1);                                     % number of instances
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
max_iter = itr_num;                                    % maximum number of iterations
epsilon = 500;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta

% Make sure P-vals are set properly
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
if ~initial_solution
    P = P * 4;                                      % lie about the P-vals to find better local minima
end

% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end
y_incs  = zeros(size(ydata));
gains = ones(size(ydata));

% Run the iterations
for iter=1:max_iter
    
    % Compute joint probability that point i and j are neighbors
    sum_ydata = sum(ydata .^ 2, 2);
    num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
    num(1:n+1:end) = 0;                                                 % set diagonal to zero
    Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
    
    % Compute the gradients (faster implementation)
    L = (P - Q) .* num;
    y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
    
    % Update the solution
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
        + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    ydata = ydata + y_incs;
    ydata = bsxfun(@minus, ydata, mean(ydata, 1));
    
    % Update the momentum if necessary
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter && ~initial_solution
        P = P ./ 4;
    end
    
    % Print out progress
    if ~rem(iter, 10)
        cost = const - sum(P(:) .* log(Q(:)));
        disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
        error(ii)=cost;
        
    end
    
    % Display scatter plot (maximally first three dimensions)
    if ~rem(iter, 10) && ~isempty(labels)
        if no_dims == 1
            scatter(ydata, ydata, 9, labels, 'filled');
        elseif no_dims == 2
            scatter(ydata(:,1), ydata(:,2), 9, labels, 'filled');
        else
            scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 40, labels, 'filled');
        end
        axis tight
        axis off
        drawnow
    end
    
    
    
end
end

function ydata = tsne(X, labels, no_dims, initial_dims, perplexity,itr_num)

if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end
if ~exist('initial_dims', 'var') || isempty(initial_dims)
    initial_dims = min(50, size(X, 2));
end
if ~exist('perplexity', 'var') || isempty(perplexity)
    perplexity = 30;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
    perplexity = initial_dims;
else
    initial_solution = false;
end

% Normalize input data
X = X - min(X(:));
X = X / max(X(:));
X = bsxfun(@minus, X, mean(X, 1));

% Perform preprocessing using PCA
if ~initial_solution
    disp('Preprocessing data using PCA...');
    if size(X, 2) < size(X, 1)
        C = X' * X;
    else
        C = (1 / size(X, 1)) * (X * X');
    end
    [M, lambda] = eig(C);
    [lambda, ind] = sort(diag(lambda), 'descend');
    M = M(:,ind(1:initial_dims));
    lambda = lambda(1:initial_dims);
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
    end
    X = bsxfun(@minus, X, mean(X, 1)) * M;
    clear M lambda ind
end

% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
P = d2p(D, perplexity, 1e-5);                                           % compute affinities using fixed perplexity
clear D

% Run t-SNE
if initial_solution
    ydata = tsne_p(P, labels, ydata, itr_num);
else
    ydata = tsne_p(P, labels, no_dims, itr_num);
end
end

function [P, beta] = d2p(D, u, tol)



if ~exist('u', 'var') || isempty(u)
    u = 15;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4;
end

% Initialize some variables
n = size(D, 1);                     % number of instances
P = zeros(n, n);                    % empty probability matrix
beta = ones(n, 1);                  % empty precision vector
logU = log(u);                      % log of perplexity (= entropy)

% Run over all datapoints
for i=1:n
    
    if ~rem(i, 500)
        disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
    end
    
    % Set minimum and maximum values for precision
    betamin = -Inf;
    betamax = Inf;
    
    % Compute the Gaussian kernel and entropy for the current precision
    [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
    
    % Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU;
    tries = 0;
    while abs(Hdiff) > tol && tries < 50
        
        % If not, increase or decrease precision
        if Hdiff > 0
            betamin = beta(i);
            if isinf(betamax)
                beta(i) = beta(i) * 2;
            else
                beta(i) = (beta(i) + betamax) / 2;
            end
        else
            betamax = beta(i);
            if isinf(betamin)
                beta(i) = beta(i) / 2;
            else
                beta(i) = (beta(i) + betamin) / 2;
            end
        end
        
        % Recompute the values
        [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
        Hdiff = H - logU;
        tries = tries + 1;
    end
    
    % Set the final row of P
    P(i, [1:i - 1, i + 1:end]) = thisP;
end
disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end



