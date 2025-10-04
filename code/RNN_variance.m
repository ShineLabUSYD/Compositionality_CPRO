%% Check that simulated time series matches the original data from Yang

% Load in activity for single RNN
load C:\Users\JoshB\Documents\Projects\compositionality_cole\RNN\Activity\0\0_neural_activity.mat

% Plot activity for single task
temp = squeeze(mean(contextdelaydm1_activity,2));
figure;
plot(temp);
xline([0 149],'LineWidth',1.5,'Color','red'); % Fixation start/end
xline([65 150 174],'LineWidth',1.5,'Color','black'); % Rule start, Response/Go start/end
xline([25 124],'LineWidth',1.5,'Color','blue'); % Stimulus start/end

% Remove fixation period from timeseries
contextdelaydm1_resp = contextdelaydm1_activity(150:end,:,:);
% Calculate variance across trials per unit and time point
cdd1_var = squeeze(var(contextdelaydm1_resp,[],2));
mean_cdd1var = mean(cdd1_var,1)';

% Load in task variance for RNN 0
cdd1_yang = h_var_all(:,14);

figure; 
scatter(mean_cdd1var,cdd1_yang,30,'filled');
xlabel('Me');
ylabel('Yang');

[r,pval] = corr(mean_cdd1var,cdd1_yang); % r = 0.9773, p = 0


%% Check Performance for all RNNs

% Load in task variation for all RNNs
% Set up folder pathway
myFolder = 'C:\Users\JoshB\Documents\Projects\compositionality_cole\RNN\Performance';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*varRule*.mat');
theFiles = dir(filePattern); % Storing all filenames

rnn_storage(length(theFiles)) = struct('name',1,'task_var',1,'task_name',1); 
% Loop to load in the data, store name and values of zscore
for ii = 1:length(theFiles)
    baseFileName = theFiles(ii).name; 
    fullFileName = fullfile(theFiles(ii).folder, baseFileName); %making absolute path to file
    load(fullFileName); %loads in data
    rnn_storage(ii).name = baseFileName;
    rnn_storage(ii).task_var = h_var_all;
    rnn_storage(ii).task_name = task_names;
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*performance*.mat');
theFiles = dir(filePattern); % Storing all filenames

% Loop to load in the data, store name and values of zscore
for ii = 1:length(theFiles)
    baseFileName = theFiles(ii).name; 
    fullFileName = fullfile(theFiles(ii).folder, baseFileName); %making absolute path to file
    behaviour = load(fullFileName); %loads in data
    rnn_storage(ii).performance = cell2mat(struct2cell(behaviour));
end

% Plot behaviour for each network
figure; 
for ii = 1:length(rnn_storage)
    perf = rnn_storage(ii).performance;
    plot(perf');
    ylim([0 1]);
    xlim([0 320]);
    title(ii);
    pause
end


%% Calculate Task Variance for all tasks and networks

% Load in task variation for all RNNs
% Set up folder pathway
myFolder = 'C:\Users\JoshB\Documents\Projects\compositionality_cole\RNN\Activity';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '**','*neural_activity.mat'); %get list of .tsv files for all subfolders
filelist = dir(filePattern); % Storing all filenames

% Create list of task names
task_names = {'fdgo';'reactgo';'delaygo';'fdanti';'reactanti';'delayanti';
    'dm1';'dm2';'contextdm1';'contextdm2';'multidm';'delaydm1';'delaydm2';
    'contextdelaydm1';'contextdelaydm2';'multidelaydm';'dmsgo';'dmsnogo';
    'dmcgo';'dmcnogo'};

% Initialize the main structure array
rnn_data = struct();

% Process each file
for k = 1:length(filelist)
    fprintf('Processing file %d of %d: %s\n', k, length(filelist), filelist(k).name);
    
    % Get full file path
    fullFileName = fullfile(filelist(k).folder, filelist(k).name);
    
    % Extract RNN name from filename (assuming format: "X_neural_activity.mat")
    [~, fileName, ~] = fileparts(filelist(k).name);
    rnn_name = strrep(fileName, '_neural_activity', '');
    
    % Load the .mat file
    loaded_data = load(fullFileName);
    
    % Initialize structure for this RNN
    rnn_data(k).name = rnn_name;
    
    % Process each task
    for task_idx = 1:length(task_names)
        task_name = task_names{task_idx};
        activity_var = [task_name '_activity'];
        
        % Check if this task's activity exists in the loaded data
        if isfield(loaded_data, activity_var)                
            % Look for timing information and extract go_cue timing
            resp_start_field = [task_name '_response_start'];
            resp_start_time = loaded_data.(resp_start_field);
            activity_data = loaded_data.(activity_var);
            rnn_data(k).(activity_var) = activity_data(resp_start_time:end,:,:);
        else
            fprintf('Warning: %s not found in %s\n', activity_var, rnn_name);
            rnn_data(k).(activity_var) = [];
        end
    end
end

fprintf('Data loading complete! Loaded %d RNNs.\n', length(rnn_data));

% Save data
save('C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/rnn_data.mat', 'rnn_data', '-v7.3');

% Plot activity time series for each network for specific task
figure;
for ii = 1:length(rnn_data)
    task_name = 'fdgo_activity';
    activity_data = rnn_data(ii).(activity_var);
    mean_activity = squeeze(mean(activity_data,2));
    plot(mean_activity);
    pause;
end


% OPTION 1: Calculate Task Variance exactly like Yang
% Calculate task variance and average activity for each network and task
yang_var = [];
rnn_activity = [];
for ii = 1:length(rnn_data)
    fprintf('Calculating variance for RNN %d of %d \n', ii, length(rnn_data));
    % Load in activity for each task
    mean_taskvar = [];
    mean_activity = [];
    mean_ts = [];
    for task_idx = 1:length(task_names)
        task_name = task_names{task_idx};
        activity_var = [task_name '_activity'];
        activity_data = rnn_data(ii).(activity_var); % Time X Trial X Unit

        % Calculate task variance
        task_var = squeeze(var(activity_data,[],2));
        mean_taskvar(:,task_idx) = mean(task_var,1)'; % 256 X Task

        % Calculate average activity
        mean_activity(:,task_idx) = squeeze(mean(activity_data,[1 2]));
    end
    % Normalise measurements
    %taskvar_z = zscore(mean_taskvar,[],1);
    %activity_z = zscore(mean_activity,[],1);
    yang_var = cat(3,yang_var,mean_taskvar);
    rnn_activity = cat(3,rnn_activity,mean_activity);
end

figure;
for ii = 1:size(rnn_activity,3)
    temp_activity = rnn_activity(:,:,ii);
    plot(temp_activity');
    pause
end

% Calculate average ts per task and network
% Only take tasks with time series of 26 length (18/20)
% delaydm1 and delaydm2 only have 16 time points (Task 12, 13)
rnn_ts = [];
for ii = 1:length(rnn_data)
    task_ts = []; % Time X Unit X RNN
    for task_idx = 1:length(task_names)
        task_name = task_names{task_idx};
        activity_var = [task_name '_activity'];
        activity_data = rnn_data(ii).(activity_var);

        mean_ts = squeeze(mean(activity_data,2)); % Time X Unit
        if size(mean_ts,1) == 26
            task_ts = cat(3,task_ts,mean_ts);
        end
    end
    rnn_ts = cat(4,rnn_ts,task_ts);
end

% OPTION 2: Use same formula as Yang but at different scale -> more
% comparable to fMRI calculation
% Calculate task variance and average activity for each network across
% tasks
rnn_var = [];
for ii = 1:size(rnn_ts,4)
    net_ts = rnn_ts(:,:,:,ii); % Time X Unit X Task
    task_var = var(net_ts,[],3);
    mean_taskvar = mean(task_var,1)';
    % Standardise task variance per network
    mean_taskvar_z = zscore(mean_taskvar,[],1);
    rnn_var = cat(2,rnn_var,mean_taskvar_z);
end


%% Compare Yang variance with our simulated Task variance

% Map rnn_data to rnn_storage order
yang_var_ord = yang_var(:,:,[1 3 4 5 6 7 8 9 10 ...
    11 12 2 14 15 16 17 18 19 20 21 22 23 13 ...
    25 26 27 28 29 30 31 32 33 34 24 35 36 ...
    37 38 39 40]);

figure;
for ii = 1:40
    new_var = yang_var_ord(:,:,ii);
    original_var = rnn_storage(ii).task_var;
    scatter(original_var,new_var,30,'filled');
    xlabel('Original Variance');
    ylabel('New Variance');
    title(ii);
    pause
end

% Check distribution of variance
figure;
for ii = 1:size(rnn_var,2)
    histogram(rnn_var(:,ii));
    pause;
end


%% Identify Component and Recombination Units

total_var_z = zscore(rnn_var,[],1);

% Take out the 2 tasks with less time points
rnn_activity2 = rnn_activity(:,[1:11 14:end],:);
total_activity = squeeze(mean(rnn_activity2,2));
total_activity_z = zscore(total_activity,[],1);

% Identify Rule Dependent and Rule Independent units
spec_unit = total_var_z>1;
gen_unit = total_var_z<=1 & total_activity_z>1;


%% Calculate Dimensionality

% Load in rnn_ts.mat, spec_unit2.mat, gen_unit2.mat
nunit = 256;
gen_expvar = [];
spec_expvar = [];
dim_gen = [];
dim_spec = [];
for ii = 1:size(rnn_ts,4)
    ts_net = rnn_ts(:,:,:,ii); % Time X Unit X Task
    ts_rs = reshape(permute(ts_net,[1 3 2]),[],nunit); % Time*Task X Unit
    ts_gen = ts_rs(:,gen_unit(:,ii)==1);
    ts_spec = ts_rs(:,spec_unit(:,ii)==1);
    
    % Calculate PCA 
    [~,~,~,~,explained_gen,~] = pca(ts_gen);
    [~,~,~,~,explained_spec,~] = pca(ts_spec);

    gen_expvar = cat(2,gen_expvar,explained_gen(1:4)); % PCs X Network
    spec_expvar = cat(2,spec_expvar,explained_spec(1:4));

    dim_gen(ii,1) = (sum(explained_gen))^2/sum(explained_gen.^2);
    dim_spec(ii,1) = (sum(explained_spec))^2/sum(explained_spec.^2);
end

mean_genvar = mean(gen_expvar,2);
mean_specvar = mean(spec_expvar,2);
figure;
p1 = plot(mean_genvar(1:3),'-','LineWidth',3,'Color',[255, 28, 99]./255);
p1.Marker = '.';
p1.MarkerFaceColor = [255, 28, 99]./255;
p1.MarkerSize = 25;
hold on
p2 = plot(mean_specvar(1:3),'-','LineWidth',3,'Color',[110, 143, 169]./255);
p2.Marker = '.';
p2.MarkerFaceColor = [110, 143, 169]./255;
p2.MarkerSize = 25;
axis('tight');
ylim([0 100]);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];


% Figure 4d
% Boxplot comparing dimensionality between gen and spec
data1 = dim_gen;
data2 = dim_spec;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [255, 28, 99;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];

% Paired t-test for dimensionality between gen and spec regions per network
data1 = dim_gen;
data2 = dim_spec;
[h,p,ci,stats] = ttest(data1,data2);
% tstat = -3.3121, df = 39, sd = 0.7782, 95% CI [-0.6564 -0.1587], p =
% 0.0020

% Recalculate exact p-values
tval = -3.3121;
df = 39;
pval = 2 * (1 - tcdf(abs(tval), df));


%% Calculate Task Correlation

% Load in rnn_ts.mat, spec_unit.mat, gen_unit.mat, gen_id.mat, spec_id.mat
[ntime,nunit,ntask,nnet] = size(rnn_ts);

% For each network
mean_corrunit = [];
for ii = 1:nnet
    net_ts = rnn_ts(:,:,:,ii);
    % For each unit
    for jj = 1:nunit
        disp([ii jj]);
        unit_ts = squeeze(net_ts(:,jj,:));
        % Compare timeseries across all tasks
        corr_unit = corr(unit_ts);
        
        % Take lower triangle and average
        template = tril(ones(18)-eye(18));
        % Get the indices of lower triangle
        template_id = find(template);
        % Take lower triangle
        corr_unit_low = corr_unit(template_id);
        % Average correlation across mini-blocks
        mean_corrunit(ii,jj) = mean(corr_unit_low,1);
    end
end

% Separate out generalised and specialised regions
corr_gen = [];
corr_spec = [];
for ii = 1:size(mean_corrunit,1)
    net_corr = mean_corrunit(ii,:)';
    corr_gen = cat(1,corr_gen,net_corr(gen_unit(:,ii)==1));
    corr_spec = cat(1,corr_spec,net_corr(spec_unit(:,ii)==1));
end

% Boxplot
data1 = corr_gen;
data2 = corr_spec;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [194, 158, 85;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);

% Get network IDs for each group of units
gen_net = sum(gen_unit,1)';
gen_id = [];
for ii = 1:size(gen_net,1)
    id = repelem(ii,gen_net(ii),1);
    gen_id = cat(1,gen_id,id);
end

spec_net = sum(spec_unit,1)';
spec_id = [];
for ii = 1:size(spec_net,1)
    id = repelem(ii,spec_net(ii),1);
    spec_id = cat(1,spec_id,id);
end

% Generalised Linear Mixed Model
data1 = corr_gen;
data2 = corr_spec;
data = [data1; data2];
net_id = [gen_id; spec_id];
group_id = [2*ones(length(data1),1); 1*ones(length(data2),1)];
% Format variables
tbl = table(data,group_id,net_id);
tbl.group_id = categorical(tbl.group_id);
tbl.net_id = categorical(tbl.net_id);
glme = fitglme(tbl,'data ~ 1 + group_id + (1|net_id)');
% estimate = 0.15321, SE = 0.018102, tstat = 8.4639, df = 1255, pval < e-16,
% CI = [0.1177 0.18872];

% Recalculate exact p-values
tval = 8.4639;
df = 1255;
pval = 2 * (1 - tcdf(abs(tval), df));


% Figure 4f
% Take example generalised/specialised region timeseries across mini-blocks
% Plot both on same plot
unitG_ts = squeeze(rnn_ts(:,2,:,1));
unitS_ts = squeeze(rnn_ts(:,99,:,1));
figure;
plot(unitG_ts,'LineWidth',1.5,'Color',[255/255 28/255 99/255 0.75]);
hold on
plot(unitS_ts,'LineWidth',1.5,'Color',[110/255 143/255 169/255 0.75]);
%yline(0,'--','Color',[0 0 0],'LineWidth',1.5);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];


%% Calculate Integration

% Load in rnn_ts.mat, spec_unit.mat, gen_unit.mat

% Calculate correlation matrix per rnn
rnn_corr = [];
for ii = 1:size(rnn_ts,4)
    net_ts = rnn_ts(:,:,:,ii);
    net_ts_rs = reshape(permute(net_ts,[1 3 2]),[],256);
    %mean_ts = mean(net_ts,3);
    corr_ts = corr(net_ts_rs);
    rnn_corr = cat(3,rnn_corr,corr_ts);
end

% Visualise each correlation matrix
limits = [-1 1];
figure;
for ii = 1:size(rnn_corr,3)
    data = rnn_corr(:,:,ii);
    imagesc(data);
    ax = gca; ax.CLim = limits;
    axis('square');
    title(ii);
    pause
end

% Gamma parameter sweep
gamma_sweep = 0.5:0.1:2;
nGamma = 16;
nNodes = 256;
nIter = 100;
nNetworks = 40;

ci_iter = zeros(nNodes,nGamma,nIter,nNetworks);
q_iter = zeros(nGamma,nIter,nNetworks);

for ii = 1:nNetworks
    net_corr = rnn_corr(:,:,ii);
    for nn = 1:nIter
        for gg = 1:nGamma
            [ci_iter(:,gg,nn,ii),q_iter(gg,nn,ii)] = community_louvain(net_corr,gamma_sweep(gg),[],'negative_asym');
        end
        sprintf('%d %d',ii,nn)
    end
end

% Check Modularity stability for each network
mean_q = squeeze(mean(q_iter,2));
figure; plot(0.5:0.1:2,mean_q);

% Calculate agreement matrix per network
% Using gamma = 1;
net_ci = [];
for ii = 1:nNetworks
    ci_net = squeeze(ci_iter(:,6,:,ii));
    D_mat = agreement(ci_net);
    % Convert agreement matrix into probability values
    D_mat_prob = D_mat./100;
    % Calculate consensus partition
    % Tau chosen as low-value based on original paper
    ci_consensus = consensus_und(D_mat_prob,0.2,100);
    net_ci = cat(2,net_ci,ci_consensus); % Unit X Network
    disp(ii);
end

% Calculate participation coefficient with consensus clusters
nodes = 256;
part_gen = [];
part_spec = [];
part_all = [];
for ii = 1:size(rnn_corr,3)
    disp(ii);
    net_corr = rnn_corr(:,:,ii);
    part = participation_coef_sign(net_corr,net_ci(:,ii));
    part_gen = cat(1,part_gen,part(gen_unit(:,ii)==1));
    part_spec = cat(1,part_spec,part(spec_unit(:,ii)==1));
    part_all = cat(2,part_all,part);
end

% Smoothing out results by averaging per network
mean_partgen = [];
mean_partspec = [];
for ii = 1:length(unique(gen_id))
    mean_partgen(ii,1) = mean(part_gen(gen_id==ii),1);
    mean_partspec(ii,1) = mean(part_spec(spec_id==ii),1);
end

% Figure 4e
% Boxplot
data1 = mean_partgen;
data2 = mean_partspec;
g = [ones(length(data1),1); 2*ones(length(data2),1)];
RGB_color = [255, 28, 99;
    110, 143, 169]./255;
figure;
boxplot([data1;data2],g,'Symbol','','OutlierSize',16,'Colors',RGB_color);
hold on
group = [data1; data2];
RGB_color2 = [repmat(RGB_color(1,:),length(data1),1);
    repmat(RGB_color(2,:),length(data2),1)];
scatter(g,group,50,RGB_color2,'filled','jitter','on','jitterAmount',0.1, ...
    'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'MarkerEdgeColor','flat');
%ylim([0.65 0.78]);
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out', ...
    'XTickLabel','');
set(findobj(gca,'type','line'),'linew',3);
hAxes = findobj(gcf,'Type','axes');
hAxes.Position = [0.1300 0.1100 0.5057 0.8028];

% Generalised Linear Mixed Model
data1 = mean_partgen;
data2 = mean_partspec;
data = [data1; data2];
group_id = [2*ones(length(data1),1); 1*ones(length(data2),1)];
net_id = [repmat([1:40]',2,1)];
% Format variables
tbl = table(data,group_id,net_id);
tbl.group_id = categorical(tbl.group_id);
tbl.net_id = categorical(tbl.net_id);
glme = fitglme(tbl,'data ~ 1 + group_id + (1|net_id)');
% estimate = 0.051212, SE = 0.0093017, tstat = 5.5057, df = 78, pval = 4.541e-07,
% CI = [0.032694 0.069731];
